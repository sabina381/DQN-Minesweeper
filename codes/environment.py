import time
import os
import pickle
from IPython.display import display
import numpy as np
import pandas as pd
from typing import Tuple
from collections import deque
import random

#################################
directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

#################################
class Environment:
    def __init__(self, gridworld_size:Tuple, num_mine:int,
                    reward_dict:dict, done_dict:dict, color_dict:dict):

        self.gridworld_size = gridworld_size
        self.nrow, self.ncol = self.gridworld_size
        self.num_mine = num_mine

        # 그리드월드의 좌표(튜플)의 리스트
        # points == action space
        self.points = np.arange(self.nrow * self.ncol)
        self.num_actions = len(self.points)
        
        # 각 좌표의 주변 좌표 리스트 딕셔너리 생성
        self.neighbor_coords_dict = {}
        for idx in self.points:
            neighbor_coords = self.get_neighbor_coords(idx)
            self.neighbor_coords_dict[idx] = neighbor_coords

        # reward, done 딕셔너리
        self.reward_dict = reward_dict
        self.done_dict = done_dict

        # 지뢰 랜덤으로 배정
        self.mine_points = np.random.choice(self.points, self.num_mine, replace=False)

        # 정답 맵
        self.map_answer, self.mine_bool = self.make_answer_map()

        # state 맵
        self.present_state = np.full((self.nrow, self.ncol), -1) # BFS로 탐색하지 않은 부분을 -1로 초기화

        # 행동 횟수 카운트
        self.move_cnt = 0

        # reder에 사용하는 color map
        self.color_dict = color_dict

    
    def _coord_to_idx(self, x:int, y:int):
        return x * self.ncol + y


    def _idx_to_coord(self, idx):
        x, y = divmod(idx, self.ncol)
        return (x, y)


    def get_neighbor_coords(self, idx):
        '''
        입력 받은 idx 주변의 좌표 리스트를 반환한다.
        '''
        x, y = self._idx_to_coord(idx)
        
        neighbor_coords = []

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.nrow) and (0 <= ny < self.ncol):
                neighbor_coords.append((nx, ny))
        
        return neighbor_coords


    def make_answer_map(self):
        '''
        랜덤 배정된 지뢰 위치에 따라 지뢰찾기 맵을 생성한다.
        지뢰 위치: -2 / 지뢰 없는 위치: 주변 8개 칸의 지뢰 개수
        '''
        answer_map = np.full(shape=(self.nrow, self.ncol), fill_value=0)
        x, y = self._idx_to_coord(self.mine_points)
        answer_map[x, y] = -2
        mine_bool = (answer_map==-2)

        for idx in self.mine_points:
            neighbor_coords = self.neighbor_coords_dict[idx]
            for x, y in neighbor_coords:
                if mine_bool[x, y] == False:
                    answer_map[x, y] += 1

        return answer_map, mine_bool


    def bfs_minesweeper(self, clicked_idx:int):
        '''
        가려져있는 state에서 클릭한 좌표에 따라 현재 state를 열어서 반환한다.
        input : 클릭한 idx
        output : 클릭한 좌표에 따라서 열린 state (array)
        '''
        act_x, act_y = self._idx_to_coord(clicked_idx)
        queue = deque([(act_x, act_y)])

        result_state = self.present_state.copy()

        while queue:
            x, y = queue.popleft()

            if result_state[x, y] != -1:
                continue

            result_state[x, y] = self.map_answer[x, y]

            if self.map_answer[x,y] == 0:
                neighbor_coords = self.neighbor_coords_dict[self._coord_to_idx(x, y)]
                queue.extend(neighbor_coords)

        return result_state


    def check_guess(self, clicked_idx:int):
        '''
        input : clicked_idx(클릭한 좌표)
        output : 해당 좌표가 guess인지 (bool)
        클릭한 좌표가 guess인지 확인하는 함수
        클릭한 좌표 주변 8칸이 모두 열리지 않은 경우 guess
        '''
        if self.move_cnt == 0:
            return False

        unopened_cnt = 0
        neighbor_coords = self.neighbor_coords_dict[clicked_idx]

        for nx, ny in neighbor_coords:
            if self.present_state[nx, ny] == -1:
                unopened_cnt += 1

        if unopened_cnt == len(neighbor_coords):
            return True
        else:
            return False


    def move_first_mine(self, action_idx:int):
        '''
        에이전트가 첫 번째로 선택한 action이 지뢰인 경우
        해당 좌표의 지뢰를 다른 곳으로 옮기는 함수
        - input : action_idx - 좌표
        '''
        empty_points = np.setdiff1d(self.points, self.mine_points)
        new_mine = np.random.choice(empty_points, 1)

        self.mine_points = np.delete(self.mine_points, np.where(self.mine_points == action_idx))
        self.mine_points = np.append(self.mine_points, new_mine[0])

        # 정답 맵
        self.map_answer, self.mine_bool = self.make_answer_map()
        # state 맵
        self.present_state = np.full((self.nrow, self.ncol), -1)


    def step(self, action_idx:int):
        '''
        에이전트가 선택한 action에 따라 주어지는 next_state, reward, done
        - input : action_idx - 좌표
        - output : next_state, reward, done, clear
        '''
        x, y = self._idx_to_coord(action_idx)

        # 첫번째 action인 경우
        if self.move_cnt == 0 :
            if action_idx in self.mine_points:
                # 만약 처음 선택한 좌표에 지뢰가 있는 경우 옮기기
                self.move_first_mine(action_idx)

        # action에 따라 계산된 state
        next_state = self.bfs_minesweeper(action_idx)

        # ======
        # reward
        if action_idx in self.mine_points:
            # 지뢰
            reward = self.reward_dict['mine']
            done = self.done_dict['mine']
            clear = False

        elif np.sum(next_state == -1) == self.num_mine:
            # 클리어
            reward = self.reward_dict['clear']
            done = self.done_dict['clear']
            clear = True

        else :
            clear = False
            guess = self.check_guess(action_idx)

            if self.present_state[x,y] != -1:
                # 중복 행동
                reward = self.reward_dict['overlapped']
                done = self.done_dict['overlapped']

            elif guess:
                # 추측 행동
                reward = self.reward_dict['guess']
                done = self.done_dict['guess']

            else:
                # 좋은 행동
                reward = self.reward_dict['empty']
                done = self.done_dict['empty']

        # 현재 state 업데이트
        self.present_state = next_state
        self.move_cnt += 1

        return next_state, reward, done, clear


    def reset(self):
        '''
        reset game
        '''
        # 지뢰 랜덤으로 배정
        self.mine_points = np.random.choice(self.points, self.num_mine, replace=False)
        # 정답 맵
        self.map_answer, self.mine_bool = self.make_answer_map()
        # state 맵
        self.present_state = np.full((self.nrow, self.ncol), -1) # BFS로 탐색하지 않은 부분을 -1로 초기화

        self.move_cnt = 0


    def render(self, state):
        '''
        입력받은 state를 dataframe 형식으로 출력
        - 열리지 않은 칸: '.'
        '''
        render_state = np.full(shape=(self.nrow, self.ncol), fill_value=".")

        for idx in self.points:
            x, y = self._idx_to_coord(idx)
            if state[x,y] == -1:
                continue
            elif state[x,y] == -2:
                render_state[x,y] = "M"
            else:
                render_state[x,y] = state[x,y]

        render_state = pd.DataFrame(render_state)
        render_state = render_state.style.applymap(self.render_color)
        display(render_state)


    def render_answer(self):
        '''
        현재 environment의 지뢰찾기 맵을 dataframe 형식으로 출력
        '''
        render_state = np.full(shape=(self.nrow, self.ncol), fill_value=".")

        for idx in self.points:
            x, y = self._idx_to_coord(idx)
            if self.map_answer[x,y] == -2:
                render_state[x,y] = "M"
            else:
                render_state[x,y] = str(self.map_answer[x,y])

        render_state = pd.DataFrame(render_state)
        render_state = render_state.style.applymap(self.render_color)
        display(render_state)


    def render_color(self, var):
        return f"color: {self.color_dict[var]}"


    # def samples(self, num:int):
    #     sample_mine_points = []

    #     for i in range(num):
    #         self.mine_points = np.random.choice(self.points, self.num_mine, replace=False)
    #         sample_mine_points.append(self.mine_points)

    #     return sample_mine_points

    # def train_reset(self, samples:np.array):
    #     self.mine_points = random.sample(samples, 1)[0]
    #     # 정답 맵
    #     self.map_answer, self.mine_bool = self.make_answer_map()
    #     # state 맵
    #     self.present_state = np.full((self.nrow, self.ncol), -1) # BFS로 탐색하지 않은 부분을 -1로 초기화

    #     self.move_cnt = 0


    # def check_18_up(self, state):
    #     if np.sum(state != -1) >= 18:
    #         return True
    #     else:
    #         return False

