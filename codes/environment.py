import time
import os
import pickle
from IPython.display import display
import numpy as np
import pandas as pd
from typing import Tuple
from collections import deque
import random


class Environment:
    def __init__(self, gridworld_size:Tuple, num_mine:int,
                    reward_dict:dict, done_dict:dict):

        self.gridworld_size = gridworld_size
        self.nrow, self.ncol = self.gridworld_size
        self.num_mine = num_mine

        # 그리드월드의 좌표(튜플)의 리스트
        # points == action space
        self.points = np.arange(self.nrow * self.ncol)
        self.num_actions = len(self.points)

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


    def make_answer_map(self):
        '''
        랜덤 배정된 지뢰 위치에 따라 지뢰찾기 맵 생성
        지뢰 위치: -2 / 지뢰 없는 위치: 주변 8개 칸의 지뢰 개수 표시
        '''
        answer_map = np.full(shape=(self.nrow, self.ncol), fill_value=0)
        x, y = np.divmod(self.mine_points, self.ncol)
        answer_map[x, y] = -2
        mine_bool = (answer_map==-2)

        # 주변 8칸 좌표
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for idx in self.points: # 모든 좌표를 탐색
            if idx in self.mine_points:
                continue
            else:
                x, y = divmod(idx, self.ncol)
                for dx, dy in directions:   # 주변 8칸을 탐색해 지뢰가 있을 때마다 +1
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.nrow and 0 <= ny < self.ncol: # 가장자리 탐색 금지 처리
                        if mine_bool[nx, ny]:
                            answer_map[x, y] += 1

        return answer_map, mine_bool


    def bfs_minesweeper(self, clicked_idx:int):
        '''
        input : 클릭한 idx
        output : 클릭한 좌표에 따라서 열린 맵(array)
        가려져있는 맵에서 클릭한 좌표에 따라 맵을 열어주는 함수
        '''
        act_x, act_y = divmod(clicked_idx, self.ncol)
        queue = deque([(act_x, act_y)])

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

        result = self.present_state.copy()

        while queue:
            x, y = queue.popleft()

            if result[x, y] != -1:
                continue

            result[x, y] = self.map_answer[x, y]

            if self.map_answer[x,y] == 0:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.nrow and 0 <= ny < self.ncol and result[nx, ny] == -1:
                        queue.append((nx,ny)) # 좌표 -> 인덱스 역산

        return result


    def check_guess(self, clicked_idx:int):
        '''
        input : clicked_idx(클릭한 좌표)
        output : 해당 좌표가 guess인지 (bool)
        클릭한 좌표가 guess인지 확인하는 함수
        클릭한 좌표 주변 8칸이 모두 열리지 않은 경우 guess
        '''
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
        x, y = divmod(clicked_idx, self.ncol)
        result = 0

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.nrow and 0 <= ny < self.ncol:
                if self.present_state[nx, ny] == -1:
                    result += 1

        if result == 8:
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
        x, y = divmod(action_idx, self.ncol)

        # 첫번째 action인 경우
        if self.move_cnt == 0 :
            if action_idx in self.mine_points:
                # 만약 start 좌표에 지뢰가 있는 경우 옮기기
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
        # 지뢰 랜덤으로 배정
        self.mine_points = np.random.choice(self.points, self.num_mine, replace=False)
        # 정답 맵
        self.map_answer, self.mine_bool = self.make_answer_map()
        # state 맵
        self.present_state = np.full((self.nrow, self.ncol), -1) # BFS로 탐색하지 않은 부분을 -1로 초기화


    def render(self, state):
        render_state = np.full(shape=(self.nrow, self.ncol), fill_value=".")

        for idx in self.points:
            x, y = divmod(idx, self.ncol)
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
        render_state = np.full(shape=(self.nrow, self.ncol), fill_value=".")

        for idx in self.points:
            x, y = divmod(idx, self.ncol)
            if self.map_answer[x,y] == -2:
                render_state[x,y] = "M"
            else:
                render_state[x,y] = str(self.map_answer[x,y])

        render_state = pd.DataFrame(render_state)
        render_state = render_state.style.applymap(self.render_color)
        display(render_state)


    def render_color(self, var):
        color = {'0':'black', '1':"skyblue", '2':'lightgreen', '3':'red', '4':'violet', '5':'brown',
                 '6':'turquoise', '7':'grey', '8':'black', 'M':'white', '.':'black'}
        return f"color: {color[var]}"


    def samples(self, num:int):
        sample_mine_points = []

        for i in range(num):
            self.mine_points = np.random.choice(self.points, self.num_mine, replace=False)
            sample_mine_points.append(self.mine_points)

        return sample_mine_points

    def train_reset(self, samples:np.array):
        self.mine_points = random.sample(samples, 1)[0]
        # 정답 맵
        self.map_answer, self.mine_bool = self.make_answer_map()
        # state 맵
        self.present_state = np.full((self.nrow, self.ncol), -1) # BFS로 탐색하지 않은 부분을 -1로 초기화


    def check_18_up(self, state):
        if np.sum(state != -1) >= 18:
            return True
        else:
            return False

