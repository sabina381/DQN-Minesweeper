import pickle
from pathlib import Path
import numpy as np
from typing import Tuple

from environment import Environment
from dqn_agent import DQN_Agent

from utils import visualize_test_log, visualize_episodes, visualize_state_and_q
from log import Log

from trainer import Trainer
########################

class Tester(Trainer):
    def __init__(self, 
                # Trainer
                FOLDER_NAME:str, PATH:str, EPISODES:int, VALID_EPISODES:int, LAG:int, MODEL_CRITERIA:int,
                UPDATE_TARGET_EVERY:int, PRINT_EVERY:int, SAVE_EVERY:int, VALID_EVERY:int,
                TRAIN_START:int, LEARN_MIN:float, LEARN_DECAY:float, LEARN_EPOCH:float,
                # Environment
                GRIDWORLD_SIZE:Tuple, NUM_MINE:int, REWARD_DICT:dict, DONE_DICT:dict, COLOR_DICT:dict, FIRST_MINE:bool,
                # DQN_Agent
                STATE_TYPE:str, EPSILON:float, EPSILON_DECAY:float, EPSILON_MIN:float, BATCH_SIZE:int, 
                GAMMA:float, MEM_SIZE:int, LEARN_MAX:float, CONV_UNITS:int, DEVICE, LOSS_FUNC, OPTIMIZER
                ): 
        super().__init__(FOLDER_NAME, PATH, EPISODES, VALID_EPISODES, LAG, MODEL_CRITERIA,
                        UPDATE_TARGET_EVERY, PRINT_EVERY, SAVE_EVERY, VALID_EVERY,
                        TRAIN_START, LEARN_MIN, LEARN_DECAY, LEARN_EPOCH,
                        # Environment
                        GRIDWORLD_SIZE, NUM_MINE, REWARD_DICT, DONE_DICT, COLOR_DICT, FIRST_MINE,
                        # DQN_Agent
                        STATE_TYPE, EPSILON, EPSILON_DECAY, EPSILON_MIN, BATCH_SIZE, 
                        GAMMA, MEM_SIZE, LEARN_MAX, CONV_UNITS, DEVICE, LOSS_FUNC, OPTIMIZER)

        # 경로 생성
        self.create_path()
        self.total_episodes = 0
        self.mode = "test"

        self.test_log = Log(MODE= 'test', FOLDER_NAME= self.folder_name, PATH=self.path_dict['logs'])
        self.cur_epi = 0


    def reset(self):
        self.env.reset()
        self.agent.reset()

        self.total_episodes = 0
        self.test_log.reset()
        self.cur_epi = 0


    def create_path(self):
        '''
        필요한 폴더, 파일 경로 생성
        '''
        self.path_dict = ({'log_message': Path(f"{self.path}/test/{self.folder_name}_test_log.txt"),
                        'logs': Path(f"{self.path}/test/logs"),
                        'graph': Path(f"{self.path}/test/graphs"),
                        'game_imgs': Path(f"{self.path}/test/game_imgs"),
                        'model': Path(f"{self.path}/models")})

        self.path_dict['log_message'].parent.mkdir(parents=True, exist_ok=True)
        self.path_dict['logs'].mkdir(parents=True, exist_ok=True)
        self.path_dict['graph'].mkdir(parents=True, exist_ok=True)
        self.path_dict['game_imgs'].mkdir(parents=True, exist_ok=True)

        # 학습 로그 파일에 제목 출력
        if not Path(self.path_dict['log_message']).exists():
            with open(self.path_dict['log_message'], "w") as f:
                f.write(f"Test Log for {self.folder_name}\n")
                f.write("="*50 + "\n")
            
        # 학습 로그 파일에 모든 경로 출력
        log_str = "< path dict >\n"

        for item in self.path_dict.items():
            log_str += str(item) + "\n"

        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)

    def _print_log(self):
        log_str = "\n"
        log_str += f"[{self.cur_epi}/{self.test_total_episodes}]" + " "
        log_str += f"- Avg Clear: {round(np.mean(self.test_log.clear_list[-self.print_every:]), 3)}" + " "
        log_str += f"- Cnt: {round(np.mean(self.test_log.cnt_list[-self.print_every:]), 3)}/{round(np.median(self.test_log.cnt_list[-self.print_every:]), 3)}" + " "
        log_str += f"- Reward: {round(np.mean(self.test_log.reward_list[-self.print_every:]), 3)}/{round(np.median(self.test_log.reward_list[-self.print_every:]), 3)}" + " "
        log_str += f"- RPC: {round(np.mean(self.test_log.rpc_list[-self.print_every:]), 3)}/{round(np.median(self.test_log.rpc_list[-self.print_every:]), 3)}" + " "
        
        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)


    def test(self, model, num_episodes, test_num):
        '''
        주어진 모델로 test를 실행하는 함수
        '''
        self.reset()
        self.test_total_episodes = num_episodes
        log_str = ""
        print("Start test")

        super().load_model(model)

        for episode in range(self.test_total_episodes):
            self.cur_epi += 1
            # reset
            state, done, clear, total_reward, cnt, _ = super()._game_reset()

            # 게임 종료까지 반복
            while not done:
                cnt+=1
                action = self.agent.get_action_test(state, model)
                state, reward, done, clear = self.env.step(action)
                total_reward += reward

                # count 제한 : 전체 칸 수 - 지뢰 개수
                if not done:
                    done = super()._check_cnt_limit(cnt)

                if done:
                    break

            # 평가지표
            self.test_log.update_logs([total_reward, clear, cnt])

            if (episode+1) % self.print_every == 0:
                self._print_log()

        self.test_log.save_logs()
        log_str += f"======== Test result ======== \n\t\
                    Total episode: {self.test_total_episodes}\n\t\
                    Average win rate: {round(np.mean(self.test_log.clear_list), 3)}\n\t\
                    Average reward: {round(np.mean(self.test_log.reward_list), 3)}\n\t\
                    Average count: {round(np.mean(self.test_log.cnt_list), 3)}\n\t\
                    Average RPC: {round(np.mean(self.test_log.rpc_list), 3)}\n"

        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)

        log_data = self.test_log.load_logs()
        visualize_test_log(log_data, lag=self.lag, save_path=f"{self.path_dict['graph']}/test_{test_num}")

    
    def render_game(self, model, num_episode, heatmap=False):
        self.reset()
        print("Game start")

        super().load_model(model)

        for epi in range(num_episode):
            # reset
            state, done, clear, total_reward, cnt, _ = super()._game_reset()
            episode_data = []

            # 게임 종료까지 반복
            while not done:
                cnt+=1
                action = self.agent.get_action_test(state, model)
                state, reward, done, clear = self.env.step(action)
                total_reward += reward

                # count 제한 : 전체 칸 수 - 지뢰 개수
                if not done:
                    done = super()._check_cnt_limit(cnt)

                q_value = self.agent.q_values
                episode_data.append((state, reward, action, clear, q_value))

                if done:
                    break

            if clear:
                print("Game clear!")
            else:
                print("Game over")
            
            if heatmap:
                visualize_state_and_q(episode_data=episode_data, save_path=f"{self.path_dict['game_imgs']}/game_heatmap_{epi:03}")
            else:
                visualize_episodes(episode_data=episode_data, save_path=f"{self.path_dict['game_imgs']}/_{epi:03})")
            