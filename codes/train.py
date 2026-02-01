import time
import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from collections import deque
import copy
from scipy.special import softmax
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from environment import Environment
from dqn_agent import DQN_Agent
########################################
class Trainer:
    def __init__(self, 
                # Trainer
                FOLDER_NAME:str, PATH:str, EPISODES:int, 
                UPDATE_TARGET_EVERY:int, PRINT_EVERY:int, SAVE_EVERY:int,
                TRAIN_START:int, LEARN_MIN:float, LEARN_DECAY:float, LEARN_EPOCH:float,
                # Environment
                GRIDWORLD_SIZE:Tuple, NUM_MINE:int, REWARD_DICT:dict, DONE_DICT:dict, COLOR_DICT:dict,
                # DQN_Agent
                STATE_TYPE:str, EPSILON:float, EPSILON_DECAY:float, EPSILON_MIN:float, BATCH_SIZE:int, 
                GAMMA:float, MEM_SIZE:int, LEARN_MAX:float, CONV_UNITS:int, DEVICE, LOSS_FUNC, OPTIMIZER
                )

        self.env = Environment(GRIDWORLD_SIZE= GRIDWORLD_SIZE, 
                            NUM_MINE= NUM_MINE, 
                            REWARD_DICT= REWARD_DICT, 
                            DONE_DICT= DONE_DICT, 
                            COLOR_DICT= COLOR_DICT)

        self.agent = DQN_Agent(STATE_SIZE= GRIDWORLD_SIZE, 
                            STATE_TYPE= STATE_TYPE, 
                            NUM_MINE= NUM_MINE, 
                            EPSILON= EPSILON, 
                            EPSILON_DECAY= EPSILON_DECAY, 
                            EPSILON_MIN= EPSILON_MIN, 
                            BATCH_SIZE= BATCH_SIZE, 
                            GAMMA= GAMMA, 
                            MEM_SIZE= MEM_SIZE, 
                            LEARN_MAX= LEARN_MAX, 
                            CONV_UNITS= CONV_UNITS, 
                            DEVICE= DEVICE, 
                            LOSS_FUNC= LOSS_FUNC, 
                            OPTIMIZER= OPTIMIZER)

        self.cnt_limit = self.env.nrow * self.env.ncol - self.env.num_mine
        self.train_start = TRAIN_START

        self.lr_min = LEARN_MIN
        self.lr_decay = LEARN_DECAY
        self.lr_epoch = LEARN_EPOCH

        self.folder_name = FOLDER_NAME
        self.path = f"{PATH}/{FOLDER_NAME}"
        self.episodes = EPISODES
        self.update_target_every = UPDATE_TARGET_EVERY
        self.print_every = PRINT_EVERY
        self.save_every = SAVE_EVERY

        self.current_epi = 0

        self.rewards_list = []
        self.avg_rewards_list = []
        self.mid_rewards_list = []

        self.clear_list = []
        self.avg_clear_list = []
        self.mid_clear_list = []

        self.cnt_list = []
        self.avg_cnt_list = []
        self.mid_cnt_list = []

        self.loss_list = []
        self.avg_loss_list = []
        self.mid_loss_list = []

        self.lr_list = []
                

    def load_model(self):
        file_path = f"{self.path}/{self.folder_name}_model.pkl"

        with open(file_path, 'rb') as f:
            model_param = pickle.load(f)

        # reset 메서드 만들어서 거기로 뺄까?
        self.agent.model.load_state_dict(model_param)
        self.agent.target_model.load_state_dict(model_param)
        print(f"Model loaded from \'{file_path}\'")


    def save_model(self):
        self.agent.model.to("cpu")

        file_path = f"{self.path}/{self.folder_name}_model.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(self.agent.model.state_dict(), f)
        
        print(f"Model saved to \'{file_path}\'")

    
    def save_memory(self):
        file_path = f"{self.path}/{self.folder_name}_memory.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(self.agent.memory, f)


    def reset(self):
        self.env = self.env.reset()
        self.agent = self.agent.reset()
        self.load_model()


    def game_reset(self):
        self.env.reset()
        state = self.env.present_state.copy()
        done = False
        clear = False
        total_reward = 0
        cnt = 0
        loss = 0

        self.agent.action_space = self.env.points.copy()

        return state, done, clear, total_reward, cnt, loss

    
    def check_cnt_limit(self, cnt):
        if cnt > self.cnt_limit:
            return True
        else:
            return False


    def update_train_log(self, total_reward, clear, cnt, loss):
        self.rewards_list.append(total_reward)
        self.avg_rewards_list.append(np.mean(self.rewards_list[-self.print_every:]))
        self.mid_rewards_list.append(np.median(self.rewards_list[-self.print_every:]))

        self.clear_list.append(clear)
        self.avg_clear_list.append(np.mean(self.clear_list[-self.print_every:]))
        self.mid_clear_list.append(np.median(self.clear_list[-self.print_every:]))

        self.cnt_list.append(cnt)
        self.avg_cnt_list.append(np.mean(self.cnt_list[-self.print_every:]))
        self.mid_cnt_list.append(np.median(self.cnt_list[-self.print_every:]))

        self.loss_list.append(loss)
        self.avg_loss_list.append(np.mean(self.loss_list[-self.print_every*10:]))
        self.mid_loss_list.append(np.median(self.loss_list[-self.print_every*10:]))

        self.lr_list.append(self.agent.optimizer.param_groups[0]['lr'])

    
    def save_train_log(self):
        df = pd.DataFrame()
        df['rewards'] = self.rewards_list
        df['avg_rewards'] = self.avg_rewards_list
        df['mid_rewards'] = self.mid_rewards_list
        df['clear'] = self.clear_list
        df['avg_clear'] = self.avg_clear_list
        df['mid_clear'] = self.mid_clear_list
        df['cnt'] = self.cnt_list
        df['avg_cnt'] = self.avg_cnt_list
        df['mid_cnt'] = self.mid_cnt_list
        df['loss'] = self.loss_list
        df['avg_loss'] = self.avg_loss_list
        df['mid_loss'] = self.mid_loss_list
        df['lr'] = self.lr_list

        file_path = f"{self.path}/{self.folder_name}_train_log.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(df, f)
        
        print(f"Eval idx saved to \'{file_path}\'")


    def print_train_log(self):
        print("="*30, end="\n")
        print(f"[{self.current_epi+1}/{self.episodes}] epsilon: {round(self.agent.epsilon, 5)} | lr: {round(self.agent.lr, 5)}", end="\n\t")
        print(f"- Avg Clear: {round(np.mean(self.clear_list[-self.print_every:]), 3)}", end="\n\t")
        print(f"- Cnt: {round(np.mean(self.cnt_list[-self.print_every:]), 3)}/{round(np.median(self.cnt_list[-self.print_every:]), 3)}", end="\n\t")
        print(f"- Reward: {round(np.mean(self.rewards_list[-self.print_every:]), 3)}/{round(np.median(self.rewards_list[-self.print_every:]), 3)}", end="\n\t")
        print(f"- Loss: {round(np.mean(self.loss_list[-self.print_every:]), 3)}/{round(np.median(self.loss_list[-self.print_every:]), 3)}", end="\n")
        self.env.render(self.env.present_state)


    def train(self):
        self.reset()

        for episode in range(self.episodes):
            self.current_epi += 1
            # reset
            state, done, clear, total_reward, cnt, loss = self.game_reset()

            # 게임 종료까지 반복
            while not done:
                cnt+=1

                state = self.env.present_state.copy()
                action = self.agent.get_action(state)
                next_state, reward, done, clear = self.env.step(action)
                total_reward += reward

                # count 제한 : 전체 칸 수 - 지뢰 개수
                done = self.check_cnt_limit()

                # replay memory에 샘플 저장
                self.agent.append_sample(state, action, reward, next_state, done, clear)

                # 학습
                if len(self.agent.memory) > self.train_start:
                    loss = self.agent.train_model()
                    loss = loss.item()

                if done or clear:
                    break

            # 평가지표
            self.update_train_log(total_reward, clear, cnt, loss)

            # 타깃 모델 업데이트
            if episode % self.update_target_every == 0:
                self.agent.update_target_model()

            # lr 조절
            if self.lr_epoch > 0:
                if (episode+1) % self.lr_epoch == 0:
                    lr = self.agent.optimizer.param_groups[0]['lr'] * self.lr_decay
                    self.agent.optimizer.param_groups[0]['lr'] = max(lr, self.lr_min)

            if ((episode+1) % self.save_every == 0) or ((episode+1) == self.episodes):
                self.save_model()
                self.save_train_log()

            if (episode+1) % self.print_every == 0:
                self.print_train_log()

        print(f"Test completed. total avg win rate: {round(np.mean(self.clear_list), 3)}")


    def visualize_logs(self):

