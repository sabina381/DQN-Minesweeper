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

########################################
with open('experiment8_model_3.pkl', 'rb') as f:
    model_param = pickle.load(f)
################################
env = Environment(grid_size, num_mines)
agent = DQN_Agent(grid_size, env.points, num_mines)

agent.model.load_state_dict(model_param)
agent.target_model.load_state_dict(model_param)
agent.epsilon = 0

rewards_list = []
clear_list = []
cnt_list = []

avg_rewards_list = []
avg_clear_list = []
avg_count_list = []

mid_rewards_list = []
mid_clear_list = []
mid_cnt_list = []

for episode in range(EPISODES):
    # reset
    env.reset()
    state = env.present_state.copy()
    done = False
    clear = False
    total_reward = 0
    cnt = 0
    loss = 0
    agent.action = env.points.copy()

    # 게임 종료까지 반복
    while not done:
        cnt+=1

        state = env.present_state.copy()
        action = agent.get_action(state)
        next_state, reward, done, clear = env.step(action)
        total_reward += reward

        # count 제한: 100
        if cnt > 81:
            done = True

        if done or clear:
            break

    # 평가지표
    rewards_list.append(total_reward)
    avg_rewards_list.append(np.mean(rewards_list[-PRINT_EVERY:]))
    mid_rewards_list.append(np.median(rewards_list[-PRINT_EVERY:]))

    clear_list.append(clear)
    avg_clear_list.append(np.mean(clear_list[-PRINT_EVERY:]))
    mid_clear_list.append(np.median(clear_list[-PRINT_EVERY:]))

    cnt_list.append(cnt)
    avg_count_list.append(np.mean(cnt_list[-PRINT_EVERY:]))
    mid_cnt_list.append(np.median(cnt_list[-PRINT_EVERY:]))

    if ((episode+1) % SAVE_EVERY == 0) or episode+1 == EPISODES:
        # 시각화 저장
        df = pd.DataFrame()
        df['rewards'] = rewards_list
        df['avg_rewards'] = avg_rewards_list
        df['mid_rewards'] = mid_rewards_list
        df['clear'] = clear_list
        df['avg_clear'] = avg_clear_list
        df['mid_clear'] = mid_clear_list
        df['cnt'] = cnt_list
        df['avg_count'] = avg_count_list
        df['mid_count'] = mid_cnt_list

        with open('experiment8_visualizing_test2.pkl', 'wb') as f:
            pickle.dump(df, f)


    if (episode+1) % PRINT_EVERY == 0:
        print(f"[{episode+1}/{EPISODES}]", end=" | ")
        print(f"avg clear: {round(np.mean(clear_list[-PRINT_EVERY*10:]), 3)}", end=" | ")
        print(f"cnt: {round(np.mean(cnt_list[-PRINT_EVERY*10:]), 3)}/{round(np.median(cnt_list[-PRINT_EVERY*10:]), 3)}", end=" | ")
        print(f"Reward: {round(np.mean(rewards_list[-PRINT_EVERY*10:]), 3)}/{round(np.median(rewards_list[-PRINT_EVERY*10:]), 3)}", end="\n")
        env.render(env.present_state)

print(f"Test completed. avg win rate: {round(np.mean(clear_list), 3)}")