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

from net import *

########################################
class DQN_Agent:
    def __init__(self, state_size:Tuple, action, num_mine,
                EPSILON, EPSILON_DECAY, EPSILON_MIN, 
                BATCH_SIZE, GAMMA, MEM_SIZE, LEARN_MAX,
                CONV_UNITS, DEVICE, LOSS_FUNC, OPTIMIZER):

        self.num_mine = num_mine
        self.state_size = state_size # 튜플로 입력받음
        self.action = torch.tensor(action)
        self.q_values = torch.zeros(self.action.shape, dtype=torch.float32)

        # 하이퍼파라미터
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA

        # 추가
        self.nrow = state_size[0]
        self.ncol = state_size[1]
        self.n = self.nrow*self.ncol

        # 리플레이 메모리
        self.memory = deque(maxlen = MEM_SIZE)

        # 모델과 타깃 모델 생성
        self.device = DEVICE

        # model, target model gpu 올리고 초기화
        self.model = Net(state_size, len(action), CONV_UNITS).to(self.device)
        self.target_model = Net(state_size, len(action), CONV_UNITS).to(self.device)
        self.update_target_model()
        self.loss_func = LOSS_FUNC

        self.optimizer = OPTIMIZER(self.model.parameters, lr=LEARN_MAX)


    def update_target_model(self):
        # 타깃 모델을 업데이트 하는 함수
        self.target_model.load_state_dict(self.model.state_dict())


    def append_sample(self, state, action, reward, next_state, done, clear):
        # 샘플을 리플레이 메모리에 저장하는 함수
        self.memory.append((state, action, reward, next_state, done, clear))


    def get_action(self, state):
        '''
        입력받은 state에 따라 action을 선택한다. (학습하지 않음)
        '''
        state = torch.tensor(state).to(self.device)

        if np.random.rand() <= self.epsilon:  # 엡실론 무작위 탐색
            act = random.choice(self.action)

        else :
            state = state.unsqueeze(0).to(dtype = torch.float32)
            state = state.unsqueeze(0)

            # 정규화
            state = state / 8

            with torch.no_grad():
                q_values = self.model(state).flatten().to("cpu")
                max_idx = torch.argmax(q_values)
                act = max_idx.item()

                self.q_values = q_values

        return act


    def train_model(self):
        '''
        리플레이 메모리에서 무작위로 추출한 배치로 학습한다.
        '''
        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        # 추출한 샘플 tensor로 가져오기
        # 상태 정규화 포함
        states = torch.tensor([sample[0]/8 for sample in mini_batch], dtype=torch.float32).to(self.device).reshape(-1,1,self.nrow,self.ncol)
        actions = torch.tensor([sample[1] for sample in mini_batch], dtype=torch.long).to(self.device).reshape(-1,1)
        rewards = torch.tensor([sample[2] for sample in mini_batch], dtype=torch.float32).to(self.device).reshape(-1,1)
        next_states = torch.tensor([sample[3]/8 for sample in mini_batch], dtype=torch.float32).to(self.device).reshape(-1,1,self.nrow,self.ncol)
        dones = torch.tensor([sample[4] for sample in mini_batch], dtype=torch.long).to(self.device).reshape(-1,1)
        clears = torch.tensor([sample[5] for sample in mini_batch], dtype=torch.bool).reshape(-1,1)

        # 현재 상태에 대한 모델의 큐함수
        predicts = self.model(states) # 현재 상태의 좌표를 준다
        one_hot_action = F.one_hot(actions, self.n).to(self.device)
        predicts = torch.sum(one_hot_action*predicts, axis=1)

         # Q(s,a) 값을 예측값으로 사용 - (batch, action_space.n)
        pred_q_values = self.model(states).gather(1, actions) # action idx의 데이터만 꺼냄

        # target 값 계산 : reward + gamma * Q(s',a')
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1).values.reshape(-1,1)
            target_q_values = rewards + (torch.ones(next_q_values.shape, device=self.device) - dones) * self.gamma * next_q_values

        # 오류 함수를 줄이는 방향으로 모델 업데이트
        loss = self.loss_func(pred_q_values, target_q_values)

        self.optimizer.zero_grad() # 반복 때마다 기울기를 새로 계산해야하므로 기울기 초기화
        loss.backward() # 역전파 알고리즘 계산
        self.optimizer.step() # 계산한 기울기를 adam 알고리즘에 맞추어 가중치를 수정

        # 엡실론 감소
        self.epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss