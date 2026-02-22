import numpy as np
from typing import Tuple
from collections import deque
import random

import torch
import torch.nn.functional as F

from net import *
from scaling import *

########################################
class DQN_Agent:
    def __init__(self, STATE_SIZE:Tuple, STATE_TYPE:str, NUM_MINE:int,
                EPSILON:float, EPSILON_DECAY:float, EPSILON_MIN:float, 
                BATCH_SIZE:int, GAMMA:float, MEM_SIZE:int, 
                LEARN_MAX:float,
                CONV_UNITS:int, 
                DEVICE, LOSS_FUNC, OPTIMIZER):

        self.num_mine = NUM_MINE
        self.state_size = STATE_SIZE # state 사이즈를 튜플로 입력받음
        self.state_type = STATE_TYPE # "original", "one-hot", "normalization"
        # state의 행, 열 개수
        self.nrow = STATE_SIZE[0]
        self.ncol = STATE_SIZE[1]

        # 현재 가능한 action idxs
        self.action_space = torch.tensor(np.arange(self.nrow * self.ncol))
        self.num_actions = len(self.action_space)
        # 현재 큐함수
        self.q_values = torch.zeros(self.action_space.shape, dtype=torch.float32)

        # 하이퍼파라미터
        self.epsilon = EPSILON
        self.epsilon_init = EPSILON # fix. 초기값 보존
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        self.gamma = GAMMA

        self.lr = LEARN_MAX
        self.lr_init = LEARN_MAX    # fix. 초기값 보존

        self.batch_size = BATCH_SIZE
        self.conv_units = CONV_UNITS

        self.device = DEVICE    # device: 'mps' or 'cpu' or 'cuda'

        # 리플레이 메모리
        self.mem_size = MEM_SIZE
        self.memory = deque(maxlen = self.mem_size)

        # 모델과 타깃 모델 생성
        # model, target model, best model gpu 올리고 초기화
        # best model: 학습 중 valid에 따라 선정된 최고 성능 모델. valid마다 최고 성능 모델이 갱신되면 업데이트됨.
        # state_type에 따라 신경망 결정
        if self.state_type == "one-hot":    # one-hot 방식
            self.model = NetOneHot(self.state_size, self.num_actions, self.conv_units).to(self.device)
            self.target_model = NetOneHot(self.state_size, self.num_actions, self.conv_units).to(self.device)
            self.best_model = NetOneHot(self.state_size, self.num_actions, self.conv_units).to(self.device)
        else:   # original, normalization 방식
            self.model = Net(self.state_size, self.num_actions, self.conv_units).to(self.device)
            self.target_model = Net(self.state_size, self.num_actions, self.conv_units).to(self.device)
            self.best_model = Net(self.state_size, self.num_actions, self.conv_units).to(self.device)
        self.update_target_model()

        # loss function, optimizer 설정
        self.loss_func = LOSS_FUNC
        self.optimizer_type = OPTIMIZER
        self.optimizer = OPTIMIZER(self.model.parameters(), lr=self.lr_init)


    def update_target_model(self):
        '''
        target model을 업데이트 한다.
        '''
        self.target_model.load_state_dict(self.model.state_dict())


    def append_sample(self, state, action, reward, next_state, done, clear):
        '''
        게임 sample 1개를 replay memory에 저장한다.
        '''
        self.memory.append((state, action, reward, next_state, done, clear))

    
    def change_state_type(self, state):
        '''
        state_type에 따라 state를 정규화한다.
        '''
        assert state.dim() == 4, f"The tensor must be 4-dimensional. 현재 차원: {state.dim()}"

        if self.state_type == "original":
            norm_state = state  # torch.Size([batch, 1, 9, 9])
        
        elif self.state_type == "normalization":
            norm_state = mine_normalize(state)  # torch.Size([batch, 1, 9, 9])

        elif self.state_type == "one-hot":
            norm_state = one_hot_scaling(state) # torch.Size([batch, 11, 9, 9])
        
        return norm_state


    def get_action(self, state):
        '''
        입력받은 state에 따라 action을 선택한다.
        '''
        state = torch.tensor(state).to(self.device)

        if np.random.rand() <= self.epsilon:  # 엡실론 무작위 탐색
            act = random.choice(self.action_space)

        else :
            state = state.unsqueeze(0).to(dtype = torch.float32)
            state = state.unsqueeze(0)  # torch.Size([1, 1, 9, 9])

            # state 정규화
            state = self.change_state_type(state)

            # q-value에 따라 action 선택
            with torch.no_grad():
                q_values = self.model(state).flatten().to("cpu")
                max_idx = torch.argmax(q_values)
                act = max_idx.item()

                self.q_values = q_values

        return act


    def train_model(self):
        '''
        replay memory에서 무작위로 배치를 추출해 1회 학습한다.
        '''
        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        # 추출한 샘플 tensor로 가져오기
        states = torch.tensor([sample[0] for sample in mini_batch], dtype=torch.float32).to(self.device).reshape(-1,1,self.nrow,self.ncol)
        actions = torch.tensor([sample[1] for sample in mini_batch], dtype=torch.long).to(self.device).reshape(-1,1)
        rewards = torch.tensor([sample[2] for sample in mini_batch], dtype=torch.float32).to(self.device).reshape(-1,1)
        next_states = torch.tensor([sample[3] for sample in mini_batch], dtype=torch.float32).to(self.device).reshape(-1,1,self.nrow,self.ncol)
        dones = torch.tensor([sample[4] for sample in mini_batch], dtype=torch.long).to(self.device).reshape(-1,1)
        clears = torch.tensor([sample[5] for sample in mini_batch], dtype=torch.bool).reshape(-1,1)

        # state 정규화
        states = self.change_state_type(states)
        next_states = self.change_state_type(next_states)

        # 현재 상태에 대한 모델의 큐함수
        predicts = self.model(states) # 현재 상태의 좌표를 준다
        one_hot_action = F.one_hot(actions, self.num_actions).to(self.device)
        predicts = torch.sum(one_hot_action*predicts, axis=1)

        # Q(s,a) 값을 예측값으로 사용 - (batch, num_actions)
        pred_q_values = self.model(states).gather(1, actions) # action idx의 데이터만 꺼냄

        # target 값 계산 : reward + gamma * Q(s',a')
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1).values.reshape(-1,1)
            target_q_values = rewards + (torch.ones(next_q_values.shape, device=self.device) - dones) * self.gamma * next_q_values

        # 오류 함수를 줄이는 방향으로 모델 업데이트
        loss = self.loss_func(pred_q_values, target_q_values)

        # 모델 학습
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

        # 엡실론 감소
        self.epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss

    
    def reset(self):
        '''
        에이전트를 완전히 초기화한다.
        '''
        self.q_values = torch.zeros(self.action_space.shape, dtype=torch.float32)

        self.epsilon = self.epsilon_init
        self.lr= self.lr_init

        self.memory = deque(maxlen = self.mem_size)

        if self.state_type == "one-hot":
            self.model = NetOneHot(self.state_size, self.num_actions, self.conv_units).to(self.device)
            self.target_model = NetOneHot(self.state_size, self.num_actions, self.conv_units).to(self.device)
        else:
            self.model = Net(self.state_size, self.num_actions, self.conv_units).to(self.device)
            self.target_model = Net(self.state_size, self.num_actions, self.conv_units).to(self.device)
        self.update_target_model()

        self.optimizer = self.optimizer_type(self.model.parameters(), lr=self.lr_init)


    def get_action_latest(self, state):
        '''
        현재 모델로 action을 선택한다. (No 엡실론 탐색)
        valid 시에 사용하는 함수이다.
        '''
        state = torch.tensor(state).to(self.device)
        state = state.unsqueeze(0).to(dtype = torch.float32)
        state = state.unsqueeze(0)  # torch.Size([1, 1, 9, 9])

        # 정규화
        state = self.change_state_type(state)

        with torch.no_grad():
            q_values = self.model(state).flatten().to("cpu")
            max_idx = torch.argmax(q_values)
            act = max_idx.item()

            self.q_values = q_values

        return act

    
    def get_action_best(self, state):
        '''
        이제껏 학습한 모델 중 best 모델로 action을 선택한다. (No 엡실론 탐색)
        valid 시에 사용하는 함수이다.
        '''
        state = torch.tensor(state).to(self.device)
        state = state.unsqueeze(0).to(dtype = torch.float32)
        state = state.unsqueeze(0)  # torch.Size([1, 1, 9, 9])

        # 정규화
        state = self.change_state_type(state)

        with torch.no_grad():
            q_values = self.best_model(state).flatten().to("cpu")
            max_idx = torch.argmax(q_values)
            act = max_idx.item()

            self.q_values = q_values

        return act