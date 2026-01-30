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
class DQN_Net(nn.Module):
    def __init__(self, state_size, action_size, conv_units):
        super().__init__()
        # 합성곱 층 정의
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=2)

        self.conv2 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_units)

        self.conv3 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_units)

        self.conv4 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)

        self.fc_size = conv_units * (state_size[-1]+2) * (state_size[-2]+2)
        self.fc = nn.Linear(self.fc_size, action_size)

    def forward(self, x):
        # 순전파
        x = F.relu(self.conv1(x))  # 첫 번째 합성곱층과 활성화 함수 적용 후 풀링
        x = F.relu(self.bn2(self.conv2(x)))  # 두 번째 합성곱층과 활성화 함수 적용
        x = F.relu(self.bn3(self.conv3(x)))  # 세 번째 합성곱층과 활성화 함수 적용
        x = F.relu(self.conv4(x))  # 네 번째 합성곱층과 활성화 함수 적용

        # flatten
        x = x.view(-1, self.fc_size)  # 배치 크기에 맞게 데이터를 평탄화
        # 완전 연결층
        x = self.fc(x)

        return x