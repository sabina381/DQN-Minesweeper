import torch
import torch.nn as nn
import torch.nn.functional as F

from scaling import *

########################################
class Net(nn.Module):
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

class NetOneHot(nn.Module):
    def __init__(self, state_size, action_size, conv_units):
        super().__init__()
        # 합성곱 층 정의
        self.conv1 = nn.Conv2d(in_channels=11, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)

        self.conv2 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_units)

        self.conv3 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_units)

        self.conv4 = nn.Conv2d(in_channels=conv_units, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        
        self.fc_size = conv_units * (state_size[-1]) * (state_size[-2])
        self.fc = nn.Linear(self.fc_size, action_size)

    def forward(self, x):
        # 순전파
        x = F.relu(self.conv1(x))  # 첫 번째 합성곱층과 활성화 함수 적용 후 풀링
        x = F.relu(self.bn2(self.conv2(x)))  # 두 번째 합성곱층과 활성화 함수 적용
        x = F.relu(self.bn3(self.conv3(x)))  # 세 번째 합성곱층과 활성화 함수 적용
        x = F.relu(self.conv4(x))  # 네 번째 합성곱층과 활성화 함수 적용

        # flatten
        x = x.reshape(-1, self.fc_size)  # 배치 크기에 맞게 데이터를 평탄화
        # 완전 연결층
        x = self.fc(x)

        return x


def main():
    t = torch.randint(-2, 9, (3, 1, 9, 9)).long()
    scaled_t = one_hot_scaling(t)
    print(scaled_t.size())
    net = NetOneHot((9, 9), 9*9, 64)
    q = net.forward(scaled_t)
    print(q.size())



if __name__ == "__main__":
    main()