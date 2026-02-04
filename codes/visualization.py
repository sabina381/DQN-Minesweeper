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
def visualize_train_log(file):
    with open(file, 'rb') as f:
        df = pickle.load(f)

    fig, axs = plt.subplots(4, 2, figsize=(20, 15), squeeze=False)
    axs[0, 0].plot(df['avg_rewards'], color = 'blue')
    axs[0, 0].plot(df['mid_rewards'], color = 'red')
    axs[0, 0].set_title("Average / Median Reward")
    axs[0, 1].bar(df['rewards'], color = 'black')
    axs[0, 1].axhline(y=0, color='black', linewidth=1)
    axs[0, 1].set_title("Reward")

    axs[1, 0].plot(df['avg_count'], color = 'blue')
    axs[1, 0].plot(df['mid_count'], color = 'red')
    axs[1, 0].set_title("Average / Median Cnt")
    axs[1, 1].plot(df['cnt'], color = 'black')
    axs[1, 1].set_title("Cnt")

    axs[2, 0].plot(df['avg_loss'], color = 'blue')
    axs[2, 0].plot(df['mid_loss'], color = 'red')
    axs[2, 0].set_title("Average / Median Loss")
    axs[2, 1].plot(df['loss'], color = 'black')
    axs[2, 1].set_title('Loss')

    axs[3, 0].plot(df['avg_clear'], color = 'blue')
    axs[3, 0].plot(df['mid_clear'], color = 'red')
    axs[3, 0].set_title("Average / Median Clear")

    axs[3, 1].plot(df['lr'], color = 'grey')
    axs[1, 3].set_title("Learning Rate")

    plt.show()


def visualize_test_log(file):
    with open(file, 'rb') as f:
        df = pickle.load(f)

    fig, axs = plt.subplots(3, 2, figsize=(20, 10), squeeze=False)
    axs[0, 0].plot(df['avg_rewards'], color = 'blue')
    axs[0, 0].plot(df['mid_rewards'], color = 'red')
    axs[0, 0].set_title("Average / Median Reward")
    axs[0, 1].bar(df['rewards'], color = 'black')
    axs[0, 1].axhline(y=0, color='black', linewidth=1)
    axs[0, 1].set_title("Reward")

    axs[1, 0].plot(df['avg_count'], color = 'blue')
    axs[1, 0].plot(df['mid_count'], color = 'red')
    axs[1, 0].set_title("Average / Median Cnt")
    axs[1, 1].plot(df['cnt'], color = 'black')
    axs[1, 1].set_title("Cnt")

    axs[3, 0].plot(df['avg_clear'], color = 'blue')
    axs[3, 0].plot(df['mid_clear'], color = 'red')
    axs[3, 0].set_title("Average / Median Clear")
    axs[3, 1].bar(df['clear'], color = 'black')
    axs[3, 1].set_title("Clear")

    plt.show()