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
def visualizing(file):
    with open(file, 'rb') as f:
        df = pickle.load(f)

    fig, axs = plt.subplots(3, 4, figsize=(20, 15), squeeze=False)
    axs[0, 0].plot(df['ave_rewards'], color = 'blue')
    axs[0, 0].set_title("Avg Reward")
    axs[0, 1].plot(df['mid_rewards'], color = 'blue')
    axs[0, 1].set_title("Median Reward")
    axs[0, 2].plot(df['rewards'], color = 'blue')
    axs[0, 2].set_title("Reward")

    axs[1, 0].plot(df['ave_count'], color = 'skyblue')
    axs[1, 0].set_title("Avg Cnt")
    axs[1, 1].plot(df['mid_count'], color = 'skyblue')
    axs[1, 1].set_title("Median Cnt")
    axs[1, 2].plot(df['cnt'], color = 'skyblue')
    axs[1, 2].set_title("Cnt")

    axs[2, 0].plot(df['ave_loss'], color = 'green')
    axs[2, 0].set_title("Avg Loss")
    axs[2, 1].plot(df['mid_loss'], color = 'green')
    axs[2, 1].set_title("Median Loss")
    axs[2, 2].plot(df['loss'], color = 'green')
    axs[2, 2].set_title('Loss')

    axs[0, 3].plot(df['ave_clear'], color = 'red')
    axs[0, 3].set_title("Avg Clear")
    axs[1, 3].plot(df['lr'], color = 'grey')
    axs[1, 3].set_title("Learning Rate")

    plt.show()


def test_visualizing(file):
    with open(file, 'rb') as f:
        df = pickle.load(f)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10), squeeze=False)
    axs[0, 0].plot(df['avg_rewards'], color = 'blue')
    axs[0, 0].set_title("Avg Reward")
    axs[0, 1].plot(df['mid_rewards'], color = 'blue')
    axs[0, 1].set_title("Median Reward")
    axs[0, 2].plot(df['rewards'], color = 'blue')
    axs[0, 2].set_title("Reward")

    axs[1, 0].plot(df['avg_count'], color = 'skyblue')
    axs[1, 0].set_title("Avg Cnt")
    axs[1, 1].plot(df['mid_count'], color = 'skyblue')
    axs[1, 1].set_title("Median Cnt")
    axs[1, 2].plot(df['cnt'], color = 'skyblue')
    axs[1, 2].set_title("Cnt")

    axs[0, 3].plot(df['avg_clear'], color = 'red')
    axs[0, 3].set_title("Avg Clear")

    plt.show()