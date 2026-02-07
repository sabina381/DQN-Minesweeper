import pickle
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

########################################
def visualize_train_log(file):
    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'rb') as f:
        df = pickle.load(f)

    fig, axs = plt.subplots(4, 2, figsize=(20, 15), squeeze=False)
    axs[0, 0].plot(df['avg_rewards'], color = 'blue')
    axs[0, 0].plot(df['mid_rewards'], color = 'pink')
    axs[0, 0].axhline(y=0, color='black', linewidth=1)
    axs[0, 0].set_title("Average / Median Reward")
    axs[0, 1].scatter(np.arange(len(df['rewards'])), df['rewards'], color = 'pink', alpha=0.7)
    axs[0, 1].axhline(y=0, color='black', linewidth=1)
    axs[0, 1].set_title("Reward")

    axs[1, 0].plot(df['avg_cnt'], color = 'blue')
    axs[1, 0].plot(df['mid_cnt'], color = 'pink')
    axs[1, 0].set_title("Average / Median Cnt")
    axs[1, 1].scatter(np.arange(len(df['cnt'])), df['cnt'], color = 'pink', alpha=0.7)
    axs[1, 1].set_title("Cnt")

    axs[2, 0].plot(df['avg_loss'], color = 'blue')
    axs[2, 0].plot(df['mid_loss'], color = 'pink')
    axs[2, 0].set_title("Average / Median Loss")
    axs[2, 1].plot(df['loss'], color = 'grey')
    axs[2, 1].set_title('Loss')

    axs[3, 0].plot(df['avg_clear'], color = 'blue')
    axs[3, 0].axhline(y=0.5, color='black', linewidth=1)
    axs[3, 0].set_title("Average Clear")

    axs[3, 1].plot(df['lr'], color = 'grey')
    axs[3, 1].set_title("Learning Rate")

    plt.show()


def visualize_test_log(file):
    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'rb') as f:
        df = pickle.load(f)

    fig, axs = plt.subplots(3, 2, figsize=(20, 10), squeeze=False)
    axs[0, 0].plot(df['avg_rewards'], color = 'blue')
    axs[0, 0].plot(df['mid_rewards'], color = 'pink')
    axs[0, 0].set_title("Average / Median Reward")
    axs[0, 1].scatter(np.arange(len(df['rewards'])), df['rewards'], color = 'pink')
    axs[0, 1].axhline(y=0, color='black', linewidth=1)
    axs[0, 1].set_title("Reward")

    axs[1, 0].plot(df['avg_cnt'], color = 'blue')
    axs[1, 0].plot(df['mid_cnt'], color = 'pink')
    axs[1, 0].set_title("Average / Median Cnt")
    axs[1, 1].scatter(np.arange(len(df['cnt'])), df['cnt'], color = 'pink', alpha=0.7)
    axs[1, 1].set_title("Cnt")

    axs[2, 0].plot(df['avg_clear'], color = 'blue')
    axs[2, 0].set_title("Average Clear")
    axs[2, 1].scatter(df['clear'], color = 'pink', alpha=0.7)
    axs[2, 1].set_title("Clear")

    plt.show()