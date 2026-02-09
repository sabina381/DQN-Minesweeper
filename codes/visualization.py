import pickle
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from config import CONFIG

########################################
def calculate_lag_avg(data, lag):
    result = data.rolling(window = lag, min_periods = 1).mean()
    return result

def calculate_lag_mid(data, lag):
    result = data.rolling(window = lag, min_periods = 1).median()
    return result


def visualize_train_log(df, lag, save_path=None):

    fig, axs = plt.subplots(3, 3, figsize=(20, 15), squeeze=False)
    axs[0, 0].plot(calculate_lag_avg(df['reward'], lag), color = 'blue')
    axs[0, 0].plot(calculate_lag_mid(df['reward'], lag), color = 'skyblue')
    axs[0, 0].axhline(y=0, color='black', linewidth=1)
    axs[0, 0].set_title("Average / Median Reward")
    axs[0, 1].scatter(np.arange(len(df['reward'])), df['reward'], color = 'pink', alpha=0.7)
    axs[0, 1].axhline(y=0, color='black', linewidth=1)
    axs[0, 1].set_title("Reward")

    axs[1, 0].plot(calculate_lag_avg(df['cnt'], lag), color = 'blue')
    axs[1, 0].plot(calculate_lag_mid(df['cnt'], lag), color = 'skyblue')
    axs[1, 0].set_title("Average / Median Cnt")
    axs[1, 1].scatter(np.arange(len(df['cnt'])), df['cnt'], color = 'pink', alpha=0.7)
    axs[1, 1].set_title("Cnt")

    axs[2, 0].plot(calculate_lag_avg(df['rpc'], lag), color = 'blue')
    axs[2, 0].plot(calculate_lag_mid(df['rpc'], lag), color = 'skyblue')
    axs[2, 0].set_title("Average / Median Reward per Cnt")
    axs[2, 1].scatter(np.arange(len(df['rpc'])), df['rpc'], color = 'pink', alpha = 0.7)
    axs[2, 1].set_title('Reward per Cnt')

    axs[0, 2].plot(calculate_lag_avg(df['clear'], lag), color = 'blue')
    axs[0, 2].axhline(y=0.5, color='black', linewidth=1)
    axs[0, 2].set_title("Average Clear")

    axs[1, 2].plot(calculate_lag_avg(df['loss'], lag), color = 'black')
    axs[1, 2].plot(calculate_lag_mid(df['loss'], lag), color = 'grey')
    axs[1, 2].set_title("Average / Median Loss")

    axs[2, 2].plot(df['lr'], color = 'grey')
    axs[2, 2].set_title("Learning Rate")


    if save_path:
        plt.savefig(save_path)
        print(f"Save image at {save_path}")

    # plt.show()
    plt.close()
    
    # print("Complete printing image.")


def visualize_test_log(df, lag, save_path=None):

    fig, axs = plt.subplots(3, 2, figsize=(20, 10), squeeze=False)
    axs[0, 0].plot(calculate_lag_avg(df['reward'], lag), color = 'blue')
    axs[0, 0].plot(calculate_lag_mid(df['reward'], lag), color = 'pink')
    axs[0, 0].set_title("Average / Median Reward")
    axs[0, 1].scatter(np.arange(len(df['reward'])), df['reward'], color = 'pink')
    axs[0, 1].axhline(y=0, color='black', linewidth=1)
    axs[0, 1].set_title("Reward")

    axs[1, 0].plot(calculate_lag_avg(df['cnt'], lag), color = 'blue')
    axs[1, 0].plot(calculate_lag_mid(df['cnt'], lag), color = 'pink')
    axs[1, 0].set_title("Average / Median Cnt")
    axs[1, 1].scatter(np.arange(len(df['cnt'])), df['cnt'], color = 'pink', alpha=0.7)
    axs[1, 1].set_title("Cnt")

    axs[2, 0].plot(calculate_lag_avg(df['clear'], lag), color = 'blue')
    axs[2, 0].set_title("Average Clear")
    axs[2, 1].plot(calculate_lag_avg(df['rpc'], lag), color = 'blue')
    axs[2, 1].plot(calculate_lag_mid(df['rpc'], lag), color = 'skyblue')
    axs[2, 1].set_title("Average / Median Reward per Cnt")

    # plt.show()
    # print("Complete printing image.")

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Save image at {save_path}")


def visualize_state(state, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6)) # 크기 조절 가능
    nrow, ncol = CONFIG.GRIDWORLD_SIZE
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.set_aspect('equal')
    ax.axis('off') # 축 숨기기

    # 그리드 그리기
    for x in range(nrow):
        for y in range(ncol):
            # 1. 테두리 네모 그리기
            rect = patches.Rectangle((y, nrow - 1 - x), 1, 1, linewidth=1, edgecolor='gray', facecolor='black')
            ax.add_patch(rect)

            # 2. 값 가져오기
            val = state[x, y]
            if val == -1: 
                text_val = "."
            elif val == -2: 
                text_val = "M"
            else: 
                text_val = str(int(val))

            # 3. 텍스트 색상 결정
            t_color = CONFIG.COLOR_DICT.get(text_val, 'black')
            
            # 4. 텍스트 쓰기 (가운데 정렬)
            ax.text(y + 0.5, nrow - 1 - x + 0.5, text_val, 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    fontsize=12, 
                    color=t_color,
                    weight='bold')

    # 저장 로직
    if save_path:
        plt.savefig(save_path)
        print(f"Save image at {save_path}")

    # plt.show()
    plt.close()
    
    # print("Complete printing game image.")