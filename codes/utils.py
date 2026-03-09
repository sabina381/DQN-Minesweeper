import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from config import CONFIG

############ Scaling ############
def one_channel_scaling(state):
    '''
    state의 차원을 그대로 유지하면서 정규화한다.
    output dim = [batch_size, 1, nrow, ncol]
    지뢰(-2): -1, 안열린 칸(-1): -0.5, 0~8: min-max normalizing
    '''
    norm_state = state.clone().float()  # torch.Size([batch_size, 1, nrow, ncol])
    batch_size, _, nrow, ncol = norm_state.size()

    # 열린 칸 정규화 처리
    mask_number = (state >= 0)
    norm_state[mask_number] = state[mask_number] / 8.0

    # 닫혀있는 칸은 -0.5로 고정
    mask_unopened = (state == -1)
    norm_state[mask_unopened] = -0.5

    # 지뢰 칸은 -1.0로 고정
    mask_mine = (state == -2)
    norm_state[mask_mine] = -1.0

    assert norm_state.dim() == 4, f"The tensor must be 4-dimensional. 현재 차원: {norm_state.dim()}"
    assert norm_state.size() == torch.Size([batch_size, 1, nrow, ncol]), f"The tensor size must be [{batch_size}, 1, {nrow}, {ncol}]. 현재 차원: {norm_state.size()}"

    return norm_state


def one_hot_scaling(state):
    '''
    state를 one-hot encoding 방식으로 변경한다.
    output dim = [batch_size, 11, nrow, ncol]
    채널 순서: -2(지뢰) ~ 8
    '''
    raw_state = state.clone()   # torch.Size([batch_size, 1, nrow, ncol])
    batch_size, _, nrow, ncol = raw_state.size()

    shifted_state = raw_state + 2
    one_hot = F.one_hot(shifted_state.squeeze(1).long(), num_classes=11) # torch.Size([batch_size, nrow, ncol, 11])
    scaled_state = one_hot.permute(0, 3, 1, 2).float().contiguous() # torch.Size([batch_size, 11, nrow, ncol])

    assert scaled_state.dim() == 4, f"The tensor must be 4-dimensional. 현재 차원: {scaled_state.dim()}"
    assert scaled_state.size() == torch.Size([batch_size, 11, nrow, ncol]), f"The tensor size must be [{batch_size}, 11, {nrow}, {ncol}]. 현재 차원: {scaled_state.size()}"

    return scaled_state


############ Visualization ############
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


def visualize_episodes(episode_data, save_path=None):
    num_steps = len(episode_data)
    
    # 1. 그리드 레이아웃 설정 (한 줄에 5개씩 출력)
    cols = 5 
    rows = (num_steps + cols - 1) // cols
    
    # 데이터 길이에 맞춰 전체 Figure 크기 동적 할당
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))

    if num_steps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    nrow, ncol = CONFIG.GRIDWORLD_SIZE

    # 2. 각 step별로 서브플롯(ax)에 그리기
    for i, (state, reward, action, clear) in enumerate(episode_data):
        ax = axes[i]
        ax.set_xlim(0, ncol)
        ax.set_ylim(0, nrow)
        ax.set_aspect('equal')
        ax.axis('off') # 축 숨기기

        # 상단에 Step과 Reward 표시
        is_clear = "clear" if clear else "."
        ax.set_title(f"Step {i+1} | Reward: {reward:.2f} | {is_clear}", fontsize=11, fontweight='bold')

        if action is not None:
            action_x, action_y = divmod(action, ncol)  # action_x = action // ncol, action_y = action % ncol
        else:
            action_x, action_y = -1, -1  # action이 없는 경우 (예외 처리)

        # 그리드 그리기
        for x in range(nrow):
            for y in range(ncol):
                val = state[x, y]
                bg_color = 'dimgray' if val == -1 else 'black'

                is_action = (x == action_x and y == action_y)

                edge_color = 'red' if is_action else 'gray'
                line_width = 3 if is_action else 1
                z_order = 2 if is_action else 1 # 빨간 테두리가 다른 칸에 가려지지 않게 맨 위로 올림

                # 테두리 네모 그리기
                rect = patches.Rectangle((y, nrow - 1 - x), 1, 1, 
                                         linewidth=line_width, 
                                         edgecolor=edge_color, 
                                         facecolor=bg_color,
                                         zorder=z_order) # zorder 적용
                ax.add_patch(rect)

                if val == -1: 
                    text_val = " "
                elif val == -2: 
                    text_val = "M"
                else: 
                    text_val = str(int(val))

                # 텍스트 색상 결정
                t_color = CONFIG.COLOR_DICT.get(text_val, 'black')
                
                # 텍스트 쓰기
                ax.text(y + 0.5, nrow - 1 - x + 0.5, text_val, 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontsize=10, 
                        color=t_color,
                        weight='bold',
                        zorder=3) # 글자도 맨 위로 올림

    # 3. 데이터가 그려지지 않은 남는 빈칸(subplot) 숨기기
    for j in range(num_steps, len(axes)):
        axes[j].axis('off')

    # 서브플롯 간 간격 자동 조절
    plt.tight_layout()

    # 4. 저장 로직
    if save_path:
        plt.savefig(save_path)
        print(f"Save episode image at {save_path}")
        
    else:
        plt.show()

    plt.close()