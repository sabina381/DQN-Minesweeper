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
    for i, (state, reward, action, clear, _) in enumerate(episode_data):
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


def visualize_state_and_q(episode_data, save_path=None):
    '''
    episode_data: [(state, action, reward, q_values), ...] 형태의 리스트
    '''
    num_steps = len(episode_data)
    
    # 1. 그리드 레이아웃 설정: 1스텝당 1줄 (열 2개: State, Q-value)
    # 이미지 높이는 스텝 수에 비례하여 길어집니다.
    fig, axes = plt.subplots(num_steps, 2, figsize=(10, num_steps * 4.5))

    # 스텝이 1개일 경우 axes가 1D 배열이 되므로, 2D 배열처럼 다루기 위해 차원 확장
    if num_steps == 1:
        axes = np.array([axes])

    nrow, ncol = CONFIG.GRIDWORLD_SIZE
    cmap = plt.cm.coolwarm

    # 2. 각 스텝(row)마다 그리기
    for i, (state, reward, action, clear, q_values) in enumerate(episode_data):
        ax_state = axes[i, 0] # 왼쪽 축 (State)
        ax_q = axes[i, 1]     # 오른쪽 축 (Q-value)

        # 기본 축 설정 (축 숨기기, 비율 맞추기)
        for ax in (ax_state, ax_q):
            ax.set_xlim(0, ncol)
            ax.set_ylim(0, nrow)
            ax.set_aspect('equal')
            ax.axis('off')

        # 상단 제목 표시
        is_clear = "clear" if clear else "."
        ax_state.set_title(f"Step {i+1} State | Reward: {reward:.2f} | {is_clear}", fontsize=12, fontweight='bold')
        ax_q.set_title(f"Step {i+1} Q-Values", fontsize=12, fontweight='bold')

        # 1D action -> 2D 좌표
        if action is not None:
            action_x, action_y = divmod(action, ncol)
        else:
            action_x, action_y = -1, -1

        # Q-values를 2D 배열로 변환
        if torch.is_tensor(q_values):
            q_values = q_values.detach().cpu().numpy()
        q_2d = q_values.reshape((nrow, ncol))
        
        # 색상 매핑을 위한 Q-value 최소/최대값 정규화
        vmin, vmax = np.min(q_2d), np.max(q_2d)
        if vmin == vmax: 
            vmin, vmax = vmin - 1e-5, vmax + 1e-5

        # 3. 그리드 그리기 (State와 Q-value를 동시에)
        for x in range(nrow):
            for y in range(ncol):
                
                is_action = (x == action_x and y == action_y)
                edge_color = 'red' if is_action else 'gray'
                line_width = 3 if is_action else 1
                z_order = 2 if is_action else 1 

                # ==========================================
                # [왼쪽] State 캔버스 그리기
                # ==========================================
                val = state[x, y]
                bg_color_state = 'dimgray' if val == -1 else 'black'

                rect_state = patches.Rectangle((y, nrow - 1 - x), 1, 1, 
                                         linewidth=line_width, edgecolor=edge_color, 
                                         facecolor=bg_color_state, zorder=z_order)
                ax_state.add_patch(rect_state)

                if val == -1: text_val_state = "" # 안 열린 곳은 빈칸으로 깔끔하게
                elif val == -2: text_val_state = "M"
                else: text_val_state = str(int(val))

                t_color_state = CONFIG.COLOR_DICT.get(text_val_state, 'black')
                
                ax_state.text(y + 0.5, nrow - 1 - x + 0.5, text_val_state, 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12, color=t_color_state, weight='bold', zorder=3)

                # ==========================================
                # [오른쪽] Q-value 캔버스 그리기
                # ==========================================
                q_val = q_2d[x, y]
                norm_q = (q_val - vmin) / (vmax - vmin)
                bg_color_q = cmap(norm_q) 

                rect_q = patches.Rectangle((y, nrow - 1 - x), 1, 1, 
                                         linewidth=line_width, edgecolor=edge_color, 
                                         facecolor=bg_color_q, zorder=z_order)
                ax_q.add_patch(rect_q)

                text_val_q = f"{q_val:.2f}"
                
                ax_q.text(y + 0.5, nrow - 1 - x + 0.5, text_val_q, 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=9, color='black', weight='bold', zorder=3)

    # 4. 마무리
    # bbox_inches='tight'를 주면 바깥 여백이 잘려서 이미지가 콤팩트해집니다.
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Save episode summary image at {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_state_with_q(episode_data, save_path=None):
    '''
    episode_data: [(state, reward, action, clear, q_values), ...] 형태의 리스트
    '''
    num_steps = len(episode_data)
    
    # 1. 그리드 레이아웃 설정 (한 줄에 5개씩)
    cols = 5
    rows = (num_steps + cols - 1) // cols
    
    # 캔버스 크기 동적 할당
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0))

    axes = axes.flatten()

    nrow, ncol = CONFIG.GRIDWORLD_SIZE
    cmap = plt.cm.coolwarm

    # 2. 각 스텝별로 그리기
    for i, (state, reward, action, clear, q_values) in enumerate(episode_data):
        ax = axes[i]
        
        ax.set_xlim(0, ncol)
        ax.set_ylim(0, nrow)
        ax.set_aspect('equal')
        ax.axis('off')

        # 상단 제목 (Clear 여부 간략히 표시)
        is_clear = "Clear!" if clear else ""
        ax.set_title(f"Step {i+1} | R: {reward:.2f} | {is_clear}", fontsize=11, fontweight='bold')

        # 1D action -> 2D 좌표
        if action is not None:
            action_x, action_y = divmod(action, ncol)
        else:
            action_x, action_y = -1, -1

        # Q-values를 2D 배열로 변환
        if torch.is_tensor(q_values):
            q_values = q_values.detach().cpu().numpy()
        q_2d = q_values.reshape((nrow, ncol))
        
        # [핵심] 색상 매핑을 위한 최소/최대값 정규화
        vmin, vmax = np.min(q_2d), np.max(q_2d)
        if vmin == vmax: 
            vmin, vmax = vmin - 1e-5, vmax + 1e-5

        # 3. 그리드 칸 그리기
        for x in range(nrow):
            for y in range(ncol):
                
                is_action = (x == action_x and y == action_y)
                edge_color = 'red' if is_action else 'gray'
                line_width = 3 if is_action else 1
                z_order = 2 if is_action else 1 

                val = state[x, y]
                q_val = q_2d[x, y]
                
                # ==================================================
                # [상태 분류] 안 열린 곳(히트맵) vs 열린 곳(기존 State)
                # ==================================================
                if val == -1: 
                    # 1) 안 열린 칸: Q-value 히트맵과 숫자 표시
                    norm_q = (q_val - vmin) / (vmax - vmin)
                    bg_color = cmap(norm_q)
                    text_val = f"{q_val:.2f}"
                    t_color = 'black'
                    f_size = 8  # 소수점이 길어서 폰트 작게
                    
                else: 
                    # 2) 열린 칸 (숫자 또는 지뢰): 검은 배경에 컬러 텍스트
                    bg_color = 'black'
                    if val == -2: 
                        text_val = "M"
                    elif val == 0:
                        text_val = "" # 0은 빈칸으로 두는 것이 깔끔합니다. (원하시면 str(int(val))로 변경 가능)
                    else: 
                        text_val = str(int(val))
                        
                    t_color = CONFIG.COLOR_DICT.get(text_val, 'white')
                    f_size = 12 # 숫자는 잘 보이게 크게

                # 사각형 배경색 칠하기
                rect = patches.Rectangle((y, nrow - 1 - x), 1, 1, 
                                         linewidth=line_width, edgecolor=edge_color, 
                                         facecolor=bg_color, zorder=z_order)
                ax.add_patch(rect)

                # 텍스트 올리기
                ax.text(y + 0.5, nrow - 1 - x + 0.5, text_val, 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=f_size, color=t_color, weight='bold', zorder=3)

    # 4. 남는 빈칸(subplot) 숨기기
    for j in range(num_steps, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Save episode summary image at {save_path}")
    else:
        plt.show()

    plt.close()