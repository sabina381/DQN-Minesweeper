import torch
import torch.nn.functional as F

###########################################
def mine_normalize(state):
    norm_state = state.clone().float()  # torch.Size([batch_size, 1, 9, 9])
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
    raw_state = state.clone()   # torch.Size([batch_size, 1, 9, 9])
    batch_size, _, nrow, ncol = raw_state.size()

    shifted_state = raw_state + 2
    one_hot = F.one_hot(shifted_state.squeeze(1).long(), num_classes=11) # torch.Size([batch_size, 9, 9, 11])
    scaled_state = one_hot.permute(0, 3, 1, 2).float().contiguous() # torch.Size([batch_size, 11, 9, 9])

    assert scaled_state.dim() == 4, f"The tensor must be 4-dimensional. 현재 차원: {scaled_state.dim()}"
    assert scaled_state.size() == torch.Size([batch_size, 11, nrow, ncol]), f"The tensor size must be [{batch_size}, 11, {nrow}, {ncol}]. 현재 차원: {scaled_state.size()}"

    return scaled_state

