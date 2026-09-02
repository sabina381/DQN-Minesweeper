from easydict import EasyDict

import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

############# Train
PRINT_EVERY = 100
SAVE_EVERY = 500
VALID_EVERY = 2500

LAG = 1000
MODEL_CRITERIA = 4

EPISODES = 400000
VALID_EPISODES = 1000

PATH = Path(__file__).resolve().parents[1]
FOLDER_NAME = "dqn_modelA_400k"

############# Environment
GRIDWORLD_SIZE = (9, 9)
NUM_MINE = 9
FIRST_MINE = True

REWARD_DICT = {'mine':-1, 'empty':1, 'overlapped':-1, 'guess':0.3, 'clear':1}
DONE_DICT = {'mine':True, 'empty':False, 'overlapped':False, 'guess':False, 'clear':True}

COLOR_DICT = {'0':'black', '1':"skyblue", '2':'lightgreen', '3':'red', '4':'violet', '5':'brown',
            '6':'turquoise', '7':'grey', '8':'black', 'M':'white', '.':'white'}

STATE_TYPE = "one-hot" # "original", "one-hot", "normalization"

############# Hyper parameters
MEM_SIZE = 50000
TRAIN_START = 1000

# REWARD_MEM_SIZE = 20000
# REWARD_MEM_SIZE_MIN = 500

BATCH_SIZE = 64
BATCH_RATE = 0

LEARN_MAX = 0.001 # 스케줄러 (learning rate 감소 모듈) 사용
LEARN_MIN = 0.0001
LEARN_DECAY = 0.5
LEARN_EPOCH = 50000

GAMMA = 0.1 #gamma

EPSILON = 0.99999
EPSILON_DECAY = 0.99997
EPSILON_MIN = 0.01

############# DQN settings
CONV_UNITS = 64
UPDATE_TARGET_EVERY = 10

LOSS_FUNC = F.mse_loss
OPTIMIZER = optim.Adam
# torch.optim.Adam(self.model.parameters(), lr=LEARN_MAX)

if torch.backends.mps.is_built():
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

elif torch.backends.cuda.is_built():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    DEVICE = torch.device("cpu")

#######################################
CONFIG = EasyDict({
    'FOLDER_NAME' : FOLDER_NAME,
    'PATH' : PATH,
    'DEVICE' : DEVICE,
    'STATE_TYPE' : STATE_TYPE,
    
    'GRIDWORLD_SIZE' : GRIDWORLD_SIZE,
    'NUM_MINE' : NUM_MINE,
    'FIRST_MINE' : FIRST_MINE,
    'REWARD_DICT' : REWARD_DICT,
    'DONE_DICT' : DONE_DICT,
    'COLOR_DICT' : COLOR_DICT,

    'MEM_SIZE' : MEM_SIZE,
    'TRAIN_START' : TRAIN_START,
    'BATCH_SIZE' : BATCH_SIZE,
    'CONV_UNITS' : CONV_UNITS,

    'LEARN_MAX' : LEARN_MAX,
    'LEARN_MIN' : LEARN_MIN,
    'LEARN_DECAY' : LEARN_DECAY,
    'LEARN_EPOCH' : LEARN_EPOCH,

    'GAMMA' : GAMMA,
    'EPSILON' : EPSILON,
    'EPSILON_DECAY' : EPSILON_DECAY,
    'EPSILON_MIN' : EPSILON_MIN,
    
    'UPDATE_TARGET_EVERY' : UPDATE_TARGET_EVERY,
    'LOSS_FUNC' : LOSS_FUNC,
    'OPTIMIZER' : OPTIMIZER,

    'PRINT_EVERY' : PRINT_EVERY,
    'SAVE_EVERY' : SAVE_EVERY,
    'VALID_EVERY' : VALID_EVERY,
    'EPISODES' : EPISODES,
    'VALID_EPISODES': VALID_EPISODES,
    'LAG' : LAG,
    'MODEL_CRITERIA': MODEL_CRITERIA
})
