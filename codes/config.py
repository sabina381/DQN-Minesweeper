from easydict import EasyDict

############# Environment
gridworld_size = (9, 9)
num_mine = 9

reward_dict = {'mine':-1, 'empty':1, 'overlapped':-1, 'guess':0.3, 'clear':1}
done_dict = {'mine':True, 'empty':False, 'overlapped':False, 'guess':False, 'clear':True}

color_dict = {'0':'black', '1':"skyblue", '2':'lightgreen', '3':'red', '4':'violet', '5':'brown',
                '6':'turquoise', '7':'grey', '8':'black', 'M':'white', '.':'black'}

############# Hyper parameters
MEM_SIZE = 50000
MEM_SIZE_MIN = 1000

REWARD_MEM_SIZE = 20000
REWARD_MEM_SIZE_MIN = 500

BATCH_SIZE = 64
BATCH_RATE = 0

LEARN_MAX = 0.001 # 스케줄러 (learning rate 감소 모듈) 사용
LEARN_MIN = 0.0001
LEARN_DECAY = 0.5
LEARN_EPOCH = 50000

GAMMA = 0.1 #gamma

EPSILON = 0
EPSILON_DECAY = 0.99995
EPSILON_MIN = 0

############# DQN settings
CONV_UNITS = 64
UPDATE_TARGET_EVERY = 5

############# Train
PRINT_EVERY = 100
SAVE_EVERY = 100

EPISODES = 10000

#######################################
CONFIG = EasyDict({
    'GRIDWORLD_SIZE': gridworld_size,
    'NUM_MINE': num_mine,
    'REWARD_DICT': reward_dict,
    'DONE_DICT': done_dict,
    'COLOR_DICT': color_dict
})
