import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple

from environment import Environment
from dqn_agent import DQN_Agent

from visualization import *
#################################

class Log:
    def __init__(self, MODE, FOLDER_NAME:str, PATH:str):
        self.mode = MODE
        self.folder_name = FOLDER_NAME

        self.file_path = f"{PATH}/{self.mode}.pkl"
        self.path = Path(self.file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.reset()

    
    def reset(self):
        self.reward_list = []
        self.clear_list = []
        self.cnt_list = []
        self.rpc_list = []

        if self.mode == 'train':
            self.loss_list = []
            self.lr_list = []
        
        print(f"Reset {self.mode} log lists.")

    
    def update_logs(self, logs:list):
        if self.mode == 'train':
            total_reward, clear, cnt, loss, lr = logs
        else:
            total_reward, clear, cnt = logs

        self.reward_list.append(total_reward)
        self.clear_list.append(clear)
        self.cnt_list.append(cnt)
        self.rpc_list.append(total_reward / cnt)

        if self.mode == 'train':
            self.loss_list.append(loss)
            self.lr_list.append(lr)

    
    def save_logs(self):
        if self.mode == 'train':
            new_df = pd.DataFrame({'reward': self.reward_list,
                                    'clear': self.clear_list,
                                    'cnt': self.cnt_list,
                                    'rpc': self.rpc_list,
                                    'loss': self.loss_list,
                                    'lr': self.lr_list})
        
        else:
            new_df = pd.DataFrame({'reward': self.reward_list,
                                    'clear': self.clear_list,
                                    'cnt': self.cnt_list,
                                    'rpc': self.rpc_list})

            if self.path.exists():
                with open(self.path, 'rb') as f:
                    old_df = pickle.load(f)
                
                new_df = pd.concat([old_df, new_df], ignore_index=True)
        
        with open(self.path, 'wb') as f:
            pickle.dump(new_df, f)

    
    def load_logs(self):
        with open(self.path, 'rb') as f:
            df = pickle.load(f)

        return df