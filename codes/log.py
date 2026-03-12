import pickle
from pathlib import Path
import pandas as pd
#################################

class Log:
    def __init__(self, MODE, FOLDER_NAME:str, PATH:str):
        self.mode = MODE    # train, valid, test
        self.folder_name = FOLDER_NAME

        self.file_path = f"{PATH}/{self.mode}.pkl"
        self.path = Path(self.file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.reset()

    
    def reset(self, new=True):
        '''
        mode에 따라 학습 지표 리스트 초기화
        '''
        # 공통 리스트
        self.reward_list = []
        self.clear_list = []
        self.cnt_list = []
        self.rpc_list = []

        # train log는 loss, lr 추가
        if self.mode == 'train':
            self.loss_list = []
            self.lr_list = []
        
        # valid 시 모델 평가 시 학습이 안된다고 판단하기 위한 지표 
        elif (self.mode == 'valid') and new:
            self.latest_update = 0
        
        print(f"Reset {self.mode} log lists.")

    
    def update_logs(self, logs:list):
        '''
        입력받은 logs에 따라 학습 지표 리스트 업데이트
        '''
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
        '''
        학습 지표 리스트를 하나의 df로 만들어 지정된 경로에 저장
        '''
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
        '''
        지정된 경로에서 학습지표 df 불러오기
        '''
        with open(self.path, 'rb') as f:
            df = pickle.load(f)

        return df

    
    def continue_logs(self):
        df = self.load_logs()

        # 공통 리스트
        self.reward_list = list(df['reward'])
        self.clear_list = list(df['clear'])
        self.cnt_list = list(df['cnt'])
        self.rpc_list = list(df['rpc'])

        # train log는 loss, lr 추가
        if self.mode == 'train':
            self.loss_list = list(df['loss'])
            self.lr_list = list(df['lr'])
        
        print(f"Load {self.mode} log lists from {self.file_path}")
