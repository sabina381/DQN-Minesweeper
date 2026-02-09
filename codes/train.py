import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple

from environment import Environment
from dqn_agent import DQN_Agent

from visualization import *
from log import Log
########################################
class Trainer:
    def __init__(self, 
                # Trainer
                FOLDER_NAME:str, PATH:str, EPISODES:int, VALID_EPISODES:int, LAG:int, MODEL_CRITERIA:int,
                UPDATE_TARGET_EVERY:int, PRINT_EVERY:int, SAVE_EVERY:int, VALID_EVERY:int,
                TRAIN_START:int, LEARN_MIN:float, LEARN_DECAY:float, LEARN_EPOCH:float,
                # Environment
                GRIDWORLD_SIZE:Tuple, NUM_MINE:int, REWARD_DICT:dict, DONE_DICT:dict, COLOR_DICT:dict, FIRST_MINE:bool,
                # DQN_Agent
                STATE_TYPE:str, EPSILON:float, EPSILON_DECAY:float, EPSILON_MIN:float, BATCH_SIZE:int, 
                GAMMA:float, MEM_SIZE:int, LEARN_MAX:float, CONV_UNITS:int, DEVICE, LOSS_FUNC, OPTIMIZER
                ):

        self.env = Environment(GRIDWORLD_SIZE= GRIDWORLD_SIZE, 
                            NUM_MINE= NUM_MINE, 
                            REWARD_DICT= REWARD_DICT, 
                            DONE_DICT= DONE_DICT, 
                            COLOR_DICT= COLOR_DICT,
                            FIRST_MINE= FIRST_MINE)

        self.agent = DQN_Agent(STATE_SIZE= GRIDWORLD_SIZE, 
                            STATE_TYPE= STATE_TYPE, 
                            NUM_MINE= NUM_MINE, 
                            EPSILON= EPSILON, 
                            EPSILON_DECAY= EPSILON_DECAY, 
                            EPSILON_MIN= EPSILON_MIN, 
                            BATCH_SIZE= BATCH_SIZE, 
                            GAMMA= GAMMA, 
                            MEM_SIZE= MEM_SIZE, 
                            LEARN_MAX= LEARN_MAX, 
                            CONV_UNITS= CONV_UNITS, 
                            DEVICE= DEVICE, 
                            LOSS_FUNC= LOSS_FUNC, 
                            OPTIMIZER= OPTIMIZER)

        self.device = DEVICE

        self.cnt_limit = self.env.nrow * self.env.ncol - self.env.num_mine
        self.train_start = TRAIN_START

        self.lr_min = LEARN_MIN
        self.lr_decay = LEARN_DECAY
        self.lr_epoch = LEARN_EPOCH

        self.folder_name = FOLDER_NAME
        self.path = f"{PATH}/{FOLDER_NAME}"
        self.create_path()

        self.episodes = EPISODES
        self.valid_total_episodes = VALID_EPISODES
        self.test_total_episodes = 0

        self.update_target_every = UPDATE_TARGET_EVERY
        self.print_every = PRINT_EVERY
        self.save_every = SAVE_EVERY
        self.valid_every = VALID_EVERY
        self.lag = LAG
        self.model_criteria = MODEL_CRITERIA
        self.mode = "train"
        
        self.train_log = Log(MODE= 'train', FOLDER_NAME= self.folder_name, PATH=self.path_dict['logs'])
        self.valid_log = Log(MODE= 'valid', FOLDER_NAME= self.folder_name, PATH=self.path_dict['logs'])
        self.test_log = Log(MODE= 'test', FOLDER_NAME= self.folder_name, PATH=self.path_dict['logs'])

        self.log_dict = dict({'train': self.train_log, 
                            'valid': self.valid_log, 
                            'test': self.test_log})

        self.cur_epi_dict = dict({'train': 0, 'valid': 0, 'test': 0})

    def create_path(self):
        self.path_dict = ({'model': Path(f"{self.path}/models"),
                        'log_message': Path(f"{self.path}/{self.folder_name}_training_log.txt"),
                        'logs': Path(f"{self.path}/logs"),
                        'graph': Path(f"{self.path}/graphs"),
                        'game_imgs': Path(f"{self.path}/game_imgs"),
                        'memory': Path(f"{self.path}/memory.pkl")})

        self.path_dict['log_message'].parent.mkdir(parents=True, exist_ok=True)
        self.path_dict['memory'].parent.mkdir(parents=True, exist_ok=True)
        self.path_dict['model'].mkdir(parents=True, exist_ok=True)
        self.path_dict['logs'].mkdir(parents=True, exist_ok=True)
        self.path_dict['graph'].mkdir(parents=True, exist_ok=True)
        self.path_dict['game_imgs'].mkdir(parents=True, exist_ok=True)

        if not Path(self.path_dict['log_message']).exists():
            with open(self.path_dict['log_message'], "w") as f:
                f.write(f"Training Log for {self.folder_name}\n")
                f.write("="*50 + "\n")
            
        log_str = "< path dict >\n"

        for item in self.path_dict.items():
            log_str += str(item) + "\n"

        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)


    def load_model(self, model_name):
        file_path = f"{self.path_dict['model']}/{model_name}.pkl"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_str = ""

        if not path.exists():
            if model_name == 'best':
                with open(path, 'wb') as f:
                    pickle.dump(self.agent.model.state_dict(), f)
                    log_str += "\nCreate best model"
                    
            else:
                log_str += "\nCreate new model"
                return

        with open(path, 'rb') as f:
            model_param = pickle.load(f)

        if model_name == 'best':
            self.agent.best_model.load_state_dict(model_param)
            self.agent.best_model.to(self.device)

        else:
            self.agent.model.load_state_dict(model_param)
            self.agent.target_model.load_state_dict(model_param)

            self.agent.model.to(self.device)
            self.agent.target_model.to(self.device)
            
        log_str += f"\nModel loaded from \'{file_path}\'"
        print(log_str)
        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)


    def save_model(self, model_name):
        self.agent.model.to("cpu")

        file_path = f"{self.path_dict['model']}/{model_name}.pkl"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(path, 'wb') as f:
            pickle.dump(self.agent.model.state_dict(), f)
        
        log_str = f"\nModel saved to \'{file_path}\'"
        # print(log_str)
        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)

    
    def change_model(self):
        file_path = f"{self.path_dict['model']}/best.pkl"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'rb') as f:
            model_param = pickle.load(f)
        
        self.agent.best_model.load_state_dict(model_param)
        self.agent.best_model.to(self.device)

        self.agent.model.load_state_dict(model_param)
        self.agent.model.to(self.device)

        self.agent.target_model.load_state_dict(model_param)
        self.agent.target_model.to(self.device)

        log_str = "Agent model changed to best model.\n"
        # print(log_str) 
        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)

    
    def _save_memory(self):
        with open(self.path_dict['memory'], 'wb') as f:
            pickle.dump(self.agent.memory, f)


    def reset(self):
        self.env.reset()
        self.agent.reset()
        self.load_model('latest')

        for key in self.cur_epi_dict.keys():
            self.cur_epi_dict[key] = 0
            self.log_dict[key].reset()


    def _game_reset(self):
        self.env.reset()

        state = self.env.present_state.copy()
        done = False
        clear = False
        total_reward = 0
        cnt = 0
        loss = 0

        self.agent.action_space = self.env.points.copy()

        return state, done, clear, total_reward, cnt, loss

    
    def _check_cnt_limit(self, cnt:int):
        done = True if cnt > self.cnt_limit else False
        return done


    def _print_log(self):
        key = self.mode

        log_str = "\n"

        if key == 'valid':
            log_str += f"\t==valid[{self.cur_epi_dict[key]}/{self.valid_total_episodes}]" + " "
        else: 
            log_str += f"[{self.cur_epi_dict[key]}/{self.episodes}]" + " "

        log_str += f"- Avg Clear: {round(np.mean(self.log_dict[key].clear_list[-self.print_every:]), 3)}" + " "
        log_str += f"- Cnt: {round(np.mean(self.log_dict[key].cnt_list[-self.print_every:]), 3)}/{round(np.median(self.log_dict[key].cnt_list[-self.print_every:]), 3)}" + " "
        log_str += f"- Reward: {round(np.mean(self.log_dict[key].reward_list[-self.print_every:]), 3)}/{round(np.median(self.log_dict[key].reward_list[-self.print_every:]), 3)}" + " "
        log_str += f"- RPC: {round(np.mean(self.log_dict[key].rpc_list[-self.print_every:]), 3)}/{round(np.median(self.log_dict[key].rpc_list[-self.print_every:]), 3)}" + " "

        if key == 'train':
            log_str += f"- Loss: {round(np.mean(self.log_dict[key].loss_list[-self.print_every:]), 3)}/{round(np.median(self.log_dict[key].loss_list[-self.print_every:]), 3)} - epsilon: {round(self.agent.epsilon, 5)} - lr: {round(self.agent.lr, 5)}"
        
        # print(log_str)
        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)
        

    def visualize_log(self):
        key = self.mode
        df = self.log_dict[key].load_logs()
        cur_epi = self.cur_epi_dict[key]
        path = f"{self.path_dict['graph']}/{key}_{cur_epi}.png"

        if key == 'train':
            visualize_train_log(df = df, lag = self.lag, save_path = path)

        else:
            visualize_test_log(df = df, lag = self.lag, save_path = path)

        log_str = "Visualizing log complete.\n"
        # print(log_str)

        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)
        


    def train(self):
        self.reset()
        print("Train start")

        for episode in range(self.episodes):
            self.mode = 'train'
            self.agent.model.to(self.device)
            self.cur_epi_dict['train'] += 1
            # reset
            state, done, clear, total_reward, cnt, loss = self._game_reset()

            # 게임 종료까지 반복
            while not done:
                cnt+=1
                state = self.env.present_state.copy()
                action = self.agent.get_action(state)
                next_state, reward, done, clear = self.env.step(action)
                total_reward += reward

                # count 제한 : 전체 칸 수 - 지뢰 개수
                if not done:
                    done = self._check_cnt_limit(cnt)

                # replay memory에 샘플 저장
                self.agent.append_sample(state, action, reward, next_state, done, clear)

                # 학습
                if len(self.agent.memory) > self.train_start:
                    loss = self.agent.train_model()
                    loss = loss.item()

                if done or clear:
                    break

            # 평가지표
            self.log_dict['train'].update_logs([total_reward, clear, cnt, loss, self.agent.lr])

            # 타깃 모델 업데이트
            if episode % self.update_target_every == 0:
                self.agent.update_target_model()

            # lr 조절
            if self.lr_epoch > 0:
                if (episode+1) % self.lr_epoch == 0:
                    lr = self.agent.optimizer.param_groups[0]['lr'] * self.lr_decay
                    self.agent.optimizer.param_groups[0]['lr'] = max(lr, self.lr_min)

            if ((episode+1) % self.save_every == 0) or ((episode+1) == self.episodes):
                self.save_model('latest')
                self.log_dict['train'].save_logs()

            if ((episode+1) % self.print_every == 0) or ((episode+1) % self.valid_every == 0):
                self._print_log()

            if (episode+1) % self.valid_every == 0:
                self.save_model('latest')
                self.log_dict['train'].save_logs()
                self.visualize_log()
                print("=== valid ===")
                self.valid()
                print("="*30)

        self.visualize_log()
        log_str = f"Train completed. total avg win rate: {round(np.mean(self.log_dict['train'].clear_list), 3)}"
        print(log_str)
        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)


    def valid(self):
        self.mode = 'valid'
        log_str = ""
        print("Start valid - latest model")
        self.agent.model.to(self.device)

        for i in range(2):
            self.cur_epi_dict['valid'] = 0
            self.log_dict['valid'].reset()
            if i == 1:  # best
                self.load_model('best')
                self.agent.best_model.to(self.device)

            # latest
            for episode in range(self.valid_total_episodes):
                self.cur_epi_dict['valid'] += 1
                # reset
                state, done, clear, total_reward, cnt, _ = self._game_reset()

                # 게임 종료까지 반복
                while not done:
                    cnt+=1

                    if i == 1:  # best
                        action = self.agent.get_action_best(state)
                    else:       # latest
                        action = self.agent.get_action_latest(state)

                    state, reward, done, clear = self.env.step(action)
                    total_reward += reward

                    # count 제한 : 전체 칸 수 - 지뢰 개수
                    if not done:
                        done = self._check_cnt_limit(cnt)

                    if done or clear:
                        break

                # 평가지표
                self.log_dict['valid'].update_logs([total_reward, clear, cnt])

                if (episode+1) % self.print_every == 0:
                    self._print_log()

            if i == 1:  # best
                best_score = np.mean(self.log_dict['valid'].reward_list)
                log_str += f"\n[Best model valid result] Avg win rate: {round(np.mean(self.log_dict['valid'].clear_list), 3)} / Avg Reward: {round(best_score, 3)} / Avg cnt: {round(np.mean(self.log_dict['valid'].cnt_list), 3)} / Avg RPC: {round(np.mean(self.log_dict['valid'].rpc_list), 3)}" + "\n"
                print(f"Valid best model completed. Avg win rate: {round(np.mean(self.log_dict['valid'].clear_list), 3)} / Avg Reward: {round(best_score, 3)} / Avg cnt: {round(np.mean(self.log_dict['valid'].cnt_list), 3)} / Avg RPC: {round(np.mean(self.log_dict['valid'].rpc_list), 3)}")
                
                path = f"{self.path_dict['game_imgs']}/valid_best.png"
                visualize_state(state = self.env.present_state, save_path = path)
                log_str += f"\nSave game image at \'{path}\'"

                with open(self.path_dict['log_message'], "a") as f:
                    f.write(log_str)

                log_str = ""

            else:       # latest
                latest_score = np.mean(self.log_dict['valid'].reward_list)
                log_str += f"\n[Latest model valid result] Avg win rate: {round(np.mean(self.log_dict['valid'].clear_list), 3)} / Avg Reward: {round(latest_score, 3)} / Avg cnt: {round(np.mean(self.log_dict['valid'].cnt_list), 3)} / Avg RPC: {round(np.mean(self.log_dict['valid'].rpc_list), 3)}" + "\n"
                print(f"Valid latest model completed. Avg win rate: {round(np.mean(self.log_dict['valid'].clear_list), 3)} / Avg Reward: {round(latest_score, 3)} / Avg cnt: {round(np.mean(self.log_dict['valid'].cnt_list), 3)} / Avg RPC: {round(np.mean(self.log_dict['valid'].rpc_list), 3)}")
                
                path = f"{self.path_dict['game_imgs']}/valid_latest.png"
                visualize_state(state = self.env.present_state, save_path = path)
                log_str += f"\nSave game image at \'{path}\'"

                with open(self.path_dict['log_message'], "a") as f:
                    f.write(log_str)

                log_str = ""

        if latest_score > best_score:
            log_str += f"Update best model. latest model score: {latest_score} > best model score: {best_score}\n"
            self.save_model('best')
            self.log_dict['valid'].latest_update = 0
        
        else:
            self.log_dict['valid'].latest_update += 1

        if self.log_dict['valid'].latest_update == self.model_criteria:
            self.change_model()
            self.log_dict['valid'].latest_update = 0

        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)

        self.log_dict['valid'].save_logs()


    def test(self, num_episodes):
        self.mode = 'test'
        self.agent.best_model.to(self.device)
        self.test_total_episodes = num_episodes
        self.cur_epi_dict['test'] = 0
        print("Test start")

        for episode in range(self.test_total_episodes):
            self.cur_epi_dict['test'] += 1
            # reset
            state, done, clear, total_reward, cnt, _ = self._game_reset()

            # 게임 종료까지 반복
            while not done:
                cnt+=1

                action = self.agent.get_action_best(state)
                state, reward, done, clear = self.env.step(action)
                total_reward += reward

                # count 제한 : 전체 칸 수 - 지뢰 개수
                if not done:
                    done = self._check_cnt_limit(cnt)

                if done or clear:
                    break

            # 평가지표
            self.log_dict['test'].update_logs([total_reward, clear, cnt])

            if (episode+1) % self.print_every == 0:
                self._print_log()

        self.log_dict['test'].save_logs()
        self.visualize_log()
        print(f"Test completed. Avg win rate: {round(np.mean(self.log_dict['test'].clear_list), 3)} / Avg Reward: {round(self.log_dict['test'].reward_list, 3)} / Avg cnt: {round(np.mean(self.log_dict['test'].cnt_list), 3)} / Avg RPC: {round(np.mean(self.log_dict['test'].rpc_list), 3)}")
