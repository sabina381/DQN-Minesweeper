import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple

from environment import Environment
from dqn_agent import DQN_Agent

from visualization import *
########################################
class Trainer:
    def __init__(self, 
                # Trainer
                FOLDER_NAME:str, PATH:str, EPISODES:int, VALID_EPISODES:int,
                UPDATE_TARGET_EVERY:int, PRINT_EVERY:int, SAVE_EVERY:int, VALID_EVERY:int,
                TRAIN_START:int, LEARN_MIN:float, LEARN_DECAY:float, LEARN_EPOCH:float,
                # Environment
                GRIDWORLD_SIZE:Tuple, NUM_MINE:int, REWARD_DICT:dict, DONE_DICT:dict, COLOR_DICT:dict,
                # DQN_Agent
                STATE_TYPE:str, EPSILON:float, EPSILON_DECAY:float, EPSILON_MIN:float, BATCH_SIZE:int, 
                GAMMA:float, MEM_SIZE:int, LEARN_MAX:float, CONV_UNITS:int, DEVICE, LOSS_FUNC, OPTIMIZER
                ):

        self.env = Environment(GRIDWORLD_SIZE= GRIDWORLD_SIZE, 
                            NUM_MINE= NUM_MINE, 
                            REWARD_DICT= REWARD_DICT, 
                            DONE_DICT= DONE_DICT, 
                            COLOR_DICT= COLOR_DICT)

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
        self.episodes = EPISODES

        self.update_target_every = UPDATE_TARGET_EVERY
        self.print_every = PRINT_EVERY
        self.save_every = SAVE_EVERY
        self.valid_every = VALID_EVERY

        self.mode = "train"

        self.reset_logs()

        self.valid_total_episodes = VALID_EPISODES
        self.test_total_episodes = 0

    def reset_valid_logs(self):
        self.valid_epi = 0

        self.valid_rewards_list = []
        self.valid_avg_rewards_list = []
        self.valid_mid_rewards_list = []

        self.valid_clear_list = []
        self.valid_avg_clear_list = []
        self.valid_mid_clear_list = []

        self.valid_cnt_list = []
        self.valid_avg_cnt_list = []
        self.valid_mid_cnt_list = []


    def reset_logs(self):
        # train
        self.current_epi = 0

        self.rewards_list = []
        self.avg_rewards_list = []
        self.mid_rewards_list = []

        self.clear_list = []
        self.avg_clear_list = []
        self.mid_clear_list = []

        self.cnt_list = []
        self.avg_cnt_list = []
        self.mid_cnt_list = []

        self.loss_list = []
        self.avg_loss_list = []
        self.mid_loss_list = []

        self.lr_list = []

        # valid
        self.reset_valid_logs()

        # test
        self.test_epi = 0

        self.test_rewards_list = []
        self.test_avg_rewards_list = []
        self.test_mid_rewards_list = []

        self.test_clear_list = []
        self.test_avg_clear_list = []
        self.test_mid_clear_list = []

        self.test_cnt_list = []
        self.test_avg_cnt_list = []
        self.test_mid_cnt_list = []


    def load_model(self, model_name):
        file_path = f"{self.path}/{self.folder_name}_model_{model_name}.pkl"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            if model_name == 'best':
                with open(path, 'wb') as f:
                    pickle.dump(self.agent.model.state_dict(), f)
                    print("Create best model")
                    
            else:
                print("Create new model")
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
            
        print(f"Model loaded from \'{file_path}\'")


    def save_model(self, model_name):
        self.agent.model.to("cpu")

        file_path = f"{self.path}/{self.folder_name}_model_{model_name}.pkl"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(path, 'wb') as f:
            pickle.dump(self.agent.model.state_dict(), f)
        
        print(f"Model saved to \'{file_path}\'")

    
    def _save_memory(self):
        file_path = f"{self.path}/{self.folder_name}_memory.pkl"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self.agent.memory, f)


    def reset(self):
        self.env.reset()
        self.agent.reset()
        self.load_model('latest')


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


    def _update_log(self, logs:list):
        if self.mode == 'train':
            total_reward, clear, cnt, loss = logs

            self.rewards_list.append(total_reward)
            self.avg_rewards_list.append(np.mean(self.rewards_list[-100:]))
            self.mid_rewards_list.append(np.median(self.rewards_list[-100:]))

            self.clear_list.append(clear)
            self.avg_clear_list.append(np.mean(self.clear_list[-100:]))
            self.mid_clear_list.append(np.median(self.clear_list[-100:]))

            self.cnt_list.append(cnt)
            self.avg_cnt_list.append(np.mean(self.cnt_list[-100:]))
            self.mid_cnt_list.append(np.median(self.cnt_list[-100:]))

            self.loss_list.append(loss)
            self.avg_loss_list.append(np.mean(self.loss_list[-100:]))
            self.mid_loss_list.append(np.median(self.loss_list[-100:]))

            self.lr_list.append(self.agent.optimizer.param_groups[0]['lr'])

        elif self.mode == 'valid':
            total_reward, clear, cnt = logs
            self.valid_rewards_list.append(total_reward)
            self.valid_avg_rewards_list.append(np.mean(self.valid_rewards_list[-100:]))
            self.valid_mid_rewards_list.append(np.median(self.valid_rewards_list[-100:]))

            self.valid_clear_list.append(clear)
            self.valid_avg_clear_list.append(np.mean(self.valid_clear_list[-100:]))
            self.valid_mid_clear_list.append(np.median(self.valid_clear_list[-100:]))

            self.valid_cnt_list.append(cnt)
            self.valid_avg_cnt_list.append(np.mean(self.valid_cnt_list[-100:]))
            self.valid_mid_cnt_list.append(np.median(self.valid_cnt_list[-100:]))

        elif self.mode == 'test':
            total_reward, clear, cnt, loss = logs
            self.test_rewards_list.append(total_reward)
            self.test_avg_rewards_list.append(np.mean(self.test_rewards_list[-100:]))
            self.test_mid_rewards_list.append(np.median(self.test_rewards_list[-100:]))

            self.test_clear_list.append(clear)
            self.test_avg_clear_list.append(np.mean(self.test_clear_list[-100:]))
            self.test_mid_clear_list.append(np.median(self.test_clear_list[-100:]))

            self.test_cnt_list.append(cnt)
            self.test_avg_cnt_list.append(np.mean(self.test_cnt_list[-100:]))
            self.test_mid_cnt_list.append(np.median(self.test_cnt_list[-100:]))


    def save_log(self):
        df = pd.DataFrame()

        if self.mode == 'train':
            df['rewards'] = self.rewards_list
            df['avg_rewards'] = self.avg_rewards_list
            df['mid_rewards'] = self.mid_rewards_list
            df['clear'] = self.clear_list
            df['avg_clear'] = self.avg_clear_list
            df['mid_clear'] = self.mid_clear_list
            df['cnt'] = self.cnt_list
            df['avg_cnt'] = self.avg_cnt_list
            df['mid_cnt'] = self.mid_cnt_list
            df['loss'] = self.loss_list
            df['avg_loss'] = self.avg_loss_list
            df['mid_loss'] = self.mid_loss_list
            df['lr'] = self.lr_list

        elif self.mode == 'valid':
            df['rewards'] = self.valid_rewards_list
            df['avg_rewards'] = self.valid_avg_rewards_list
            df['mid_rewards'] = self.valid_mid_rewards_list
            df['clear'] = self.valid_clear_list
            df['avg_clear'] = self.valid_avg_clear_list
            df['mid_clear'] = self.valid_mid_clear_list
            df['cnt'] = self.valid_cnt_list
            df['avg_cnt'] = self.valid_avg_cnt_list
            df['mid_cnt'] = self.valid_mid_cnt_list

        elif self.mode == 'test':
            df['rewards'] = self.test_rewards_list
            df['avg_rewards'] = self.test_avg_rewards_list
            df['mid_rewards'] = self.test_mid_rewards_list
            df['clear'] = self.test_clear_list
            df['avg_clear'] = self.test_avg_clear_list
            df['mid_clear'] = self.test_mid_clear_list
            df['cnt'] = self.test_cnt_list
            df['avg_cnt'] = self.test_avg_cnt_list
            df['mid_cnt'] = self.test_mid_cnt_list

        file_path = f"{self.path}/{self.folder_name}_{self.mode}_log.pkl"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(df, f)
            
        print(f"{self.mode} logs saved to \'{file_path}\'")

    
    def load_log(self):
        file_path = f"{self.path}/{self.folder_name}_{self.mode}_log.pkl"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'rb') as f:
            train_log = pickle.load(f)  # df
        
        return train_log


    def _print_log(self):
        if self.mode == 'train':
            print("="*30, end="\n")
            print(f"[{self.current_epi}/{self.episodes}] epsilon: {round(self.agent.epsilon, 5)} | lr: {round(self.agent.lr, 5)}", end="\n\t")
            print(f"- Avg Clear: {round(np.mean(self.clear_list[-self.print_every:]), 3)}", end="\t")
            print(f"- Cnt: {round(np.mean(self.cnt_list[-self.print_every:]), 3)}/{round(np.median(self.cnt_list[-self.print_every:]), 3)}", end="\t")
            print(f"- Reward: {round(np.mean(self.rewards_list[-self.print_every:]), 3)}/{round(np.median(self.rewards_list[-self.print_every:]), 3)}", end="\t")
            print(f"- Loss: {round(np.mean(self.loss_list[-self.print_every:]), 3)}/{round(np.median(self.loss_list[-self.print_every:]), 3)}", end="\n")

        elif self.mode == 'valid':
            print("=valid"*5, end="\n")
            print(f"[{self.valid_epi}/{self.valid_total_episodes}]", end="\n\t")
            print(f"- Avg Clear: {round(np.mean(self.valid_clear_list[-self.print_every:]), 3)}", end="\t")
            print(f"- Cnt: {round(np.mean(self.valid_cnt_list[-self.print_every:]), 3)}/{round(np.median(self.valid_cnt_list[-self.print_every:]), 3)}", end="\t")
            print(f"- Reward: {round(np.mean(self.valid_rewards_list[-self.print_every:]), 3)}/{round(np.median(self.valid_rewards_list[-self.print_every:]), 3)}", end="\n")
            self.env.render(self.env.present_state)

        elif self.mode == 'test':
            print("="*30, end="\n")
            print(f"[{self.test_epi}/{self.test_total_episodes}]", end="\n\t")
            print(f"- Avg Clear: {round(np.mean(self.test_clear_list[-self.print_every:]), 3)}", end="\t")
            print(f"- Cnt: {round(np.mean(self.test_cnt_list[-self.print_every:]), 3)}/{round(np.median(self.test_cnt_list[-self.print_every:]), 3)}", end="\t")
            print(f"- Reward: {round(np.mean(self.test_rewards_list[-self.print_every:]), 3)}/{round(np.median(self.test_rewards_list[-self.print_every:]), 3)}", end="\n")
            self.env.render(self.env.present_state)

        
    def visualize_log(self):
        file_path = f"{self.path}/{self.folder_name}_{self.mode}_log.pkl"

        if self.mode == 'train':
            visualize_train_log(file_path)

        else:
            visualize_test_log(file_path)

        print("Visualizing complete")


    def train(self):
        self.reset()
        print("Train start")

        for episode in range(self.episodes):
            self.mode = 'train'
            self.agent.model.to(self.device)
            self.current_epi += 1
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
            self._update_log([total_reward, clear, cnt, loss])

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
                self.save_log()

            if (episode+1) % self.print_every == 0:
                self._print_log()

            if (episode+1) % self.valid_every == 0:
                self.visualize_log()

                self.valid()

        self.visualize_log()
        print(f"Train completed. total avg win rate: {round(np.mean(self.clear_list), 3)}")


    def valid(self):
        self.mode = 'valid'
        print("Start valid - latest model")
        self.agent.model.to(self.device)

        for i in range(2):
            if i == 1:  # best
                self.load_model('best')
                self.agent.best_model.to(self.device)

            # latest
            for episode in range(self.valid_total_episodes):
                self.valid_epi += 1
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
                self._update_log([total_reward, clear, cnt])

                if (episode+1) % self.print_every == 0:
                    self._print_log()

            if i == 1:  # best
                best_score = np.mean(self.valid_rewards_list)
                print(f"Valid best model completed. Average win rate: {round(np.mean(self.valid_clear_list), 3)} / Average cnt: {round(np.mean(self.valid_cnt_list), 3)} / Average Reward: {round(best_score, 3)}")
                
            else:       # latest
                latest_score = np.mean(self.valid_rewards_list)
                print(f"Valid latest model completed. Average win rate: {round(np.mean(self.valid_clear_list), 3)} / Average cnt: {round(np.mean(self.valid_cnt_list), 3)} / Average Reward: {round(latest_score, 3)}")
            
            self.reset_valid_logs()

        if latest_score > best_score:
            print(f"Update best model. latest model score: {latest_score} > best model score: {best_score}")
            self.save_model('best')


    def test(self, num_episodes):
        self.mode = 'test'
        self.agent.best_model.to(self.device)
        self.test_total_episodes = num_episodes
        print("Test start")

        for episode in range(self.test_total_episodes):
            self.test_epi += 1
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
            self._update_log([total_reward, clear, cnt])

            if (episode+1) % self.print_every == 0:
                self._print_log()

        self.save_log()
        self.visualize_log()
        print(f"Test completed. Average win rate: {round(np.mean(self.test_clear_list), 3)} / Average cnt: {round(np.mean(self.test_cnt_list), 3)} / Average Reward: {round(np.mean(self.test_rewards_list), 3)}")
