import pickle
from pathlib import Path
import numpy as np
from typing import Tuple

from environment import Environment
from dqn_agent import DQN_Agent

from utils import visualize_train_log, visualize_test_log, visualize_state
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

        self.device = DEVICE    # device: 'mps' or 'cpu' or 'cuda'

        self.cnt_limit = self.env.nrow * self.env.ncol - self.env.num_mine  # 행동 횟수 제한
        self.train_start = TRAIN_START  # 리플레이 메모리에 쌓인 데이터가 train_start 이상일 때 학습 시작

        # 경로 생성
        self.folder_name = FOLDER_NAME
        self.path = f"{PATH}/{FOLDER_NAME}"
        self.create_path()

        # 에피소드
        self.episodes = EPISODES
        self.valid_total_episodes = VALID_EPISODES
        self.test_total_episodes = 0

        # 하이퍼파라미터
        self.update_target_every = UPDATE_TARGET_EVERY
        self.print_every = PRINT_EVERY
        self.save_every = SAVE_EVERY
        self.valid_every = VALID_EVERY
        self.lag = LAG
        self.model_criteria = MODEL_CRITERIA

        self.lr_min = LEARN_MIN
        self.lr_decay = LEARN_DECAY
        self.lr_epoch = LEARN_EPOCH

        # 현재 학습 상태 추적 (train, valid, test)
        self.mode = "train"
        
        # 학습 지표 관리 객체 생성
        self.log_dict = dict({'train': Log(MODE= 'train', FOLDER_NAME= self.folder_name, PATH=self.path_dict['logs']), 
                            'valid': Log(MODE= 'valid', FOLDER_NAME= self.folder_name, PATH=self.path_dict['logs'])
                            })

        # 현재 에피소드 횟수 추적
        self.cur_epi_dict = dict({'train': 0, 'valid': 0})


    def create_path(self):
        '''
        필요한 폴더, 파일 경로 생성
        '''
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

        # 학습 로그 파일에 제목 출력
        if not Path(self.path_dict['log_message']).exists():
            with open(self.path_dict['log_message'], "w") as f:
                f.write(f"Training Log for {self.folder_name}\n")
                f.write("="*50 + "\n")
            
        # 학습 로그 파일에 모든 경로 출력
        log_str = "< path dict >\n"

        for item in self.path_dict.items():
            log_str += str(item) + "\n"

        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)


    def load_model(self, model_name):
        '''
        model_name에 따라 agent의 신경망을 교체
        '''
        file_path = f"{self.path_dict['model']}/{model_name}.pkl"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_str = ""

        if not path.exists():
            # best model이 존재하지 않는 경우 latest model로 지정
            if model_name == 'best':
                with open(path, 'wb') as f:
                    pickle.dump(self.agent.model.state_dict(), f)
                    log_str += "\nCreate best model"
                    
            else:
                # latest model이 존재하지 않는 경우 agent의 초기화된 모델 그대로 두기
                log_str += "\nCreate new model"
                return

        with open(path, 'rb') as f:
            model_param = pickle.load(f)

        if model_name == 'best':    # best model 로드
            self.agent.best_model.load_state_dict(model_param)
            self.agent.best_model.to(self.device)

        else:   # 학습하는 메인 model 로드
            self.agent.model.load_state_dict(model_param)
            self.agent.target_model.load_state_dict(model_param)

            self.agent.model.to(self.device)
            self.agent.target_model.to(self.device)
        
        # print log
        log_str += f"\nModel loaded from \'{file_path}\'"
        print(log_str)
        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)


    def save_model(self, model_name):
        '''
        model_name에 따라 agent의 model을 지정된 경로에 pkl 형식으로 저장
        '''
        self.agent.model.to("cpu")

        file_path = f"{self.path_dict['model']}/{model_name}.pkl"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(path, 'wb') as f:
            pickle.dump(self.agent.model.state_dict(), f)
        
        # print log
        log_str = f"\nModel saved to \'{file_path}\'"
        # print(log_str)
        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)

    
    def change_model(self):
        '''
        valid 시 model criteria에 따라 현재 모델을 이전 best model로 교체
        '''
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

        # print log
        log_str = "\nAgent model changed to best model."
        # print(log_str) 
        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)

    
    def _save_memory(self):
        '''
        현재 리플레이 메모리를 pkl 형식으로 저장
        '''
        with open(self.path_dict['memory'], 'wb') as f:
            pickle.dump(self.agent.memory, f)

    
    def _load_memory(self):
        '''
        리플레이 메모리를 불러옴
        '''
        with open(self.path_dict['memory'], 'rb') as f:
            self.agent.memory = pickle.load(f)


    def reset(self):
        '''
        현재 학습에 사용되는 env, agent, 추적 변수 모두 초기화.
        신경망은 저장된 latest model이 존재하는 경우 그걸로 변경.
        '''
        self.env.reset()
        self.agent.reset()
        self.load_model('latest')

        for key in self.cur_epi_dict.keys():
            self.cur_epi_dict[key] = 0
            self.log_dict[key].reset()
    
    def _continue_train(self):
        with open(f"{self.path_dict['logs']}/train.pkl") as f:
            self.log_dict['train'].continue_logs()
        
        self.cur_epi_dict['train'] = len(self.log_dict['train'].clear_list)

        avg_cnt = np.mean(self.log_dict['train'].cnt_list)
        self.agent.epsilon = max(self.agent.epsilon_init * (self.agent.epsilon_decay ** (self.cur_epi_dict['train'] * avg_cnt)), self.agent.epsilon_min)
        self.agent.lr = self.agent.lr_init * (self.lr_decay ** (self.cur_epi_dict['train'] // self.lr_epoch))


    def _game_reset(self):
        '''
        한 에피소드를 시작할 때 게임 정보 초기화
        '''
        self.env.reset()

        state = self.env.present_state.copy()
        done = False
        clear = False
        total_reward = 0
        cnt = 0
        loss = 0

        return state, done, clear, total_reward, cnt, loss

    
    def _check_cnt_limit(self, cnt:int):
        '''
        행동 횟수 제한 확인
        '''
        done = True if cnt > self.cnt_limit else False
        return done


    def _print_log(self):
        '''
        trainig_log.txt에 학습 지표 출력
        '''
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
        '''
        학습 지표 그래프 출력
        '''
        key = self.mode
        df = self.log_dict[key].load_logs()
        cur_epi = self.cur_epi_dict[key]
        path = f"{self.path_dict['graph']}/{key}_{cur_epi}.png"

        if key == 'train':
            visualize_train_log(df = df, lag = self.lag, save_path = path)

        else:
            visualize_test_log(df = df, lag = self.lag, save_path = path)

        # print log
        log_str = "\nComplete visualizing train log."
        # print(log_str)

        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)
        


    def train(self):
        '''
        전체 학습을 실행하는 함수
        '''
        self.reset()
        if Path(f"{self.path_dict['model']}/best.pkl").exists():
            self._continue_train()
            log_str = "\nTrain continue"
            print(log_str)
            episodes = self.episodes - self.cur_epi_dict['train']

            with open(self.path_dict['log_message'], "a") as f:
                f.write(log_str)

        else:
            episodes = self.episodes
            print("Train start")

        for _ in range(episodes):
            self.mode = 'train'
            self.agent.model.to(self.device)
            self.cur_epi_dict['train'] += 1
            # reset 1 episode
            state, done, clear, total_reward, cnt, loss = self._game_reset()

            # 게임 종료까지 반복
            while not done:
                cnt+=1
                state = self.env.present_state.copy()
                action = int(self.agent.get_action(state))
                next_state, reward, done, clear = self.env.step(action)
                total_reward += reward

                # 행동 횟수 제한 : 전체 칸 수 - 지뢰 개수
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
            if self.cur_epi_dict['train'] % self.update_target_every == 0:
                self.agent.update_target_model()

            # lr 조절
            if (self.cur_epi_dict['train']+1) % self.lr_epoch == 0:
                lr = self.agent.optimizer.param_groups[0]['lr'] * self.lr_decay
                self.agent.optimizer.param_groups[0]['lr'] = max(lr, self.lr_min)
                self.agent.lr = lr

            # model, 학습지표를 파일로 저장
            if (self.cur_epi_dict['train'] % self.save_every == 0) or (self.cur_epi_dict['train'] == self.episodes):
                self.save_model('latest')
                self.log_dict['train'].save_logs()

            # 학습 로그 출력
            if (self.cur_epi_dict['train'] % self.print_every == 0) or (self.cur_epi_dict['train'] % self.valid_every == 0):
                self._print_log()

            # valid
            if (self.cur_epi_dict['train']+1) % self.valid_every == 0:
                self.save_model('latest')
                self.log_dict['train'].save_logs()
                self.visualize_log()    # 학습 지표 그래프 출력
                print("=== valid ===")
                self.valid()
                print("="*30)

        # 학습 지표 그래프 출력
        self.visualize_log()

        # print log
        log_str = f"Train completed. total avg win rate: {round(np.mean(self.log_dict['train'].clear_list), 3)}"
        print(log_str)
        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)


    def valid(self):
        '''
        valid를 실행하는 함수
        latest model, best model 순서대로 각각 한 번씩 valid한다.
        '''
        self.mode = 'valid'
        log_str = ""
        print("Start valid - latest model")
        self.agent.model.to(self.device)

        for i in range(2):
            self.cur_epi_dict['valid'] = 0
            self.log_dict['valid'].reset(new=False)

            if i == 1:  # best
                cur_model = 'best'
                self.load_model('best')
                self.agent.best_model.to(self.device)

            else:   # latest
                cur_model = 'latest'

            # valid 1회 수행
            for episode in range(self.valid_total_episodes):
                self.cur_epi_dict['valid'] += 1
                # reset
                state, done, clear, total_reward, cnt, _ = self._game_reset()

                # 게임 종료까지 반복
                while not done:
                    cnt+=1

                    action = self.agent.get_action_test(state, cur_model)

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

            self.log_dict['valid'].save_logs()

            # model 평가는 RPC(Reward per Cnt)를 기준으로 함
            if i == 1:  # best
                best_score = np.mean(self.log_dict['valid'].rpc_list)   # valid 결과 best model의 점수
                log_str += f"\n[Best model valid result] Avg win rate: {round(np.mean(self.log_dict['valid'].clear_list), 3)} / Avg Reward: {round(np.mean(self.log_dict['valid'].reward_list), 3)} / Avg cnt: {round(np.mean(self.log_dict['valid'].cnt_list), 3)} / Avg RPC: {round(best_score, 3)}" + "\n"
                print(f"Valid best model completed. Avg win rate: {round(np.mean(self.log_dict['valid'].clear_list), 3)} / Avg Reward: {round(best_score, 3)} / Avg cnt: {round(np.mean(self.log_dict['valid'].cnt_list), 3)} / Avg RPC: {round(np.mean(self.log_dict['valid'].rpc_list), 3)}")

                cur_epi = self.cur_epi_dict['train']
                # 맨 마지막 게임의 게임 결과 화면을 png로 저장
                path = f"{self.path_dict['game_imgs']}/valid_best_{cur_epi}.png"
                visualize_state(state = self.env.present_state, save_path = path)
                log_str += f"Save game image at \'{path}\'"

                with open(self.path_dict['log_message'], "a") as f:
                    f.write(log_str)

                log_str = ""

            else:       # latest
                latest_score = np.mean(self.log_dict['valid'].rpc_list) # valid 결과 latest model의 점수
                log_str += f"\n[Latest model valid result] Avg win rate: {round(np.mean(self.log_dict['valid'].clear_list), 3)} / Avg Reward: {round(np.mean(self.log_dict['valid'].reward_list), 3)} / Avg cnt: {round(np.mean(self.log_dict['valid'].cnt_list), 3)} / Avg RPC: {round(latest_score, 3)}" + "\n"
                print(f"Valid latest model completed. Avg win rate: {round(np.mean(self.log_dict['valid'].clear_list), 3)} / Avg Reward: {round(latest_score, 3)} / Avg cnt: {round(np.mean(self.log_dict['valid'].cnt_list), 3)} / Avg RPC: {round(np.mean(self.log_dict['valid'].rpc_list), 3)}")
                
                cur_epi = self.cur_epi_dict['train']
                path = f"{self.path_dict['game_imgs']}/valid_latest_{cur_epi}.png"
                visualize_state(state = self.env.present_state, save_path = path)
                log_str += f"Save game image at \'{path}\'"

                with open(self.path_dict['log_message'], "a") as f:
                    f.write(log_str)

                log_str = ""

        # Evaluate model. latest model의 점수가 더 높은 경우 best model로 업데이트
        if latest_score > best_score:
            log_str += f"\n>>>> Update best model. latest model score: {latest_score} > best model score: {best_score}\n"
            self.save_model('best')
            self.log_dict['valid'].latest_update = 0
        
        else:
            self.log_dict['valid'].latest_update += 1

        # 만약 기준 valid 횟수 동안 best model이 업데이트되지 않은 경우
        # 학습이 제대로 되고 있지 않다고 판단하고 
        # 현재 모델을 기존의 best model로 변경
        if self.log_dict['valid'].latest_update >= self.model_criteria:
            self.change_model()
            log_str += "\t!!!! Model changed !!!!\n"
            self.log_dict['valid'].latest_update = 0

        with open(self.path_dict['log_message'], "a") as f:
            f.write(log_str)