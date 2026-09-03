# Minesweeper (DQN)
DQN(Deep Q-Network)으로 지뢰찾기 게임을 스스로 클리어하는 에이전트를 학습시킨 강화학습 프로젝트.

## Overview
- **문제정의**: 주변 지뢰 개수라는 제한된 시그널만으로 숨겨진 지뢰(risk)를 피해 가장 안전한 칸을 선택하는 최적화 문제.
- **목표**: 강화학습의 핵심 방법론인 DQN을 직접 구현해, 알고리즘적으로 정답을 계산하기는 어려운 지뢰찾기 게임을 에이전트가 스스로 풀도록 학습시킨다.
- 동아리 강화학습 팀프로젝트(2인, 개인 파트 진행)로 시작해, 이후 구조 개선·버그 수정·실험 확장을 거쳐 개인적으로 리팩토링·디벨롭한 버전.

## Features
- one-hot encoding으로 정규화한 state (11채널)와 BFS 기반 빈 칸 자동 확장을 갖춘 지뢰찾기 환경
- Batch normalization을 적용한 CNN 기반 Q-network
- ε-greedy 탐험 + replay memory 기반 DQN 에이전트
- `config.py` 하나로 모든 하이퍼파라미터를 관리하고, 실험마다 `config.txt`/`config.pkl`로 자동 저장
- 학습 중 `best`/`latest` 모델을 분리 관리하며, 성능이 일정 기준 이상 정체되면 이전 `best` 모델로 롤백하는 안정화 로직
- `Trainer`(학습) / `Tester`(`Trainer` 상속, 평가) / `Log`(지표 기록) 클래스로 분리된 학습·평가 파이프라인
- 지뢰 개수·첫 클릭 안전 여부(`FIRST_MINE`)를 조합해 여러 환경에서 반복 테스트하고, R로 통계 검정까지 수행하는 결과 분석 체계

## Tech Stack
- Language: Python 3.13
- Deep Learning: PyTorch (`torch`, `torch.nn`, `torch.optim`)
- Config: `easydict`
- 데이터/시각화: `numpy`, `pandas`, `matplotlib`, `seaborn`, `IPython.display`
- 통계 분석: R (Chi-squared test, Wilcoxon rank sum test, Welch's t-test 등)

## Environment & Reward Design
State는 (nrow, ncol) 보드를 11채널 one-hot으로 인코딩(지뢰~빈칸까지 각 값을 채널로 분리)하고, 클릭한 칸이 0(빈 칸)이면 BFS로 주변을 연쇄 오픈한다.

| Event | 상황 | Reward |
|---|---|---|
| `mine` | 지뢰 선택 (게임오버) | -1 |
| `clear` | 지뢰를 제외한 전체 칸 오픈 | +1 |
| `overlapped` | 이미 연 칸 재선택 | -1 |
| `guess` | 주변이 모두 닫힌 칸 선택 (첫 수 제외) | +0.3 |
| `empty` | 그 외 안전한 선택 | +1 |

처음엔 좋은 행동에만 +1, 나머지는 0을 주는 양수 보상 체계로 시작했지만 클리어 여부와 보상의 상관관계가 약해 학습이 느렸고 "중복 선택 시 게임 종료" 조건과도 맞지 않아 폐기했다. 이후 게임오버·중복 행동엔 -1, 좋은 행동·클리어엔 +1, 애매한 추측엔 소량의 +0.3을 주는 음수 보상 체계로 바꾸자 학습 초반부터 안정적인 속도로 성능이 향상됐다.

## Architecture
`main.ipynb`에서 `config.py`의 파라미터를 적용해 학습을 실행하는 구조로, 기능별로 모듈을 분리했다.

| 모듈 | 역할 |
|---|---|
| `environment.py` | 지뢰찾기 게임 환경 |
| `dqn_agent.py` | DQN 알고리즘 기반 에이전트 |
| `net.py` | 행동 선택에 사용하는 CNN Q-network |
| `utils.py` | state 정규화 + 학습 지표/게임 시각화 함수 |
| `log.py` | 학습/검증/테스트 지표 관리 |
| `trainer.py` | 학습 총괄 — 모델·지표 저장, 학습 로그 출력 |
| `tester.py` | 평가 총괄 (`trainer.py` 상속) |
| `config.py` | 하이퍼파라미터·경로 설정 |

## Hyperparameters
최종 모델(A2) 기준 주요 설정값:

| 파라미터 | 값 | 비고 |
|---|---|---|
| `STATE_TYPE` | `one-hot` | `original`/`normalization` 옵션도 있으나 실험엔 미사용 |
| `CONV_UNITS` | 64 | 128로 늘렸을 때 연산 속도만 느려지고 성능은 오히려 낮아짐 |
| `BATCH_SIZE` | 64 | 128은 학습이 거의 되지 않음 |
| `LEARN_MAX → LEARN_MIN` | 0.001 → 0.0001 (50,000 episode마다 0.5배 감쇠) | lr을 0.01→0.001로 낮춘 시점에 가장 큰 성능 향상 |
| `GAMMA` | 0.1 | 지뢰찾기처럼 한 게임이 짧은 태스크엔 낮은 할인율이 적합 |
| `EPSILON_MIN` | 0.01 | 후반까지 1% 확률로 탐험을 유지해 무한 에피소드 방지 |
| `MODEL_CRITERIA` | 3 | valid 3회 동안 `best` 모델이 갱신되지 않으면 `best`로 롤백 |

`best`/`latest` 모델 이원화 도입 전에는 15만 episode를 넘기면서 성능이 급락하는 경우가 잦았지만, 도입 후에는 30만 episode까지 학습해도 성능이 꾸준히 우상향했다.

## Project Structure
```
Minesweeper/
├── codes/
│   ├── config.py       # 전체 하이퍼파라미터 + 경로 설정 (EasyDict)
│   ├── environment.py  # 지뢰찾기 게임 환경
│   ├── net.py           # CNN Q-network
│   ├── dqn_agent.py     # DQN 에이전트
│   ├── trainer.py       # 학습 오케스트레이션
│   ├── tester.py        # 평가 오케스트레이션 (Trainer 상속)
│   ├── log.py           # 학습/검증/테스트 지표 기록
│   ├── utils.py         # state 정규화 + 시각화 함수 모음
│   ├── main.ipynb       # 학습/테스트 실행 노트북
│   └── test_main.ipynb  # 테스트 전용 노트북
├── final_model/         # 최종 선정 모델(A2)
│   ├── config.txt        # 학습 설정
│   ├── training_curve.png # 학습 곡선
│   └── models/            # 모델 가중치 (best.pkl, latest.pkl)
└── README.md
```
> 학습 실험별 원본 로그·체크포인트 이미지와 R 통계 분석 스크립트·데이터는 용량 문제로 리포지토리에는 포함하지 않고 로컬에서 관리합니다.

## Installation & Usage
필요한 패키지: `torch`, `easydict`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `ipython`(주피터 노트북 실행용)

```bash
pip install torch easydict numpy pandas matplotlib seaborn ipython
```

`codes/config.py`에서 실험 설정(`FOLDER_NAME`, `EPISODES`, `NUM_MINE`, `FIRST_MINE` 등)을 지정한 뒤, `codes/main.ipynb`에서 학습·평가를 실행합니다.

```python
from trainer import Trainer
from tester import Tester
from config import CONFIG

# 학습
trainer = Trainer(FOLDER_NAME=CONFIG.FOLDER_NAME, PATH=CONFIG.PATH, EPISODES=CONFIG.EPISODES, ...)
trainer.train()

# 평가
tester = Tester(FOLDER_NAME=CONFIG.FOLDER_NAME, PATH=CONFIG.PATH, ...)
tester.render_game(model='best', num_episode=30, heatmap=True)
```

공개된 최종 모델(A2)을 바로 테스트해보고 싶다면, `config.py`의 `FOLDER_NAME`을 `"final_model"`로 지정하면 됩니다 — `final_model/models/best.pkl`을 그대로 불러옵니다.

## Results
지뢰 9개, `best` 모델 기준 각 10,000판 실측 결과:

| 모델 | 학습 episode | FIRST_MINE | first_mine=True 승률 | first_mine=True 평균 RPC |
|---|---|---|---|---|
| A1 | 190,000 | True | 67.7% | 0.638 |
| **A2** | **300,000** | **True** | **72.1%** | **0.682** |
| B | 200,000 | False | 58.4% | 0.567 |

R로 수행한 통계 검정 결과, A2는 A1보다 승률·reward·RPC 모두에서 통계적으로 유의하게 우수했고(유의수준 5%), 분산도 더 작아 더 안정적인 성능을 보였습니다. 최종적으로 **A2를 최종 모델로 선정**했습니다.

## Trouble Shooting
- **Reward Hacking**: 초기엔 지뢰만 피하면 되는 구조라 이미 연 칸을 반복 선택하는 "중복 행동"에 빠지는 문제가 심했다. 중복 행동에 지뢰와 같은 -1 보상을 주자 충분히 학습된 에이전트는 이 행동을 거의 하지 않게 됐다.
- **Max Pooling 제거**: 신경망에 Max pooling을 넣자 학습이 전혀 안 됐다. 보드 크기(9×9)가 작고, pooling으로 뭉개지는 각 칸의 정보 자체가 중요한 태스크라 적합하지 않다고 판단해 제거했다.
- **Conv unit 수 비교**: `CONV_UNITS`를 64에서 128로 늘리자 연산 속도만 느려지고 성능은 오히려 낮아져, 64를 최종 채택했다.

## Limitations & Future Work
- `config.txt`가 실제 실행된 설정을 항상 정확히 반영하지는 않는 경우가 있어(예: `dqn_modelB_200k`), 실험 종료 시점에 실제 도달한 episode 수를 별도로 기록하는 방식이 필요합니다.
- `utils.py`의 `original`/`normalization` state encoding 경로는 실제 실험에서는 사용되지 않았습니다 — 향후 인코딩 방식 비교 실험에 활용할 수 있습니다.
- 지뢰 개수·`first_mine` 조합별 반복 테스트는 이뤄졌지만, 9×9보다 큰 보드 크기에 대한 일반화 검증은 아직 진행하지 않았습니다.

## References
- 데이터: 프로젝트 자체 시뮬레이션(자체 구현한 지뢰찾기 환경)에서 생성한 학습/테스트 로그
- 이웅원, 양혁렬, 김건우, 이영우, 이의령, 『파이썬과 케라스로 배우는 강화학습』
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks*
- Mnih, V., et al. (2013). *Playing Atari with Deep Reinforcement Learning*
- Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*
