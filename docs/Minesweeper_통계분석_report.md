# Minesweeper — 통계분석 보고서

> `Minesweeper_report.md`(프로젝트 전반 보고서)의 부속 문서. A1/A2/B 세 모델의 성능 차이가 실제로 통계적으로 유의한지를 검증한 분석 과정과 결과를 다룹니다.

## 1. 서론

`Minesweeper_report.md`의 실측 결과에서 A2(30만 episode)가 A1(19만 episode)·B(20만 episode, `FIRST_MINE=False`)보다 승률·RPC가 높게 나왔지만, 이 차이가 우연이 아니라 통계적으로 유의한 차이인지, 그리고 지뢰 개수가 늘어나는 등 난이도가 올라가도 이 우위가 유지되는지를 확인할 필요가 있었습니다. 이를 위해 지뢰 개수·`FIRST_MINE` 조합별로 반복 테스트한 로그를 R로 통계 검정했습니다.

## 2. 방법

**실험 설정** — 기본 학습 환경은 9×9 보드, 지뢰 9개이며, 비교 대상 세 모델의 설정은 다음과 같습니다.

| 모델 | 학습 episode | First-Click Safety |
|---|---|---|
| A1 | 190,000 | 없음 (`FIRST_MINE=True`) |
| A2 | 300,000 | 없음 (`FIRST_MINE=True`) |
| B | 200,000 | 있음 (`FIRST_MINE=False`) |

**성능 테스트** — 지뢰 개수(7~12개) × `FIRST_MINE` 조합으로 총 12개 환경을 만들고, 각 환경에서 1,000판씩 10회(총 10,000판) 반복 테스트했습니다. 통계 검정에는 이 중 First-Click Safety가 없는 상황(`FIRST_MINE=False` 테스트 환경), 이동 횟수(cnt) 2 이상인 표본만 사용했습니다.

**평가지표** — 승률(Clear Rate), 총 보상(Total Reward), RPC(Reward per Count = 총 보상 / 이동 횟수).

**검정 방법** — 승률은 Pearson's Chi-squared test로, 총 보상·RPC는 원래 Welch's t-test를 계획했으나 정규성 검정(QQ-plot) 결과 모든 경우에서 정규성을 만족하지 않아 비모수 검정인 Wilcoxon rank sum test를 사용했습니다. 등분산성은 F-test로 확인했습니다.

## 3. 결과

### 3-1. 검정 가정 확인

표본 크기는 각 조건당 약 9,000개 내외로 QQ-plot을 통해 Reward·RPC의 정규성을 확인했으나, **모든 경우에서 정규성을 만족하지 않았습니다.**

| Reward QQ-plot | RPC QQ-plot |
|---|---|
| ![QQ-plot: Reward](images/QQplot_reward.png) | ![QQ-plot: RPC](images/QQplot_rpc.png) |

F-test로 확인한 등분산성 역시 **모든 경우에서 만족하지 않았습니다.**

![등분산성 검정 p-value](images/Var_pvalue.png)

### 3-2. A2 vs A1 — 학습 episode 수에 따른 성능 차이

같은 하이퍼파라미터로 학습 episode 수만 다른(19만 vs 30만) 두 모델을 비교한 검정입니다.

![A2 vs A1 p-value](images/A_pvalue.png)

지뢰 개수 7개인 경우의 총 보상, 지뢰 개수 11개인 경우의 RPC를 제외한 모든 조합에서 통계적으로 유의했고, 등분산성 검정과 승률 검정은 모든 지뢰 개수에서 유의했습니다.

| Clear rate | RPC | Total Reward |
|---|---|---|
| ![PairA table - Clear](images/T_A_clear.png) | ![PairA table - RPC](images/T_A_RPC.png) | ![PairA table - Reward](images/T_A_Reward.png) |

지뢰 개수에 따른 세 지표의 신뢰구간·평균, 그리고 분산 비율(F-statistics):

| 신뢰구간 | 평균 |
|---|---|
| ![A_CI](images/A_CI.png) | ![A_avg](images/A_avg.png) |

![A_Var_F](images/A_Var_F.png)

- 승률·RPC는 지뢰 개수와 무관하게 A2가 거의 일정하게 더 우수했습니다.
- 지뢰 개수가 늘어날수록 총 보상의 차이가 커져, A2가 더 복잡한 상황에 잘 적응하는 것으로 나타났습니다.
- 모든 지뢰 개수에서 A2의 분산이 A1보다 작아(F < 1) 더 안정적인 성능을 보였고, 지뢰 개수가 늘어날수록 두 모델의 변동성 차이는 줄어들었습니다.

**정리**: 같은 하이퍼파라미터에서 더 많은 episode를 학습한 A2가 A1보다 성능·강건성 모두에서 우수했습니다. 모든 지뢰 개수 환경에서 A2의 승률이 평균 5%p 높았습니다.

### 3-3. A1 vs B — First-Click Safety 여부에 따른 성능 차이

동일하게 20만 episode 안팎을 학습했지만 `FIRST_MINE` 설정이 다른 두 모델을 비교한 검정입니다.

![A1 vs B p-value](images/B_pvalue.png)

모든 검정이 통계적으로 유의했고, 등분산성 검정도 유의했습니다 — A1이 B보다 모든 면에서 통계적으로 유의미하게 우수했습니다.

| Clear rate | RPC | Total Reward |
|---|---|---|
| ![PairB table - Clear](images/T_B_Clear.png) | ![PairB table - RPC](images/T_B_RPC.png) | ![PairB table - Reward](images/T_B_Reward.png) |

| 신뢰구간 | 평균 |
|---|---|
| ![B_CI](images/B_CI.png) | ![B_avg](images/B_avg.png) |

![B_Var_F](images/B_Var_F.png)

- 지뢰 개수가 늘어날수록 승률·총 보상·RPC 세 지표의 차이가 모두 커져, A1이 더 복잡한 상황에 잘 적응하는 것으로 나타났습니다.
- 지뢰 개수가 9개 이상일 때 두 모델의 변동성(F-statistics) 차이가 커졌습니다.

**정리**: 학습 episode 수가 비슷한 두 모델 중, First-Click Safety 없이(첫 클릭도 지뢰일 수 있는 환경에서) 학습한 A1이 그렇지 않은 B보다 성능과 강건성 모두 우수했습니다. 모든 지뢰 개수 환경에서 A1의 승률이 평균 10%p 높았습니다.

### 3-4. 2번째 턴 패배율 분석

첫 클릭이 안전하게 보장된 이후, 두 번째 클릭에서 지뢰를 밟는 비율로 모델의 판단력을 검증했습니다. 지뢰찾기의 확률적 구조를 잘 학습한 모델일수록 패배율이 이론적 확률보다 더 낮을 것이라는 가설로, 2번째 턴에 게임 오버된 경우에 대해 Pearson's Chi-squared test를 실시했습니다.

**모델 간 비교(A2 vs A1 / A1 vs B)**

![2nd turn p-value](images/EF_pvalue.png)

| A2 vs A1 | A1 vs B |
|---|---|
| ![EF table - A](images/T_EF_A.png) | ![EF table - B](images/T_EF_B.png) |

![EF 신뢰구간](images/EF_CI.png)

A2는 모든 지뢰 개수에서 A1보다 통계적으로 유의하게 패배율이 낮았습니다(약 5%p). 반면 A1과 B는 지뢰 개수 10개 이하에서는 패배율에 유의한 차이가 없었고, 11개 이상에서는 유의하긴 했지만 A2의 경우보다 신뢰도가 낮았습니다 — 95% 신뢰구간이 대부분 0을 포함해, 두 모델의 패배율에는 큰 차이가 없다고 볼 수 있습니다.

**이론적 패배율과의 비교**

![이론적 패배율 비교 p-value](images/EFT_pvalue.png)

| A2 | A1 | B |
|---|---|---|
| ![EFT table - A2](images/T_EFT_A2.png) | ![EFT table - A1](images/T_EFT_A1.png) | ![EFT table - B](images/T_EFT_B.png) |

![EFT 신뢰구간](images/EFT_CI.png)

세 모델 모두 이론적 패배율보다 통계적으로 유의하게 낮은 패배율을 보였고, 그중 A2의 패배율이 나머지 두 모델보다 크게 낮았습니다. B는 지뢰 개수 7·8개에서 다른 모델들에 비해 신뢰도가 낮았습니다.

**정리**: 2번째 턴 조기 패배율에서도 A2가 A1보다 유의하게 낮아(약 5%p) 지뢰 위치에 대한 확률적 판단 능력이 세 모델 중 가장 뛰어났고, A1과 B의 차이는 통계적으로 뚜렷하지 않았습니다. 세 모델 모두 이론적 패배율보다는 낮은 패배율을 보여, 학습을 통해 어느 정도 확률적 추론 능력을 갖췄음을 확인했습니다.

## 4. 의의

세 가지 검정을 종합하면, **학습 episode 수를 늘리는 것(A1→A2)이 승률·강건성·판단력을 모두 유의하게 향상시켰고**, 흥미롭게도 **First-Click Safety 없이(첫 클릭이 지뢰일 위험을 감수하며) 학습한 모델(A1)이 그 위험이 제거된 환경에서 학습한 모델(B)보다 오히려 더 견고한 성능**을 보였습니다. 이는 첫 클릭에서도 지뢰를 만날 수 있는 환경이 에이전트에게 더 풍부한 학습 신호를 제공했을 가능성을 시사합니다.

이 결과를 근거로 `Minesweeper_report.md`에서 **A2를 최종 모델로 선정**했습니다.
