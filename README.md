# Traffic Signal Control with Q-Learning: from a single intersection to a multi-intersection grid

An RL agent controls traffic lights in a small, seedable simulator written in pure Python (numpy, matplotlib).
It is compared with a **fixed-timer** controller and a **greedy longest-queue-first** controller, first on a single
intersection (tabular Q-learning), then on a 2x2 grid (two tabular variants), and finally with a **DQN** to show where
the table stops scaling. All numbers below come from the runs in this repository (`results/metrics.csv`).

## Problem statement

Each intersection has four incoming lanes (N, S, E, W) and two phases (North-South green, East-West green). Cars arrive
randomly (Poisson), two cars per green lane can pass per step, and every phase change costs a 2-step yellow/all-red
period in which nobody moves. The goal is to minimise waiting time and queue length without starving any direction.

**Why RL can beat fixed and greedy controllers**

- **Fixed timer:** ignores traffic. It wastes green on empty lanes and cannot follow an asymmetric or changing load (rush hour).
- **Greedy (longest queue first):** myopic. It switches whenever the other side is longer, ignoring that each switch costs 2 dead steps, so under heavy load it thrashes (high load: 95 switches per episode, 14.95 steps average wait). It can also starve a short queue, which is why it needs a max-wait safeguard.
- **Neither coordinates** with neighbouring intersections (green waves, spillback).
- **RL** optimises the long-run discounted reward, so it can learn *to hold a green longer* because switching has a future cost.

## Design

**State (per intersection).** Queue length of each lane binned `0 | 1-3 | 4-7 | 8+` (4 levels), current phase (2), time since last
switch capped at 5 (6 values). Time-since-switch counts from the switch decision, so values 1-2 mean yellow, 3-4 mean green with
switching locked, 5 means a switch is allowed (yellow 2 + min green 3 = cap 5).

`4 lanes -> 4^4 = 256;  x 2 phases = 512;  x 6 timer values = 3,072 states, x 2 actions`

With neighbour information (grid variant b), each existing neighbour adds its **total** queue binned as `0 | 1-6 | 7-14 | 15+`:
`3,072 x 4^k` states for `k` neighbours. In the 2x2 grid every intersection has k=2, giving **49,152** states; an interior intersection of a
3x3 grid (k=4) would need 786,432.

**Actions.** keep / switch. A switch requested before the minimum green has elapsed is ignored.

**Reward.** `-(cars waiting) - switch_penalty * [switched] + 0.2 * (cars passed)`, all weights in `config.py`.
The waiting term is the objective; the pass term gives a small dense signal; the switch penalty (1.0) discourages flapping on top of the
2 dead steps that every switch already costs.

**Scenarios.** `low` (0.15 cars/step/lane), `medium` (N/S 0.45, E/W 0.35), `high` (N/S 0.70, E/W 0.60, near saturation),
`rush_hour` (N/S heavy, then balanced, then E/W heavy, with asymmetric lanes). In the grid, external cars enter on boundary lanes only;
inner lanes are fed by upstream intersections and a full downstream lane blocks the car (spillback).

**Metrics.** Average wait per car (steps, per intersection crossed; cars still queued at the end are included, so starvation is not hidden),
average queue per lane, throughput (cars that left the network), phase switches, max wait of any car, and dropped arrivals (lane full).

## How to run

```bash
pip install -r requirements.txt          # torch is only needed for Stage 3
python train.py && python evaluate.py    # trains everything, evaluates, saves results/ and plots/
pytest -q tests                          # 6 quick tests
python tune.py                           # optional: the hyperparameter search described below
```

Training takes about 36 s (Stage 1), 4 min (Stage 2, 2 workers) and 6 min (Stage 3) on 2 CPU cores. Everything is seeded.
`python train.py --stage 1|2|3` runs a single stage. Evaluation uses seeds 1000-1019 (20 episodes per scenario), identical for every controller,
epsilon = 0 for learned agents. Training seeds (10000+) and tuning seeds (500+) never overlap with evaluation seeds.

## Results

Percentages are reductions relative to the baseline (positive = RL better, lower wait/queue is better). The fixed-timer period is **tuned per
scenario** on separate seeds (N = 6, 6, 10, 14 on the single intersection), so the fixed baseline is not a straw man.

### Stage 1: single intersection, tabular Q-learning (3,072 states)

| Scenario | Fixed | Greedy | Q-learning | Wait vs fixed | Wait vs greedy | Queue vs fixed | Queue vs greedy |
|---|---|---|---|---|---|---|---|
| low | 3.14 | 2.28 | **2.19** | +30.1% | +4.0% | +22.8% | +2.7% |
| medium | 3.80 | 3.61 | **3.14** | +17.5% | +13.0% | +13.9% | +10.2% |
| high | 6.60 | 14.95 | **6.24** | +5.4% | +58.3% | +4.7% | +53.3% |
| rush_hour | 8.23 | 3.67 | **3.18** | +61.4% | +13.3% | +53.8% | +10.5% |

Average wait per car (steps). Starvation and switching, Q-learning vs greedy vs fixed: max wait low 8/8/8, medium 14/13/14, high 29/37/24, rush hour 34/16/24 steps;
switches per episode in high load 62/95/50. **Honest caveats:** the +4% at low load is small, and in rush hour the Q-agent's worst-case wait (34 steps)
is more than twice greedy's (16), because the reward does not penalise individual long waits.

![Training reward](plots/training_reward.png)
![Epsilon decay](plots/epsilon_decay.png)
![Stage 1 comparison](plots/comparison_stage1.png)
![Queue over time](plots/queue_over_time.png)
![Switch histogram](plots/switch_histogram.png)

**Tuning history.** My first configuration (gamma 0.95, alpha 0.1, switch penalty 2.0) did *not* beat the baselines everywhere: at low load it lost to greedy
(2.47 vs 2.28 wait, -8%) and at high load it tied the tuned fixed timer (6.72 vs 6.60, -2%). I tuned on validation seeds (500-507), using 3 training seeds per
configuration, and found two things. (1) gamma 0.9, alpha 0.2 and switch penalty 1.0 were best. (2) Some runs produced a catastrophic episode (average wait 49) when the greedy
policy wandered into never-visited states, where Q-values tie and the agent keeps one phase green forever. Random tie-breaking during training fixed the
exploration side; only the chosen config was stable across all three seeds (worst validation episode 6.7 vs 49.3). Those settings are the defaults in `config.py`.

### Stage 2: 2x2 grid, independent tabular agents

| Scenario | Fixed | Greedy | Max-pressure | Q local | Q +neighbours |
|---|---|---|---|---|---|
| low | **1.90** | 2.44 | 2.96 | 3.56 | 5.11 |
| medium | **2.51** | 3.71 | 3.97 | 3.32 | 3.50 |
| high | **3.93** | 12.06 | 11.59 | 6.17 | 7.35 |
| rush_hour | 4.70 | 3.65 | 6.33 | **3.01** | 3.77 |

Average wait per car per intersection (steps), fixed-timer N = 6, 8, 10, 14. The tuned fixed timer wins three of four grid scenarios. All intersections switch on the same clock, which acts as
a crude green wave that independent agents do not learn. Q-local beats greedy in medium, high and rush hour; **Q-learning with neighbour information is
worse than Q-local in every scenario** and loses to greedy in rush hour (3.77 vs 3.65).

![Stage 2 comparison](plots/comparison_stage2.png)

### Where tabular Q-Learning breaks down (measured, Stage 2)

| | States per agent | Visited in training (low / medium / high / rush) | Eval decisions in never-visited states (low / medium / high / rush) |
|---|---|---|---|
| Single intersection | 3,072 | 518 / 1,853 / 2,444 / 2,133 | 0.0 / 0.02 / 0.0 / 0.04 % |
| Grid, local | 3,072 | 572 / 2,044 / 2,522 / 2,350 (agent 0) | 1.67 / 0.04 / 0.005 / 0.0 % |
| Grid, +neighbours | **49,152** | 2,974 / 14,413 / 14,307 / 19,594 (agent 0) | 3.37 / 0.45 / 0.44 / 0.33 % |

- Adding two neighbour features multiplies the table by 16, and only 6% to 40% of it was ever visited in 2,000 episodes, so each visited state gets about 16x fewer updates.
- The neighbour variant got **worse** (wait +5% to +44% vs local, e.g. 5.11 vs 3.56 at low load) even though it has strictly more information. The extra information cannot be used because the table cannot generalise between similar states.
- Unvisited states are where agents fail: in the low-load grid scenario one evaluation episode of Q-local averaged a 29.6-step wait and one of Q-neighbour 56.7 (typical episodes: about 2), because a lane got stuck in unseen states. The per-scenario standard deviation (5.98 and 11.83) shows it in `results/metrics.csv`.
- A 3x3 grid or finer bins would push the table to hundreds of thousands of states per agent.

### Stage 3: DQN on the grid (justified by Stage 2)

Stage 2 showed the neighbour information is wasted by the table, which motivates a function approximator. The DQN (`dqn_agent.py`) is an MLP (22 inputs -> 64 -> 64 -> 2) on **raw queue lengths** of the local intersection and its
4 neighbour slots, plus phase and timer, with experience replay, a target network and epsilon-greedy exploration. One network is shared by the four
intersections (parameter sharing). 600 episodes per scenario, same training seeds as the tabular agents.

| Scenario | Fixed | Greedy | Q local | Q +neighbours | **DQN** | DQN vs best tabular | DQN vs fixed | DQN vs greedy |
|---|---|---|---|---|---|---|---|---|
| low | 1.90 | 2.44 | 3.56 | 5.11 | **1.79** | +49.8% | +5.8% | +26.8% |
| medium | **2.51** | 3.71 | 3.32 | 3.50 | 2.72 | +18.0% | -8.7% | +26.6% |
| high | **3.93** | 12.06 | 6.17 | 7.35 | 5.24 | +15.0% | -33.6% | +56.5% |
| rush_hour | 4.70 | 3.65 | 3.01 | 3.77 | **2.68** | +10.9% | +43.0% | +26.6% |

![Stage 2/3 state space and performance](plots/stage2_state_space_and_performance.png)


## Limitations and future work

- **Simplified simulation, not SUMO:** discrete time, cars go straight, no turning movements, pedestrians, lane geometry or driver behaviour; arrival rates are synthetic and agents are trained and tested within the same scenario.
- Wait time is measured per car per intersection crossed (not door to door), with one fixed-timer period tuned per scenario.
- Tabular results use the hyperparameters tuned for Stage 1 on 3 seeds; the DQN was not tuned and was trained with one seed. Differences of a few percent (e.g. +4% vs greedy at low load) should not be over-read.
- The reward does not penalise long individual waits, so RL max-wait can be worse than greedy's (rush hour).
- **Future work:** explicit multi-agent coordination (shared reward, communication, offsets for green waves), turning movements, a SUMO/TraCI integration with real demand, multi-seed DQN runs and a policy that generalises across scenarios.

#   a d a p t i v e - t r a f f i c - s i g n a l - r l  
 