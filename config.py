"""Central configuration: every hyperparameter and scenario lives here."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

LANES = ("N", "S", "E", "W")  # lane index 0..3; lane N carries cars travelling south, etc.

# Arrival rate (cars/step) per lane (N, S, E, W) as a piecewise-constant schedule:
# list of (start_fraction_of_episode, (rate_N, rate_S, rate_E, rate_W)).
# In a grid, external arrivals only enter on boundary lanes; inner lanes are fed by upstream intersections.
SCENARIOS: Dict[str, List[Tuple[float, Tuple[float, float, float, float]]]] = {
    "low": [(0.0, (0.15, 0.15, 0.15, 0.15))],
    "medium": [(0.0, (0.45, 0.45, 0.35, 0.35))],
    "high": [(0.0, (0.70, 0.70, 0.60, 0.60))],
    # Rush hour: heavy N/S inbound, then balanced, then heavy E/W; asymmetric between lanes.
    "rush_hour": [
        (0.00, (0.90, 0.80, 0.20, 0.15)),
        (0.35, (0.40, 0.40, 0.40, 0.40)),
        (0.65, (0.15, 0.20, 0.80, 0.90)),
    ],
}


@dataclass(frozen=True)
class EnvConfig:
    """Traffic simulation parameters."""

    episode_len: int = 500
    capacity: int = 20              # max cars per lane
    service_rate: int = 2           # cars passing per green lane per step
    yellow_steps: int = 2           # all-red/yellow steps after a switch (no cars pass)
    min_green: int = 3              # minimum green steps before another switch is allowed
    lane_bin_edges: Tuple[int, ...] = (1, 4, 8)         # bins: 0 | 1-3 | 4-7 | 8+
    neighbour_bin_edges: Tuple[int, ...] = (1, 7, 15)   # bins on neighbour's TOTAL queue: 0 | 1-6 | 7-14 | 15+
    wait_weight: float = 1.0        # reward: -wait_weight * (cars waiting)
    switch_penalty: float = 1.0     # reward: -switch_penalty * (1 if switched)
    pass_reward: float = 0.2        # reward: +pass_reward * (cars passed)

    @property
    def time_cap(self) -> int:
        """Cap of the 'time since last switch' state feature (= yellow + min green)."""
        return self.yellow_steps + self.min_green


@dataclass(frozen=True)
class QConfig:
    """Tabular Q-learning hyperparameters."""

    alpha: float = 0.2
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.998    # multiplicative, applied once per episode


@dataclass(frozen=True)
class ControllerConfig:
    """Baseline controller parameters."""

    fixed_period: int = 10                      # default period if not tuned
    fixed_candidates: Tuple[int, ...] = (6, 8, 10, 14, 20, 30, 40)  # tuned per scenario on tuning seeds
    lqf_max_wait: Optional[int] = 50            # starvation safeguard for greedy (None disables)
    lqf_margin: int = 0                         # switch only if other phase has > own + margin cars


@dataclass(frozen=True)
class TrainConfig:
    """Training schedule."""

    episodes: int = 2000
    seed: int = 0
    train_seed_base: int = 10_000   # episode e uses arrival seed train_seed_base + e
    n_workers: int = 2
    grid_size: int = 2              # NxN grid for Stage 2/3


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation protocol (identical seeds for every controller)."""

    n_episodes: int = 20
    eval_seed_base: int = 1_000
    tune_seed_base: int = 500       # seeds used to tune the fixed-timer period (never used for evaluation)
    tune_episodes: int = 4
    trace_seed: int = 1_000         # episode used for the queue-over-time plot


@dataclass(frozen=True)
class DQNConfig:
    """DQN hyperparameters (Stage 3)."""

    hidden: Tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    gamma: float = 0.9
    batch_size: int = 64
    buffer_size: int = 50_000
    warmup: int = 1_000
    target_update: int = 500        # env steps between hard target-network syncs
    train_every: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.993    # per episode (reaches the floor after ~430 episodes)
    episodes: int = 600
    reward_scale: float = 0.1       # rewards are multiplied by this before learning (keeps Q-values small)


ENV = EnvConfig()
Q = QConfig()
CTRL = ControllerConfig()
TRAIN = TrainConfig()
EVAL = EvalConfig()
DQN = DQNConfig()

SCENARIO_NAMES = tuple(SCENARIOS.keys())
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
