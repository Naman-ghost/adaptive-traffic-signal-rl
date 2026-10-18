

from typing import List, Sequence

import numpy as np

from config import Q, QConfig
from controllers import Controller
from traffic_env import TrafficEnv


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning with per-episode multiplicative epsilon decay."""

    def __init__(self, n_states: int, n_actions: int = 2, cfg: QConfig = Q, seed: int = 0) -> None:
        self.cfg = cfg
        self.q = np.zeros((n_states, n_actions))
        self.visits = np.zeros(n_states, dtype=np.int64)
        self.epsilon = cfg.epsilon_start
        self.rng = np.random.default_rng(seed)

    def act(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action.

        Training: ties between equal Q-values (e.g. never-seen states) are broken randomly so unseen states
        get explored. Greedy/evaluation (epsilon = 0): ties resolve to 'keep' (0), deterministic.
        """
        row = self.q[state]
        if greedy:
            return int(np.argmax(row))
        if self.rng.random() < self.epsilon or row[0] == row[1]:
            return int(self.rng.integers(self.q.shape[1]))
        return int(np.argmax(row))

    def update(self, s: int, a: int, r: float, s2: int) -> None:
        """One-step Q-learning update (episodes are time-truncated, so we always bootstrap)."""
        self.visits[s] += 1
        target = r + self.cfg.gamma * self.q[s2].max()
        self.q[s, a] += self.cfg.alpha * (target - self.q[s, a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    @property
    def n_states(self) -> int:
        return self.q.shape[0]

    @property
    def n_visited(self) -> int:
        return int((self.visits > 0).sum())

    def save(self, path: str) -> None:
        """Save the Q-table to `path` (.npy) and visit counts to `<path stem>_visits.npy`."""
        np.save(path, self.q)
        np.save(path.replace(".npy", "_visits.npy"), self.visits)

    @classmethod
    def load(cls, path: str, cfg: QConfig = Q) -> "QLearningAgent":
        table = np.load(path)
        agent = cls(table.shape[0], table.shape[1], cfg)
        agent.q = table
        try:
            agent.visits = np.load(path.replace(".npy", "_visits.npy"))
        except FileNotFoundError:
            agent.visits = (np.abs(table).sum(axis=1) > 0).astype(np.int64)
        agent.epsilon = 0.0
        return agent


class TabularPolicy(Controller):
    """Greedy (epsilon = 0) policy using one Q-table per intersection."""

    def __init__(self, agents: Sequence[QLearningAgent], mode: str, name: str) -> None:
        self.agents, self.mode, self.name = list(agents), mode, name

    def select(self, env: TrafficEnv, i: int) -> int:
        return self.agents[i].act(env.encode(i, self.mode), greedy=True)
