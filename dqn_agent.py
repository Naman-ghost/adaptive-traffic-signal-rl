
from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from config import DQN, DQNConfig, RESULTS_DIR, SCENARIO_NAMES, TRAIN
from controllers import Controller
from traffic_env import TrafficEnv

MODEL_DIR = os.path.join(RESULTS_DIR, "models")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")


class QNetwork(nn.Module):
    """MLP: obs -> Q(keep), Q(switch)."""

    def __init__(self, obs_dim: int, hidden=DQN.hidden, n_actions: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-size ring buffer of (s, a, r, s2) transitions (episodes are time-truncated, so no done flag)."""

    def __init__(self, size: int, obs_dim: int, seed: int) -> None:
        self.s = np.zeros((size, obs_dim), np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros(size, np.int64)
        self.r = np.zeros(size, np.float32)
        self.size, self.n, self.i = size, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2) -> None:
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i] = s, a, r, s2
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.n, batch)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]))


class DQNAgent:
    """DQN with replay, target network and epsilon-greedy exploration."""

    def __init__(self, obs_dim: int, cfg: DQNConfig = DQN, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.cfg = cfg
        self.q = QNetwork(obs_dim, cfg.hidden)
        self.target = QNetwork(obs_dim, cfg.hidden)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim, seed)
        self.epsilon = cfg.epsilon_start
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    @torch.no_grad()
    def greedy_actions(self, obs: np.ndarray) -> np.ndarray:
        return self.q(torch.from_numpy(obs)).argmax(dim=1).numpy()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Epsilon-greedy actions for a batch of observations (one row per intersection)."""
        actions = self.greedy_actions(obs)
        explore = self.rng.random(len(actions)) < self.epsilon
        return np.where(explore, self.rng.integers(0, 2, len(actions)), actions)

    def learn(self) -> Optional[float]:
        if self.buffer.n < max(self.cfg.warmup, self.cfg.batch_size):
            return None
        s, a, r, s2 = self.buffer.sample(self.cfg.batch_size)
        with torch.no_grad():
            target = r + self.cfg.gamma * self.target(s2).max(dim=1).values
        pred = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.q.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)


class DQNPolicy(Controller):
    """Greedy (epsilon = 0) evaluation policy; batches all intersections in one forward pass."""

    name = "dqn"

    def __init__(self, net: QNetwork) -> None:
        self.net = net.eval()

    @torch.no_grad()
    def act(self, env: TrafficEnv) -> List[int]:
        obs = np.stack([env.raw_obs(i) for i in range(env.n_int)])
        return self.net(torch.from_numpy(obs)).argmax(dim=1).tolist()


def train_scenario(args) -> Dict[str, object]:
    """Train one shared DQN on the grid for a scenario. args = (scenario, episodes)."""
    scenario, episodes = args
    torch.set_num_threads(1)
    env = TrafficEnv(scenario, TRAIN.grid_size)
    agent = DQNAgent(env.raw_obs_dim, DQN, seed=TRAIN.seed)
    rewards, epsilons, waits = [], [], []
    step_count, t0 = 0, time.time()
    for ep in range(episodes):
        env.reset(TRAIN.train_seed_base + ep)
        obs = np.stack([env.raw_obs(i) for i in range(env.n_int)])
        ep_reward, done = 0.0, False
        while not done:
            actions = agent.act(obs)
            r, done = env.step(actions.tolist())
            obs2 = np.stack([env.raw_obs(i) for i in range(env.n_int)])
            for i in range(env.n_int):
                agent.buffer.add(obs[i], actions[i], r[i] * DQN.reward_scale, obs2[i])
            obs = obs2
            ep_reward += r.sum() / env.n_int
            step_count += 1
            if step_count % DQN.train_every == 0:
                agent.learn()
            if step_count % DQN.target_update == 0:
                agent.sync_target()
        rewards.append(ep_reward)
        epsilons.append(agent.epsilon)
        waits.append(env.metrics()["avg_wait"])
        agent.decay_epsilon()
        if (ep + 1) % 100 == 0:
            print(f"[dqn_{scenario}] ep {ep + 1}/{episodes} reward(last50)={np.mean(rewards[-50:]):.1f} "
                  f"wait(last50)={np.mean(waits[-50:]):.2f} eps={agent.epsilon:.3f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    torch.save(agent.q.state_dict(), os.path.join(MODEL_DIR, f"dqn_{scenario}.pt"))
    np.savez(os.path.join(LOG_DIR, f"s2_dqn_{scenario}.npz"), reward=rewards, epsilon=epsilons, wait=waits)
    return {"scenario": scenario}


def train_all(scenarios=SCENARIO_NAMES, episodes: int = DQN.episodes) -> None:
    jobs = [(s, episodes) for s in scenarios]
    with mp.get_context("fork").Pool(min(TRAIN.n_workers, len(jobs))) as pool:
        for res in pool.imap_unordered(train_scenario, jobs):
            print(f"done dqn_{res['scenario']}", flush=True)


def load_policy(scenario: str, n_agents: int) -> Optional[DQNPolicy]:
    """Load the trained DQN for a scenario, or None if it has not been trained."""
    path = os.path.join(MODEL_DIR, f"dqn_{scenario}.pt")
    if not os.path.exists(path):
        return None
    net = QNetwork(TrafficEnv(scenario, TRAIN.grid_size).raw_obs_dim)
    net.load_state_dict(torch.load(path))
    return DQNPolicy(net)


if __name__ == "__main__":
    train_all()
