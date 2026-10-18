"""Train tabular Q-learning agents (Stage 1 single intersection, Stage 2 grid) and, if present, DQN (Stage 3).

Usage:  python train.py [--stage 1|2|3|all] [--episodes N] [--scenarios low medium ...]
Everything is seeded: the same command reproduces the same Q-tables.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from config import Q, RESULTS_DIR, SCENARIO_NAMES, TRAIN
from qlearning_agent import QLearningAgent
from traffic_env import TrafficEnv

MODEL_DIR = os.path.join(RESULTS_DIR, "models")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")


def model_path(tag: str, agent: int) -> str:
    return os.path.join(MODEL_DIR, f"{tag}_agent{agent}.npy")


def make_tag(stage: int, mode: str, scenario: str) -> str:
    return f"s1_{scenario}" if stage == 1 else f"s2_{mode}_{scenario}"


def train_tabular(job: Tuple[int, str, str, int]) -> Dict[str, object]:
    """Train one independent Q-agent per intersection. job = (stage, mode, scenario, episodes)."""
    stage, mode, scenario, episodes = job
    grid = 1 if stage == 1 else TRAIN.grid_size
    tag = make_tag(stage, mode, scenario)
    env = TrafficEnv(scenario, grid)
    agents = [QLearningAgent(env.state_size(i, mode), 2, Q, seed=TRAIN.seed + 100 * i) for i in range(env.n_int)]
    rewards, epsilons, waits = [], [], []
    t0 = time.time()
    for ep in range(episodes):
        env.reset(TRAIN.train_seed_base + ep)
        states = [env.encode(i, mode) for i in range(env.n_int)]
        ep_reward, done = 0.0, False
        while not done:
            actions = [agents[i].act(states[i]) for i in range(env.n_int)]
            r, done = env.step(actions)
            nxt = [env.encode(i, mode) for i in range(env.n_int)]
            for i in range(env.n_int):
                agents[i].update(states[i], actions[i], float(r[i]), nxt[i])
            states = nxt
            ep_reward += r.sum() / env.n_int
        rewards.append(ep_reward)
        epsilons.append(agents[0].epsilon)
        waits.append(env.metrics()["avg_wait"])
        for a in agents:
            a.decay_epsilon()
        if (ep + 1) % 500 == 0:
            print(f"[{tag}] ep {ep + 1}/{episodes}  reward(last50)={np.mean(rewards[-50:]):.1f}  "
                  f"wait(last50)={np.mean(waits[-50:]):.2f}  eps={agents[0].epsilon:.3f}  {time.time() - t0:.0f}s", flush=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    for i, a in enumerate(agents):
        a.save(model_path(tag, i))
    np.savez(os.path.join(LOG_DIR, f"{tag}.npz"), reward=rewards, epsilon=epsilons, wait=waits,
             n_states=[a.n_states for a in agents], n_visited=[a.n_visited for a in agents])
    return {"tag": tag, "stage": stage, "mode": mode, "scenario": scenario,
            "n_states": [a.n_states for a in agents], "n_visited": [a.n_visited for a in agents]}


def write_state_space_csv() -> None:
    """Collect Q-table size and visited-state counts of all Stage 2 runs into results/state_space.csv."""
    rows = []
    for scenario in SCENARIO_NAMES:
        for mode in ("local", "neighbour"):
            path = os.path.join(LOG_DIR, f"{make_tag(2, mode, scenario)}.npz")
            if os.path.exists(path):
                d = np.load(path)
                for i, (n, v) in enumerate(zip(d["n_states"], d["n_visited"])):
                    rows.append((scenario, f"tabular_{mode}", i, int(n), int(v), round(100 * v / n, 2)))
    if rows:
        with open(os.path.join(RESULTS_DIR, "state_space.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenario", "variant", "agent", "n_states", "n_visited", "pct_visited"])
            w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["1", "2", "3", "all"])
    p.add_argument("--episodes", type=int, default=TRAIN.episodes)
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIO_NAMES))
    p.add_argument("--workers", type=int, default=TRAIN.n_workers)
    args = p.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs: List[Tuple[int, str, str, int]] = []
    if args.stage in ("1", "all"):
        jobs += [(1, "local", s, args.episodes) for s in args.scenarios]
    if args.stage in ("2", "all"):
        jobs += [(2, m, s, args.episodes) for s in args.scenarios for m in ("local", "neighbour")]
    if jobs:
        with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
            for res in pool.imap_unordered(train_tabular, jobs):
                print(f"done {res['tag']}: states/agent={res['n_states'][0]:,} visited={res['n_visited']}", flush=True)
        write_state_space_csv()

    if args.stage in ("3", "all") and os.path.exists("dqn_agent.py"):
        import dqn_agent  # Stage 3 is optional: only runs if the file exists
        dqn_agent.train_all(args.scenarios)


if __name__ == "__main__":
    main()
