
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import numpy as np

from config import CTRL, EVAL, RESULTS_DIR, SCENARIO_NAMES, TRAIN
from controllers import Controller, FixedTimerController, LongestQueueFirstController, MaxPressureController
from qlearning_agent import QLearningAgent, TabularPolicy
from train import make_tag, model_path
from traffic_env import TrafficEnv

METRICS = ["avg_wait", "avg_queue", "throughput", "switches", "max_wait", "dropped"]


def run_episode(env: TrafficEnv, ctrl: Controller, seed: int, trace: bool = False) -> Tuple[Dict[str, float], List[float]]:
    """Run one full episode; returns (metrics, per-step average queue per lane if trace else [])."""
    env.reset(seed)
    ctrl.reset(env)
    queue_trace: List[float] = []
    done = False
    while not done:
        _, done = env.step(ctrl.act(env))
        if trace:
            queue_trace.append(env.total_queue() / (4 * env.n_int))
    return env.metrics(), queue_trace


def tune_fixed_period(scenario: str, grid: int) -> int:
    """Pick the fixed-timer period with the lowest wait on tuning seeds (disjoint from evaluation seeds)."""
    env = TrafficEnv(scenario, grid)
    best, best_wait = CTRL.fixed_period, float("inf")
    for period in CTRL.fixed_candidates:
        w = np.mean([run_episode(env, FixedTimerController(period), EVAL.tune_seed_base + k)[0]["avg_wait"]
                     for k in range(EVAL.tune_episodes)])
        if w < best_wait:
            best, best_wait = period, w
    return best


def load_tabular(stage: int, mode: str, scenario: str, n_agents: int) -> Controller:
    tag = make_tag(stage, mode, scenario)
    agents = [QLearningAgent.load(model_path(tag, i)) for i in range(n_agents)]
    name = "q_learning" if stage == 1 else f"q_{mode}"
    return TabularPolicy(agents, mode, name)


def build_controllers(stage: int, scenario: str, grid: int) -> Dict[str, Controller]:
    period = tune_fixed_period(scenario, grid)
    ctrls: Dict[str, Controller] = {f"fixed(N={period})": FixedTimerController(period),
                                    "greedy": LongestQueueFirstController()}
    if grid > 1:
        ctrls["maxpressure"] = MaxPressureController()
    n_agents = grid * grid
    modes = ["local"] if stage == 1 else ["local", "neighbour"]
    for m in modes:
        if os.path.exists(model_path(make_tag(stage, m, scenario), 0)):
            c = load_tabular(stage, m, scenario, n_agents)
            ctrls[c.name] = c
    if stage == 2 and os.path.exists("dqn_agent.py"):
        import dqn_agent
        c = dqn_agent.load_policy(scenario, n_agents)
        if c is not None:
            ctrls[c.name] = c
    return ctrls


def evaluate_stage(stage: int, rows: List[dict], eps_rows: List[dict], traces: Dict[str, np.ndarray]) -> None:
    grid = 1 if stage == 1 else TRAIN.grid_size
    for scenario in SCENARIO_NAMES:
        env = TrafficEnv(scenario, grid)
        for name, ctrl in build_controllers(stage, scenario, grid).items():
            kind = name.split("(")[0]
            res = []
            for k in range(EVAL.n_episodes):
                seed = EVAL.eval_seed_base + k
                m, tr = run_episode(env, ctrl, seed, trace=(seed == EVAL.trace_seed))
                res.append(m)
                eps_rows.append({"stage": stage, "scenario": scenario, "controller": kind, "seed": seed, **m})
                if tr:
                    traces[f"s{stage}_{scenario}_{kind}"] = np.array(tr)
            row = {"stage": stage, "scenario": scenario, "controller": kind, "label": name, "episodes": len(res)}
            for key in METRICS:
                vals = [r[key] for r in res]
                row[f"{key}_mean"], row[f"{key}_std"] = float(np.mean(vals)), float(np.std(vals))
            rows.append(row)
            print(f"  stage {stage} {scenario:<10} {name:<16} wait={row['avg_wait_mean']:6.2f}  "
                  f"queue={row['avg_queue_mean']:5.2f}  thr={row['throughput_mean']:6.1f}", flush=True)


def unseen_state_rates() -> List[dict]:
    """Share of evaluation decisions taken in states never visited during training (tabular agents only)."""
    out = []
    for stage, mode in ((1, "local"), (2, "local"), (2, "neighbour")):
        grid = 1 if stage == 1 else TRAIN.grid_size
        for scenario in SCENARIO_NAMES:
            pol = load_tabular(stage, mode, scenario, grid * grid)
            env = TrafficEnv(scenario, grid)
            unseen = total = 0
            for k in range(EVAL.n_episodes):
                env.reset(EVAL.eval_seed_base + k)
                done = False
                while not done:
                    for i in range(env.n_int):
                        unseen += int(pol.agents[i].visits[env.encode(i, mode)] == 0)
                        total += 1
                    _, done = env.step(pol.act(env))
            out.append({"stage": stage, "variant": pol.name, "scenario": scenario,
                        "pct_decisions_in_unvisited_states": round(100 * unseen / total, 3)})
    return out


def pct_better(base: float, new: float) -> float:
    """Percent reduction of a lower-is-better metric relative to a baseline."""
    return 100.0 * (base - new) / base if base else 0.0


def print_summary(rows: List[dict]) -> None:
    for stage in sorted({r["stage"] for r in rows}):
        title = "Stage 1: single intersection" if stage == 1 else f"Stage 2: {TRAIN.grid_size}x{TRAIN.grid_size} grid"
        print(f"\n=== {title} (negative % = RL worse; wait/queue improvement vs baseline) ===")
        print(f"{'scenario':<10} {'agent':<12} {'wait':>6} {'queue':>6} | {'vs fixed (wait/queue)':>22} | {'vs greedy (wait/queue)':>23}")
        for scenario in SCENARIO_NAMES:
            sub = {r["controller"]: r for r in rows if r["stage"] == stage and r["scenario"] == scenario}
            fixed, greedy = sub["fixed"], sub["greedy"]
            for name, r in sub.items():
                if not name.startswith("q_") and name != "dqn":
                    continue
                print(f"{scenario:<10} {name:<12} {r['avg_wait_mean']:6.2f} {r['avg_queue_mean']:6.2f} | "
                      f"{pct_better(fixed['avg_wait_mean'], r['avg_wait_mean']):>10.1f}% {pct_better(fixed['avg_queue_mean'], r['avg_queue_mean']):>9.1f}% | "
                      f"{pct_better(greedy['avg_wait_mean'], r['avg_wait_mean']):>10.1f}% {pct_better(greedy['avg_queue_mean'], r['avg_queue_mean']):>10.1f}%")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows: List[dict] = []
    eps_rows: List[dict] = []
    traces: Dict[str, np.ndarray] = {}
    for stage in (1, 2):
        print(f"Evaluating stage {stage} ...")
        evaluate_stage(stage, rows, eps_rows, traces)
    for path, data in (("metrics.csv", rows), ("episodes.csv", eps_rows)):
        with open(os.path.join(RESULTS_DIR, path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    np.savez(os.path.join(RESULTS_DIR, "queue_trace.npz"), **traces)
    unseen = unseen_state_rates()
    with open(os.path.join(RESULTS_DIR, "unseen_states.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(unseen[0].keys()))
        w.writeheader()
        w.writerows(unseen)
    print_summary(rows)
    import plotter
    plotter.make_all()


if __name__ == "__main__":
    main()
