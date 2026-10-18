
from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR, RESULTS_DIR, SCENARIO_NAMES, TRAIN

COLORS = {"fixed": "#eb6834", "greedy": "#6250d6", "maxpressure": "#1baf7a",
          "q_learning": "#2a78d6", "q_local": "#2a78d6", "q_neighbour": "#e87ba4", "dqn": "#008300"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "font.size": 10, "figure.dpi": 110, "savefig.dpi": 150})


def _read(name: str) -> List[dict]:
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return list(csv.DictReader(f))


def _save(fig, name: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, name))
    plt.close(fig)


def _moving_avg(x: np.ndarray, w: int = 50) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training() -> None:
    """Training reward per episode with moving average (Stage 1), and the epsilon schedule."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    eps = None
    for ax, sc in zip(axes, SCENARIO_NAMES):
        path = os.path.join(RESULTS_DIR, "logs", f"s1_{sc}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        r = d["reward"]
        eps = d["epsilon"]
        ax.plot(r, color=COLORS["q_learning"], alpha=0.25, lw=0.7, label="episode reward")
        ax.plot(np.arange(49, len(r)), _moving_avg(r), color=COLORS["q_learning"], lw=2, label="moving avg (50)")
        ax.set_title(sc)
        ax.set_xlabel("episode")
    axes[0].set_ylabel("episode reward")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, "training_reward.png")
    if eps is not None:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot(eps, color=COLORS["q_learning"], lw=2)
        ax.set_xlabel("episode")
        ax.set_ylabel("epsilon")
        ax.set_title("Epsilon decay (single series)")
        _save(fig, "epsilon_decay.png")


def plot_comparison(stage: int) -> None:
    """Grouped bars: average wait and average queue per scenario for all controllers."""
    rows = [r for r in _read("metrics.csv") if int(r["stage"]) == stage]
    ctrls = list(dict.fromkeys(r["controller"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    width = 0.8 / len(ctrls)
    for ax, (key, label) in zip(axes, (("avg_wait", "avg wait per car (steps)"), ("avg_queue", "avg queue per lane (cars)"))):
        for k, c in enumerate(ctrls):
            vals = [float(next(r for r in rows if r["scenario"] == s and r["controller"] == c)[f"{key}_mean"]) for s in SCENARIO_NAMES]
            errs = [float(next(r for r in rows if r["scenario"] == s and r["controller"] == c)[f"{key}_std"]) for s in SCENARIO_NAMES]
            ax.bar(np.arange(len(SCENARIO_NAMES)) + k * width - 0.4 + width / 2, vals, width * 0.9, yerr=errs,
                   color=COLORS.get(c, MUTED), label=c, error_kw={"lw": 0.8, "ecolor": MUTED})
        ax.set_xticks(range(len(SCENARIO_NAMES)))
        ax.set_xticklabels(SCENARIO_NAMES)
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("lower is better (error bars: std over eval episodes)", fontsize=9, color=MUTED)
    _save(fig, f"comparison_stage{stage}.png")


def plot_queue_trace() -> None:
    """Queue length over time in one evaluation episode: RL vs fixed vs greedy (Stage 1)."""
    path = os.path.join(RESULTS_DIR, "queue_trace.npz")
    d = np.load(path)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=False)
    for ax, sc in zip(axes, SCENARIO_NAMES):
        for c in ("fixed", "greedy", "q_learning"):
            key = f"s1_{sc}_{c}"
            if key in d:
                ax.plot(_moving_avg(d[key], 5), color=COLORS[c], lw=1.4, label=c)
        ax.set_title(sc)
        ax.set_xlabel("step")
    axes[0].set_ylabel("avg queue per lane (5-step avg)")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, "queue_over_time.png")


def plot_switch_hist() -> None:
    """Histogram of phase switches per evaluation episode (Stage 1)."""
    rows = [r for r in _read("episodes.csv") if int(r["stage"]) == 1]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, sc in zip(axes, SCENARIO_NAMES):
        for c in ("fixed", "greedy", "q_learning"):
            v = [float(r["switches"]) for r in rows if r["scenario"] == sc and r["controller"] == c]
            if v:
                ax.hist(v, bins=np.arange(0, 131, 5), color=COLORS[c], alpha=0.65, label=c)
        ax.set_title(sc)
        ax.set_xlabel("switches per episode")
    axes[0].set_ylabel("episodes")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, "switch_histogram.png")


def plot_state_space() -> None:
    """Stage 2: Q-table size vs states actually visited (log scale), and wait time per variant."""
    path = os.path.join(RESULTS_DIR, "state_space.csv")
    if not os.path.exists(path):
        return
    ss = _read("state_space.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    variants = ["tabular_local", "tabular_neighbour"]
    table = [np.mean([float(r["n_states"]) for r in ss if r["variant"] == v]) for v in variants]
    visited = [np.mean([float(r["n_visited"]) for r in ss if r["variant"] == v]) for v in variants]
    x = np.arange(2)
    a1.bar(x - 0.2, table, 0.38, color="#9d9c95", label="table size (states / agent)")
    a1.bar(x + 0.2, visited, 0.38, color=[COLORS["q_local"], COLORS["q_neighbour"]], label="visited in training (mean over scenarios)")
    a1.set_yscale("log")
    a1.set_xticks(x)
    a1.set_xticklabels(["local", "local + neighbours"])
    for xi, t, v in zip(x, table, visited):
        a1.text(xi - 0.2, t * 1.15, f"{t:,.0f}", ha="center", fontsize=8)
        a1.text(xi + 0.2, v * 1.15, f"{v:,.0f}\n({100 * v / t:.1f}%)", ha="center", fontsize=8)
    a1.set_ylim(top=table[1] * 12)
    a1.set_ylabel("states (log scale)")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    rows = [r for r in _read("metrics.csv") if int(r["stage"]) == 2]
    ctrls = [c for c in ("q_local", "q_neighbour", "dqn") if any(r["controller"] == c for r in rows)]
    width = 0.8 / (len(ctrls) + 1)
    series = ctrls + ["greedy"]
    for k, c in enumerate(series):
        vals = [float(next(r for r in rows if r["scenario"] == s and r["controller"] == c)["avg_wait_mean"]) for s in SCENARIO_NAMES]
        a2.bar(np.arange(len(SCENARIO_NAMES)) + k * width - 0.4 + width / 2, vals, width * 0.9, color=COLORS[c], label=c)
    a2.set_xticks(range(len(SCENARIO_NAMES)))
    a2.set_xticklabels(SCENARIO_NAMES)
    a2.set_ylabel("avg wait per car (steps, lower is better)")
    a2.legend(frameon=False, fontsize=8)
    _save(fig, "stage2_state_space_and_performance.png")


def make_all() -> None:
    plot_training()
    plot_comparison(1)
    plot_comparison(2)
    plot_queue_trace()
    plot_switch_hist()
    plot_state_space()
    print(f"plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    make_all()
