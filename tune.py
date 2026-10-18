"""Hyperparameter tuning used to pick the final reward/Q settings: 3 training seeds per config, scored on
validation seeds 500..507 (never the evaluation seeds). Usage: python tune.py  (about 6 minutes)."""
import sys, dataclasses, multiprocessing as mp
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
from config import ENV, Q, TRAIN, SCENARIO_NAMES
from traffic_env import TrafficEnv
from qlearning_agent import QLearningAgent, TabularPolicy
from evaluate import run_episode

CONFIGS = {
 "g.9 a.1 sp1 e.05":  (dict(switch_penalty=1.0), dict(gamma=0.9, alpha=0.1)),
 "g.9 a.2 sp1 e.05":  (dict(switch_penalty=1.0), dict(gamma=0.9, alpha=0.2)),
 "g.9 a.1 sp1 e.10":  (dict(switch_penalty=1.0), dict(gamma=0.9, alpha=0.1, epsilon_end=0.10)),
 "g.9 a.2 sp.5 e.10": (dict(switch_penalty=0.5), dict(gamma=0.9, alpha=0.2, epsilon_end=0.10)),
}

def job(args):
    name, scenario, seed = args
    ecfg_kw, qcfg_kw = CONFIGS[name]
    ecfg = dataclasses.replace(ENV, **ecfg_kw); qcfg = dataclasses.replace(Q, **qcfg_kw)
    env = TrafficEnv(scenario, 1, ecfg); ag = QLearningAgent(env.local_state_size, 2, qcfg, seed=seed)
    for ep in range(2000):
        env.reset(TRAIN.train_seed_base + ep); s = env.encode(0, "local"); done = False
        while not done:
            a = ag.act(s); r, done = env.step([a]); s2 = env.encode(0, "local")
            ag.update(s, a, float(r[0]), s2); s = s2
        ag.decay_epsilon()
    pol = TabularPolicy([ag], "local", "q"); venv = TrafficEnv(scenario, 1, ecfg)
    res = [run_episode(venv, pol, 500 + k)[0] for k in range(8)]
    return name, scenario, seed, np.mean([r["avg_wait"] for r in res]), max(r["avg_wait"] for r in res), ag.n_visited

if __name__ == "__main__":
    jobs = [(n, s, sd) for n in CONFIGS for s in SCENARIO_NAMES for sd in (0, 1, 2)]
    with mp.get_context("fork").Pool(2) as p:
        out = p.map(job, jobs)
    print("config               " + "".join(f"{s:>16}" for s in SCENARIO_NAMES) + "     mean   worst-ep")
    for n in CONFIGS:
        cells, means, worst = [], [], 0
        for s in SCENARIO_NAMES:
            o = [x for x in out if x[0] == n and x[1] == s]
            m = np.mean([x[3] for x in o]); means.append(m); worst = max(worst, max(x[4] for x in o))
            cells.append(f"{m:6.2f}({min(x[3] for x in o):.1f}-{max(x[3] for x in o):.1f})")
        print(f"{n:<20} " + "".join(f"{c:>16}" for c in cells) + f"{np.mean(means):9.2f}{worst:10.1f}")
