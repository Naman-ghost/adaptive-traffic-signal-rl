"""Small sanity tests for the simulator. Run: pytest -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import ENV
from controllers import FixedTimerController
from traffic_env import TrafficEnv


def test_queues_never_negative_or_over_capacity():
    env = TrafficEnv("high", 2, seed=3)
    ctrl = FixedTimerController(6)
    ctrl.reset(env)
    done = False
    while not done:
        _, done = env.step(ctrl.act(env))
        assert all(0 <= len(l) <= ENV.capacity for inter in env.q for l in inter)


def test_switch_causes_yellow_period():
    env = TrafficEnv("high", 1, seed=1)
    for _ in range(10):                       # build queues in both phases
        env.step([0])
    env.since[0] = env.cap                    # allow a switch
    before = env.crossings
    env.step([1])                             # switch: step 1 of yellow
    env.step([0])                             # step 2 of yellow
    assert env.crossings == before            # nobody passed during yellow
    env.step([0])
    assert env.crossings > before             # new phase passes cars afterwards


def test_reward_is_computed_correctly():
    env = TrafficEnv("low", 1, seed=0)
    env.q[0][0].extend([0] * 5)               # 5 cars in N lane (phase 0 green)
    env.since[0] = env.cap
    env._arrivals = [[[0, 0, 0, 0]]] * env.T  # no new arrivals
    r, _ = env.step([0])                      # 2 cars pass, 3 remain
    assert np.isclose(r[0], -ENV.wait_weight * 3 + ENV.pass_reward * 2)
    env.since[0] = env.cap
    r, _ = env.step([1])                      # switch: yellow, 0 pass, 3 wait
    assert np.isclose(r[0], -ENV.wait_weight * 3 - ENV.switch_penalty)


def test_cars_move_between_grid_intersections():
    env = TrafficEnv("low", 2, seed=0)
    env._arrivals = [[[0] * 4 for _ in range(4)]] * env.T
    env.q[0][0].extend([0, 0])                # intersection (0,0), lane N heads south to (1,0) = index 2
    env.step([0, 0, 0, 0])
    assert len(env.q[0][0]) == 0 and len(env.q[2][0]) == 2


def test_spillback_when_downstream_full():
    env = TrafficEnv("low", 2, seed=0)
    env._arrivals = [[[0] * 4 for _ in range(4)]] * env.T
    env.q[0][0].extend([0, 0])
    env.q[2][0].extend([0] * ENV.capacity)    # downstream lane full and not served (phase 0 serves it too,
    env.phase[2] = 1                          # so make intersection 2 red for lane N)
    env.step([0, 0, 0, 0])
    assert len(env.q[0][0]) == 2              # blocked


def test_seed_reproducibility():
    a, b = TrafficEnv("rush_hour", 2, seed=5), TrafficEnv("rush_hour", 2, seed=5)
    assert a._arrivals == b._arrivals
