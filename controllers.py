"""Baseline controllers. All share the interface reset(env) / act(env) -> list of actions (0 keep, 1 switch)."""
from __future__ import annotations

from typing import List, Optional

from config import CTRL
from traffic_env import PHASE_LANES, TrafficEnv


class Controller:
    """Base class: one decision per intersection per step."""

    name = "base"

    def reset(self, env: TrafficEnv) -> None:
        """Called at the start of every episode."""

    def select(self, env: TrafficEnv, i: int) -> int:
        raise NotImplementedError

    def act(self, env: TrafficEnv) -> List[int]:
        return [self.select(env, i) for i in range(env.n_int)]


class FixedTimerController(Controller):
    """Requests a phase switch every `period` steps (green time = period - yellow)."""

    name = "fixed"

    def __init__(self, period: int = CTRL.fixed_period) -> None:
        self.period = period
        self.count: List[int] = []

    def reset(self, env: TrafficEnv) -> None:
        self.count = [0] * env.n_int

    def select(self, env: TrafficEnv, i: int) -> int:
        self.count[i] += 1
        if self.count[i] >= self.period:
            self.count[i] = 0
            return 1
        return 0


class LongestQueueFirstController(Controller):
    """Greedy: serve the phase with more waiting cars, respecting min green.

    Optional starvation safeguard: switch if the oldest car of the other phase waited > max_wait steps.
    """

    name = "greedy"

    def __init__(self, max_wait: Optional[int] = CTRL.lqf_max_wait, margin: int = CTRL.lqf_margin) -> None:
        self.max_wait, self.margin = max_wait, margin

    def _phase_load(self, env: TrafficEnv, i: int, phase: int) -> float:
        q = env.queue_lengths(i)
        return sum(q[l] for l in PHASE_LANES[phase])

    def select(self, env: TrafficEnv, i: int) -> int:
        if not env.can_switch(i):
            return 0
        cur = env.phase[i]
        if self._phase_load(env, i, 1 - cur) > self._phase_load(env, i, cur) + self.margin:
            return 1
        if self.max_wait is not None and max(env.head_wait(i, l) for l in PHASE_LANES[1 - cur]) > self.max_wait:
            return 1
        return 0


class MaxPressureController(LongestQueueFirstController):
    """Max-pressure: phase pressure = sum over served lanes of (own queue - downstream queue).

    Identical to greedy on a single intersection (no downstream queues); differs on the grid.
    """

    name = "maxpressure"

    def _phase_load(self, env: TrafficEnv, i: int, phase: int) -> float:
        return sum(len(env.q[i][l]) - env.downstream_queue(i, l) for l in PHASE_LANES[phase])
