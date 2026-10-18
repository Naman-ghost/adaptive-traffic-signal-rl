"""Discrete-time traffic simulator: one 4-way intersection or an NxN grid of them.

Lane indices: 0=N, 1=S, 2=E, 3=W. Lane N holds cars travelling south, S north, E west, W east.
Phase 0 serves lanes N,S (north-south green); phase 1 serves lanes E,W.
Cars go straight: a car leaving lane L of one intersection joins lane L of the next intersection
in its direction of travel, or leaves the network at the grid edge. A full downstream lane blocks
the car (spillback).

Timing: switching starts `yellow_steps` steps in which no car passes. `since` counts steps since the
last switch decision (capped at yellow_steps + min_green), so a switch is allowed exactly when
`since == cap`, which guarantees `min_green` green steps for every phase.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import ENV, SCENARIOS, EnvConfig

PHASE_LANES = {0: (0, 1), 1: (2, 3)}
# Where cars leaving lane L go (row, col offset), and which neighbours exist (N,S,E,W order).
_DOWN_OFFSET = {0: (1, 0), 1: (-1, 0), 2: (0, -1), 3: (0, 1)}
_UP_OFFSET = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}
_NEIGHBOUR_DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))  # N, S, E, W neighbours


class TrafficEnv:
    """Single intersection (grid=1) or NxN grid with one signal per intersection."""

    def __init__(self, scenario: str = "medium", grid: int = 1, cfg: EnvConfig = ENV, seed: int = 0) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
        self.cfg, self.scenario, self.grid = cfg, scenario, grid
        self.n_int = grid * grid
        self.cap = cfg.time_cap
        self.T = cfg.episode_len
        self._build_topology()
        self._lane_lut = self._make_lut(cfg.lane_bin_edges, cfg.capacity)
        self._nb_lut = self._make_lut(cfg.neighbour_bin_edges, 4 * cfg.capacity)
        self.n_lane_bins = len(cfg.lane_bin_edges) + 1
        self.n_nb_bins = len(cfg.neighbour_bin_edges) + 1
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _make_lut(edges: Sequence[int], max_val: int) -> List[int]:
        return [sum(v >= e for e in edges) for v in range(max_val + 1)]

    def _build_topology(self) -> None:
        n = self.grid
        idx = lambda r, c: r * n + c
        inside = lambda r, c: 0 <= r < n and 0 <= c < n
        self.down: List[List[int]] = []      # downstream intersection per lane (-1 = leaves network)
        self.boundary: List[List[bool]] = []  # lane gets external arrivals
        self.nbrs: List[List[int]] = []      # neighbouring intersections (N,S,E,W order, existing only)
        for r in range(n):
            for c in range(n):
                self.down.append([idx(r + _DOWN_OFFSET[l][0], c + _DOWN_OFFSET[l][1])
                                  if inside(r + _DOWN_OFFSET[l][0], c + _DOWN_OFFSET[l][1]) else -1 for l in range(4)])
                self.boundary.append([not inside(r + _UP_OFFSET[l][0], c + _UP_OFFSET[l][1]) for l in range(4)])
                self.nbrs.append([idx(r + dr, c + dc) for dr, dc in _NEIGHBOUR_DIRS if inside(r + dr, c + dc)])
        self._nbr_slots = [[idx(r + dr, c + dc) if inside(r + dr, c + dc) else -1 for dr, dc in _NEIGHBOUR_DIRS]
                           for r in range(n) for c in range(n)]

    def _build_arrivals(self, rng: np.random.Generator) -> List[List[List[int]]]:
        """Pre-sample Poisson arrivals, shape (T, n_int, 4). Depends only on the seed, not on the controller."""
        schedule = SCENARIOS[self.scenario]
        rates = np.zeros((self.T, self.n_int, 4))
        for t in range(self.T):
            frac = t / self.T
            seg = [r for start, r in schedule if frac >= start][-1]
            rates[t] = np.array(seg)[None, :]
        mask = np.array(self.boundary, dtype=float)[None, :, :]
        return rng.poisson(rates * mask).tolist()

    # ------------------------------------------------------------------ episode control
    def reset(self, seed: Optional[int] = None) -> None:
        """Start a new episode. The same seed always produces the same arrival stream."""
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        self._arrivals = self._build_arrivals(self.rng)
        self.t = 0
        self.q: List[List[Deque[int]]] = [[deque() for _ in range(4)] for _ in range(self.n_int)]
        self.phase = [0] * self.n_int
        self.since = [self.cap] * self.n_int
        self.yellow = [0] * self.n_int
        self.wait_sum = 0
        self.crossings = 0
        self.exited = 0
        self.switches = 0
        self.max_wait = 0
        self.dropped = 0
        self.queue_sum = 0.0

    # ------------------------------------------------------------------ dynamics
    def can_switch(self, i: int) -> bool:
        return self.since[i] >= self.cap and self.yellow[i] == 0

    def step(self, actions: Sequence[int]) -> Tuple[np.ndarray, bool]:
        """Advance one step. actions[i] in {0: keep, 1: switch}. Returns (per-intersection rewards, done)."""
        cfg, t = self.cfg, self.t
        switched = [0] * self.n_int
        passed = [0] * self.n_int
        pending = [[0] * 4 for _ in range(self.n_int)]
        moves: List[Tuple[int, int]] = []

        for i in range(self.n_int):
            if actions[i] == 1 and self.can_switch(i):          # an illegal switch request is ignored
                self.phase[i] = 1 - self.phase[i]
                self.yellow[i] = cfg.yellow_steps
                self.since[i] = 0
                switched[i] = 1
                self.switches += 1
            if self.yellow[i] > 0:
                self.yellow[i] -= 1                              # yellow/all-red: nobody passes
            else:
                for lane in PHASE_LANES[self.phase[i]]:
                    queue, d = self.q[i][lane], self.down[i][lane]
                    for _ in range(cfg.service_rate):
                        if not queue:
                            break
                        if d >= 0:
                            if len(self.q[d][lane]) + pending[d][lane] >= cfg.capacity:
                                break                            # spillback: downstream lane is full
                            pending[d][lane] += 1
                            moves.append((d, lane))
                        else:
                            self.exited += 1
                        wait = t - queue.popleft()
                        self.wait_sum += wait
                        self.max_wait = max(self.max_wait, wait)
                        self.crossings += 1
                        passed[i] += 1
            self.since[i] = min(self.since[i] + 1, self.cap)

        for d, lane in moves:                                    # cars arrive downstream next step
            self.q[d][lane].append(t + 1)
        arrivals = self._arrivals[t]
        for i in range(self.n_int):
            for lane in range(4):
                for _ in range(arrivals[i][lane]):
                    if len(self.q[i][lane]) < cfg.capacity:
                        self.q[i][lane].append(t + 1)
                    else:
                        self.dropped += 1

        waiting = np.array([sum(len(l) for l in self.q[i]) for i in range(self.n_int)], dtype=float)
        rewards = (-cfg.wait_weight * waiting - cfg.switch_penalty * np.array(switched)
                   + cfg.pass_reward * np.array(passed))
        self.queue_sum += waiting.sum() / (4 * self.n_int)
        self.t += 1
        return rewards, self.t >= self.T

    # ------------------------------------------------------------------ queries used by controllers
    def queue_lengths(self, i: int) -> List[int]:
        return [len(l) for l in self.q[i]]

    def head_wait(self, i: int, lane: int) -> int:
        """Waiting time of the oldest car in a lane (0 if empty)."""
        q = self.q[i][lane]
        return self.t - q[0] if q else 0

    def downstream_queue(self, i: int, lane: int) -> int:
        d = self.down[i][lane]
        return len(self.q[d][lane]) if d >= 0 else 0

    def total_queue(self) -> int:
        return sum(len(l) for inter in self.q for l in inter)

    # ------------------------------------------------------------------ state encodings
    @property
    def local_state_size(self) -> int:
        return self.n_lane_bins ** 4 * 2 * (self.cap + 1)

    def neighbour_state_size(self, i: int) -> int:
        return self.local_state_size * self.n_nb_bins ** len(self.nbrs[i])

    def state_size(self, i: int, mode: str) -> int:
        return self.local_state_size if mode == "local" else self.neighbour_state_size(i)

    def encode_local(self, i: int) -> int:
        s = 0
        for lane_q in self.q[i]:
            s = s * self.n_lane_bins + self._lane_lut[len(lane_q)]
        return (s * 2 + self.phase[i]) * (self.cap + 1) + self.since[i]

    def encode_neighbour(self, i: int) -> int:
        s = self.encode_local(i)
        for j in self.nbrs[i]:
            s = s * self.n_nb_bins + self._nb_lut[sum(len(l) for l in self.q[j])]
        return s

    def encode(self, i: int, mode: str) -> int:
        return self.encode_local(i) if mode == "local" else self.encode_neighbour(i)

    def raw_obs(self, i: int) -> np.ndarray:
        """Unbinned features for function approximation: own 4 queues, 4 neighbour slots x 4 queues
        (zeros if no neighbour), phase, time-since-switch; queues scaled by capacity."""
        cap = float(self.cfg.capacity)
        feats = [len(l) / cap for l in self.q[i]]
        for j in self._nbr_slots[i]:
            feats += [len(l) / cap for l in self.q[j]] if j >= 0 else [0.0] * 4
        feats += [float(self.phase[i]), self.since[i] / self.cap]
        return np.asarray(feats, dtype=np.float32)

    @property
    def raw_obs_dim(self) -> int:
        return 4 + 16 + 2

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> Dict[str, float]:
        """Episode metrics. Cars still queued are included with their wait so far (no hiding of starvation).
        Wait is per car per intersection crossed, in steps."""
        rem_n = rem_sum = rem_max = 0
        for inter in self.q:
            for lane_q in inter:
                for join in lane_q:
                    w = self.t - join
                    rem_n, rem_sum, rem_max = rem_n + 1, rem_sum + w, max(rem_max, w)
        n = self.crossings + rem_n
        return {
            "avg_wait": (self.wait_sum + rem_sum) / n if n else 0.0,
            "avg_queue": self.queue_sum / max(self.t, 1),
            "throughput": float(self.exited),
            "switches": float(self.switches),
            "max_wait": float(max(self.max_wait, rem_max)),
            "dropped": float(self.dropped),
        }
