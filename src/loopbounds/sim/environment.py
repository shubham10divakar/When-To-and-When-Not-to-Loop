"""Simulated LLM server and tool server (design doc §4.1)."""

from __future__ import annotations

import numpy as np
import simpy

from loopbounds.params import Params


def lognormal_mean1(rng: np.random.Generator, sigma: float) -> float:
    """Multiplicative noise with E = 1."""
    if sigma <= 0:
        return 1.0
    return float(rng.lognormal(-0.5 * sigma * sigma, sigma))


class LLMServer:
    """Concurrent-request limit (SimPy Resource) + optional token-bucket rate limit.

    Service time = (TTFT(T_new, T_cached) + out/τ) · lognormal noise (mean 1).
    """

    def __init__(self, env: simpy.Environment, params: Params, rng: np.random.Generator, *,
                 capacity: int | None = None, tokens_per_min: float | None = None):
        self.env = env
        self.params = params
        self.rng = rng
        self.resource = simpy.Resource(env, capacity) if capacity else None
        self.bucket = None
        if tokens_per_min:
            self.bucket_cap = float(tokens_per_min)
            self.bucket = simpy.Container(env, capacity=self.bucket_cap, init=self.bucket_cap)
            env.process(self._refill(tokens_per_min / 60.0))

    def _refill(self, per_s: float):
        while True:
            yield self.env.timeout(1.0)
            room = self.bucket_cap - self.bucket.level
            if room > 0:
                yield self.bucket.put(min(per_s, room))

    def service_time(self, t_new: float, t_cached: float, out: float) -> float:
        P = self.params
        base = P.ttft_a0 + P.ttft_a1 * t_new + P.ttft_a2 * t_cached + out / P.tps
        return base * lognormal_mean1(self.rng, P.llm_lat_sigma)

    def call(self, t_new: float, t_cached: float, out: float, stats: dict):
        """SimPy process: wait for rate limit + slot, then serve. Updates ``stats``."""
        t_start = self.env.now
        if self.bucket is not None:
            need = min(t_new + t_cached, self.bucket_cap)
            yield self.bucket.get(need)
        t_rl = self.env.now
        stats["rate_limit_wait_s"] += t_rl - t_start
        if self.resource is not None:
            with self.resource.request() as req:
                yield req
                stats["queue_wait_s"] += self.env.now - t_rl
                s = self.service_time(t_new, t_cached, out)
                yield self.env.timeout(s)
        else:
            s = self.service_time(t_new, t_cached, out)
            yield self.env.timeout(s)
        stats["llm_s"] += s


class ToolServer:
    """Tool latency (lognormal or bootstrap from L3 samples) + failure injection with retries."""

    def __init__(self, env: simpy.Environment, params: Params, rng: np.random.Generator,
                 samples: np.ndarray | None = None):
        self.env = env
        self.params = params
        self.rng = rng
        self.samples = None if samples is None or len(samples) == 0 else np.asarray(samples, float)

    def latency(self) -> float:
        if self.samples is not None:
            return float(self.samples[self.rng.integers(len(self.samples))])
        return self.params.tool_lat * lognormal_mean1(self.rng, self.params.tool_lat_sigma)

    def read(self, injected: bool, stats: dict):
        """SimPy process; returns outcome clean | silent | exhausted (non-chain reads are clean)."""
        P = self.params
        attempts = 0
        outcome = "exhausted"
        for _ in range(P.retries + 1 if injected else 1):
            attempts += 1
            t = self.latency()
            yield self.env.timeout(t)
            stats["tool_s"] += t
            if not injected or self.rng.random() >= P.f_inject:
                outcome = "clean"
                break
            if self.rng.random() < P.d_detect:
                continue
            outcome = "silent"
            break
        stats["tool_attempts"] += attempts
        return outcome
