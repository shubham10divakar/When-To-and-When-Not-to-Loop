"""Monte Carlo driver for the SimPy model (design doc §4.2).

``run_sim(params, arm, n_runs, seed, concurrency=1)`` returns one row per
simulated run with the same column names as the flattened L3 run record, so L2
and L3 outputs go through the same analysis code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import simpy

from loopbounds.params import Params
from loopbounds.sim.agent_sim import AgentProcess
from loopbounds.sim.environment import LLMServer, ToolServer


def run_sim(params: Params, arm: str, n_runs: int, seed: int = 0, *, concurrency: int = 1,
            tokens_per_min: float | None = None, max_concurrent_requests: int | None = None,
            tool_samples: np.ndarray | None = None) -> pd.DataFrame:
    """Simulate ``n_runs`` agent runs with ``concurrency`` agents sharing one LLM server.

    Each worker runs episodes back-to-back. With concurrency 1 and no rate limit
    the server has no contention, so results depend only on the parameters.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x51A1]))
    env = simpy.Environment()
    shared = concurrency > 1 or tokens_per_min is not None or max_concurrent_requests is not None
    cap = (max_concurrent_requests or concurrency) if shared else None
    llm = LLMServer(env, params, rng, capacity=cap, tokens_per_min=tokens_per_min)
    tools = ToolServer(env, params, rng, tool_samples)
    rows: list[dict] = []
    counter = iter(range(n_runs))

    def worker(wid: int):
        agent = AgentProcess(env, params, arm, llm, tools, rng)
        for i in counter:
            start = env.now
            yield from agent.run(f"sim-{arm}-{i:06d}", rows)
            rows[-1]["worker"] = wid
            rows[-1]["start_s"] = start

    procs = [env.process(worker(w)) for w in range(concurrency)]
    env.run(until=simpy.AllOf(env, procs))
    df = pd.DataFrame(rows)
    df["concurrency"] = concurrency
    df["makespan_s"] = env.now
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Mean/SE/tail summary used by agreement tests and figures."""
    s = df["success"].astype(float)
    n = len(df)
    return {
        "n": n,
        "p_success": float(s.mean()),
        "p_success_se": float(np.sqrt(s.mean() * (1 - s.mean()) / n)) if n else np.nan,
        "cost_mean": float(df["cost_usd"].mean()),
        "cost_se": float(df["cost_usd"].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        "latency_mean": float(df["wall_s"].mean()),
        "latency_p95": float(df["wall_s"].quantile(0.95)),
        "e2e_p95": float(df["e2e_s"].quantile(0.95)),
        "e2e_p99": float(df["e2e_s"].quantile(0.99)),
        "throughput_runs_per_s": float(n / df["makespan_s"].iloc[0]) if n else np.nan,
        "steps_mean": float(df["steps"].mean()),
    }


def p95_latency_fn(n_runs: int = 2000, seed: int = 0):
    """Adapter for ``model.feasible`` latency constraint (L2 p95)."""
    def fn(params: Params, arm: str) -> float:
        return summarize(run_sim(params, arm, n_runs, seed))["latency_p95"]
    return fn
