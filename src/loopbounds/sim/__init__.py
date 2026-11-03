"""Layer 2 — SimPy discrete-event simulation (Monte Carlo cross-check of L1 + time dimension)."""

from loopbounds.sim.run_sim import p95_latency_fn, run_sim, summarize

__all__ = ["run_sim", "summarize", "p95_latency_fn"]
