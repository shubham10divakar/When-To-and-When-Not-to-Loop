"""Context trajectory and the context wall (design doc §3.1).

Convention used throughout L1: calls are indexed ``i = 0, 1, ..., m-1``.
``T(i)`` is the input context (tokens) of LLM call ``i``. Under the naive and
cached arms ``T(i) = T0 + i*δ`` (Eq. 1).

The summarized arm (Eq. 5) keeps a history ``h`` (excluding T0). After every
``k`` calls, if the run continues, the history is replaced by a summary of size
``γ·h``; the fixed point is ``h* = γkδ/(1-γ)``. :func:`summarized_schedule` is a
*direct step simulation of the deterministic recurrence*, exact for any n.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loopbounds.params import Params


def context_at_step(i, t0: float, delta: float):
    """Eq. 1: T(i) = T0 + i·δ (naive / cached arms)."""
    return t0 + np.asarray(i) * delta


def n_max(window: float, t0: float, delta: float) -> int:
    """Eq. 2: number of calls a naive loop can make before overflowing W.

    Call i is feasible iff T0 + iδ <= W, so calls 0..floor((W-T0)/δ) fit and the
    first overflowing call index is floor((W-T0)/δ) + 1.
    """
    if delta <= 0:
        return np.iinfo(np.int64).max
    return int(np.floor((window - t0) / delta))


@dataclass(frozen=True)
class SummarySchedule:
    context: np.ndarray        # T(i) for i in 0..m-1 (main-call input context)
    summ_input: np.ndarray     # summary-call input tokens attached to call i (0 if none)
    summ_output: np.ndarray    # summary-call output tokens attached to call i
    h_star: float              # analytic fixed point γkδ/(1-γ)


def summarized_schedule(params: Params, m: int) -> SummarySchedule:
    """Exact deterministic recurrence for the summarized arm.

    A summary is performed *before* call j whenever j > 0 and j % k == 0. It reads
    T0_summ + h and writes γ·h tokens, which become the new history. The cost of
    that summary is attached to call j (so a run that ends at call j-1 never pays
    for it).
    """
    k = int(params.summ_every)
    g = float(params.gamma)
    ctx = np.zeros(m)
    s_in = np.zeros(m)
    s_out = np.zeros(m)
    h = 0.0
    for j in range(m):
        if k > 0 and j > 0 and j % k == 0:
            s_in[j] = params.t0_summ + h
            h = g * h
            s_out[j] = h
        ctx[j] = params.t0 + h
        h += params.delta
    h_star = g * k * params.delta / (1.0 - g) if g < 1 else np.inf
    return SummarySchedule(ctx, s_in, s_out, h_star)


def context_trajectory(arm: str, params: Params, m: int) -> np.ndarray:
    """T(i) for calls 0..m-1 under the given arm."""
    if arm in ("naive", "cached"):
        return context_at_step(np.arange(m), params.t0, params.delta).astype(float)
    if arm == "summarized":
        return summarized_schedule(params, m).context
    raise ValueError(f"unknown arm {arm!r}")
