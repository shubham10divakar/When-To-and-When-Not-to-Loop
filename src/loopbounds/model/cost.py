"""Token cost formulas (design doc Â§3.2).

Token categories are disjoint, matching the provider's usage fields:
``input_new`` (uncached, full price), ``cache_read`` (ÎºÂ·c_in), ``cache_write``
((1+Ï‰)Â·c_in) and ``output`` (c_out).

``cost(m)`` is the cost of a run that makes ``m`` LLM calls. Expected cost under
the stochastic reliability model is ``Î£_m P(run ends at m) Â· cost(m)``
(:func:`expected_cost`).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from loopbounds.model.context import context_at_step, n_max, summarized_schedule
from loopbounds.params import Params


@dataclass(frozen=True)
class CostBreakdown:
    input_new: float
    cache_read: float
    cache_write: float
    output: float
    summary_input: float
    summary_output: float
    dollars: float

    @property
    def total_input(self) -> float:
        return self.input_new + self.cache_read + self.cache_write

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total_input"] = self.total_input
        return d


def dollars(params: Params, input_new, cache_read, cache_write, output):
    return params.price_in * (
        np.asarray(input_new) + params.kappa * np.asarray(cache_read)
        + (1.0 + params.omega) * np.asarray(cache_write)
    ) + params.price_out * np.asarray(output)


def per_call_tokens(arm: str, params: Params, m: int) -> dict[str, np.ndarray]:
    """Per-call token arrays (length m) for calls 0..m-1.

    Summary-call tokens are attached to the call they precede and are counted
    inside ``input_new`` / ``output`` as well as in the ``summary_*`` arrays.
    """
    i = np.arange(m, dtype=float)
    out = np.full(m, float(params.a_out))
    zeros = np.zeros(m)
    if arm == "naive":
        T = context_at_step(i, params.t0, params.delta)
        return dict(input_new=T, cache_read=zeros, cache_write=zeros.copy(), output=out,
                    summary_input=zeros.copy(), summary_output=zeros.copy())
    if arm == "cached":
        # call 0 writes T0; call i>=1 reads T(i-1) and writes the Î´ new tokens.
        cr = np.where(i >= 1, context_at_step(i - 1, params.t0, params.delta), 0.0)
        cw = np.where(i >= 1, params.delta, params.t0)
        return dict(input_new=zeros, cache_read=cr, cache_write=cw.astype(float), output=out,
                    summary_input=zeros.copy(), summary_output=zeros.copy())
    if arm == "summarized":
        s = summarized_schedule(params, m)
        return dict(input_new=s.context + s.summ_input, cache_read=zeros, cache_write=zeros.copy(),
                    output=out + s.summ_output, summary_input=s.summ_input, summary_output=s.summ_output)
    raise ValueError(f"unknown arm {arm!r}")


def per_call_dollars(arm: str, params: Params, m: int) -> np.ndarray:
    t = per_call_tokens(arm, params, m)
    return dollars(params, t["input_new"], t["cache_read"], t["cache_write"], t["output"])


def cost_breakdown(arm: str, params: Params, m: int) -> CostBreakdown:
    """Deterministic cost of a run of exactly ``m`` LLM calls."""
    t = {k: float(v.sum()) for k, v in per_call_tokens(arm, params, m).items()}
    return CostBreakdown(
        input_new=t["input_new"], cache_read=t["cache_read"], cache_write=t["cache_write"],
        output=t["output"], summary_input=t["summary_input"], summary_output=t["summary_output"],
        dollars=float(dollars(params, t["input_new"], t["cache_read"], t["cache_write"], t["output"])),
    )


# ---------------------------------------------------------------------------
# Closed forms (used in the paper and checked against per_call_tokens in tests)
# ---------------------------------------------------------------------------

def naive_input_tokens(m, t0: float, delta: float):
    """Eq. 3: C_in(m) = mÂ·T0 + Î´Â·m(m-1)/2."""
    m = np.asarray(m, dtype=float)
    return m * t0 + delta * m * (m - 1) / 2.0


def cached_input_dollars(m, params: Params):
    """Eq. 4: c_inÂ·[T0(1+Ï‰) + Î£_{i=1}^{m-1}(ÎºÂ·T(i-1) + (1+Ï‰)Î´)]."""
    m = np.asarray(m, dtype=float)
    j = np.maximum(m - 1, 0)
    reads = j * params.t0 + params.delta * j * (j - 1) / 2.0
    return params.price_in * (params.t0 * (1 + params.omega) + params.kappa * reads
                              + (1 + params.omega) * params.delta * j)


def leading_coeff(arm: str, params: Params) -> float:
    """Coefficient of mÂ² in input dollars: c_inÂ·Î´/2 (naive), c_inÂ·ÎºÂ·Î´/2 (cached), 0 (summarized)."""
    if arm == "naive":
        return params.price_in * params.delta / 2.0
    if arm == "cached":
        return params.price_in * params.kappa * params.delta / 2.0
    if arm == "summarized":
        return 0.0
    raise ValueError(arm)


def breakeven_n(params: Params, arm_a: str, arm_b: str, n_hi: int | None = None) -> int | None:
    """Smallest call count m at which cumulative cost(arm_b) < cost(arm_a) (first crossing).

    Scans m = 1..n_hi (default: the naive context wall). Returns None if no crossing.
    """
    n_hi = n_hi or max(2, min(n_max(params.window, params.t0, params.delta) + 1, 100_000))
    ca = np.cumsum(per_call_dollars(arm_a, params, n_hi))
    cb = np.cumsum(per_call_dollars(arm_b, params, n_hi))
    idx = np.nonzero(cb < ca * (1 - 1e-12))[0]
    return int(idx[0] + 1) if idx.size else None


def expected_cost(arm: str, params: Params, end_dist: np.ndarray) -> float:
    """E[cost] given the distribution of the number of calls a run makes.

    ``end_dist[..., m]`` = P(run makes exactly m calls), m = 0..M (batched OK).
    """
    end_dist = np.asarray(end_dist)
    M = end_dist.shape[-1] - 1
    if M <= 0:
        return 0.0
    cum = np.concatenate([[0.0], np.cumsum(per_call_dollars(arm, params, M))])
    v = end_dist @ cum
    return float(v) if np.ndim(v) == 0 else v
