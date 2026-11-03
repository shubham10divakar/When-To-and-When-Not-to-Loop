"""Latency model (design doc §3.5).

Per call: L(i) = TTFT(i) + a/τ (+ tool latency·E[A] when the call issues a tool),
TTFT(i) = α0 + α1·T_new(i) + α2·T_cached(i). Summary calls add their own
TTFT + output decode time. L1 returns means; tails come from L2.
"""

from __future__ import annotations

import numpy as np

from loopbounds.model.cost import per_call_tokens
from loopbounds.model.reliability import MarkovResult
from loopbounds.model.retry import expected_attempts
from loopbounds.params import Params


def ttft(params: Params, t_new, t_cached):
    return params.ttft_a0 + params.ttft_a1 * np.asarray(t_new) + params.ttft_a2 * np.asarray(t_cached)


def per_call_llm_latency(arm: str, params: Params, m: int) -> np.ndarray:
    """LLM time (s) of calls 0..m-1, including any summary call attached to them."""
    t = per_call_tokens(arm, params, m)
    main_new = t["input_new"] - t["summary_input"] + t["cache_write"]
    lat = ttft(params, main_new, t["cache_read"]) + params.a_out / params.tps
    has_summ = t["summary_input"] > 0
    lat = lat + np.where(has_summ, ttft(params, t["summary_input"], 0.0) + t["summary_output"] / params.tps, 0.0)
    return lat


def tool_latency_per_call(params: Params) -> float:
    return params.tool_lat * expected_attempts(params.f_inject, params.d_detect, params.retries)


def run_latency(arm: str, params: Params, m: int, n_tool_calls: int | None = None) -> float:
    """Deterministic latency of a run of m calls (default: m-1 tool calls, last call answers)."""
    n_tool = m - 1 if n_tool_calls is None else n_tool_calls
    return float(per_call_llm_latency(arm, params, m).sum() + n_tool * tool_latency_per_call(params))


def expected_latency(arm: str, params: Params, res: MarkovResult) -> float:
    """Mean end-to-end latency under the reliability model's termination distribution."""
    M = res.success_dist.shape[-1] - 1
    cum = np.concatenate([[0.0], np.cumsum(per_call_llm_latency(arm, params, max(M, 1)))])[: M + 1]
    m = np.arange(M + 1)
    tl = tool_latency_per_call(params)
    answered = res.success_dist + res.wrong_dist          # last call was `answer`
    unanswered = res.budget_dist + res.overflow_dist      # every call issued a tool
    v = answered @ (cum + tl * np.maximum(m - 1, 0)) + unanswered @ (cum + tl * m)
    return float(v) if np.ndim(v) == 0 else v
