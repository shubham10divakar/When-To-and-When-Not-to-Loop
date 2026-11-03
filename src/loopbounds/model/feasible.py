"""Feasible region and agent-vs-single-shot dominance (design doc §3.7).

Constraints for the agentic arm (per grid cell):
  1. P(success) >= s_min
  2. E[cost]    <= B
  3. latency    <= SLO   (L1 mean by default; pass ``latency_fn`` for L2 p95)
  4. context of a perfect run <= W (summarized arm bounds it)

Dominance: agent preferred iff all constraints hold AND (single-shot infeasible
OR single-shot success < s_min OR agent cost-per-success < single-shot's).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from loopbounds.model import baseline
from loopbounds.model.context import context_trajectory
from loopbounds.model.cost import expected_cost
from loopbounds.model.latency import expected_latency
from loopbounds.model.reliability import markov_dp, step_p, coupled_p
from loopbounds.model.retry import delivery
from loopbounds.params import Params

CONSTRAINTS = ("success", "cost", "latency", "context")


@dataclass
class CellResult:
    p_success: float
    cost: float
    latency: float
    context: float
    ss_feasible: bool
    ss_success: float
    ss_cost_per_success: float


@dataclass
class FeasibleGrid:
    arm: str
    x_name: str                         # second axis ("p" or "delta")
    n_values: np.ndarray
    x_values: np.ndarray
    metrics: dict[str, np.ndarray]      # each (len(n), len(x))
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    binding: np.ndarray | None = None   # str labels per cell
    feasible: np.ndarray | None = None
    agent_preferred: np.ndarray | None = None

    def boundary(self, mask: np.ndarray | None = None) -> np.ndarray:
        """For each x, the largest n that is in ``mask`` (default: feasible); NaN if none."""
        mask = self.feasible if mask is None else mask
        out = np.full(len(self.x_values), np.nan)
        for j in range(len(self.x_values)):
            idx = np.nonzero(mask[:, j])[0]
            if idx.size:
                out[j] = self.n_values[idx.max()]
        return out


def _solve_batch(params: Params, arm: str, n: int, p_vals: np.ndarray):
    """Solve a batch of cells sharing n (and context) but differing in p."""
    P = params.with_(depth=int(n))
    M = P.M
    d = delivery(P.f_inject, P.d_detect, P.retries)
    ctx = context_trajectory(arm, P, M)
    if P.coupling in (None, "none"):
        p2 = np.repeat(np.asarray(p_vals, float)[:, None], M, axis=1)
    else:  # the grid axis plays the role of p0
        p2 = np.stack([coupled_p(ctx, pv, P.lam, P.coupling) for pv in p_vals])
    res = markov_dp(p2, P.r, P.e, P.depth, M, answer_call=P.answer_call,
                    deliver=(d.clean, d.silent, d.exhaust), context=ctx, window=P.window)
    cost = expected_cost(arm, P, res.end_dist)
    lat = expected_latency(arm, P, res)
    need = context_trajectory(arm, P, P.depth + int(P.answer_call)).max()
    return res.p_success, cost, lat, need


def evaluate_cell(params: Params, arm: str) -> CellResult:
    s, c, lat, ctx = _solve_batch(params, arm, params.depth, np.array([params.p if params.coupling in (None, "none") else params.p0]))
    ss = baseline.single_shot(params)
    return CellResult(float(s[0]), float(c[0]), float(lat[0]), float(ctx), ss.feasible, ss.p_success,
                      ss.cost_per_success)


def _classify(grid: FeasibleGrid, params: Params) -> FeasibleGrid:
    m = grid.metrics
    viol = {
        "success": (params.s_min - m["p_success"]) / params.s_min,
        "cost": (m["cost"] - params.budget) / params.budget,
        "latency": (m["latency"] - params.slo) / params.slo,
        "context": (m["context"] - params.window) / params.window,
    }
    grid.masks = {k: v <= 0 for k, v in viol.items()}
    grid.feasible = np.logical_and.reduce([grid.masks[k] for k in CONSTRAINTS])
    stack = np.stack([viol[k] for k in CONSTRAINTS])
    worst = np.array(CONSTRAINTS, dtype=object)[np.argmax(stack, axis=0)]
    grid.binding = np.where(grid.feasible, "feasible", worst).astype(object)
    with np.errstate(divide="ignore", invalid="ignore"):
        agent_cps = m["cost"] / m["p_success"]
    ss_bad = (~m["ss_feasible"].astype(bool)) | (m["ss_success"] < params.s_min)
    grid.agent_preferred = grid.feasible & (ss_bad | (agent_cps < m["ss_cps"]))
    return grid


def feasible_grid_np(params: Params, arm: str, n_values, p_values,
                     latency_fn: Callable[[Params, str], float] | None = None) -> FeasibleGrid:
    """Grid over (n, p) (Figure 1)."""
    n_values = np.asarray(n_values, int)
    p_values = np.asarray(p_values, float)
    shape = (len(n_values), len(p_values))
    keys = ("p_success", "cost", "latency", "context", "ss_feasible", "ss_success", "ss_cps")
    met = {k: np.zeros(shape) for k in keys}
    for i, n in enumerate(n_values):
        s, c, lat, need = _solve_batch(params, arm, n, p_values)
        met["p_success"][i], met["cost"][i], met["latency"][i], met["context"][i] = s, c, lat, need
        if latency_fn is not None:
            for j, pv in enumerate(p_values):
                met["latency"][i, j] = latency_fn(params.with_(depth=int(n), p=float(pv)), arm)
        ss = baseline.single_shot(params.with_(depth=int(n)))
        met["ss_feasible"][i] = ss.feasible
        met["ss_success"][i] = ss.p_success
        met["ss_cps"][i] = ss.cost_per_success
    return _classify(FeasibleGrid(arm, "p", n_values, p_values, met), params)


def feasible_grid_ndelta(params: Params, arm: str, n_values, delta_values, *,
                         delta_overhead: float | None = None,
                         latency_fn: Callable[[Params, str], float] | None = None) -> FeasibleGrid:
    """Grid over (n, δ). File size for the single-shot baseline is δ - overhead."""
    overhead = params.a_out + 40.0 if delta_overhead is None else delta_overhead
    n_values = np.asarray(n_values, int)
    delta_values = np.asarray(delta_values, float)
    shape = (len(n_values), len(delta_values))
    keys = ("p_success", "cost", "latency", "context", "ss_feasible", "ss_success", "ss_cps")
    met = {k: np.zeros(shape) for k in keys}
    p_axis = np.array([params.p if params.coupling in (None, "none") else params.p0])
    for i, n in enumerate(n_values):
        for j, dl in enumerate(delta_values):
            P = params.with_(depth=int(n), delta=float(dl), file_tokens=max(1.0, float(dl) - overhead))
            s, c, lat, need = _solve_batch(P, arm, n, p_axis)
            met["p_success"][i, j], met["cost"][i, j], met["context"][i, j] = s[0], c[0], need
            met["latency"][i, j] = latency_fn(P, arm) if latency_fn else lat[0]
            ss = baseline.single_shot(P)
            met["ss_feasible"][i, j] = ss.feasible
            met["ss_success"][i, j] = ss.p_success
            met["ss_cps"][i, j] = ss.cost_per_success
    return _classify(FeasibleGrid(arm, "delta", n_values, delta_values, met), params)


def predict_feasible(params: Params, arm: str) -> dict:
    """Point prediction used by E9: is this config inside the agent's feasible region?"""
    cell = evaluate_cell(params, arm)
    ok = {
        "success": cell.p_success >= params.s_min,
        "cost": cell.cost <= params.budget,
        "latency": cell.latency <= params.slo,
        "context": cell.context <= params.window,
    }
    feasible = all(ok.values())
    agent_cps = cell.cost / cell.p_success if cell.p_success > 0 else np.inf
    ss_bad = (not cell.ss_feasible) or cell.ss_success < params.s_min
    return {**{f"ok_{k}": v for k, v in ok.items()}, "feasible": feasible,
            "agent_preferred": feasible and (ss_bad or agent_cps < cell.ss_cost_per_success),
            "p_success": cell.p_success, "cost": cell.cost, "latency": cell.latency, "context": cell.context}


__all__ = ["FeasibleGrid", "feasible_grid_np", "feasible_grid_ndelta", "evaluate_cell", "predict_feasible",
           "step_p"]
