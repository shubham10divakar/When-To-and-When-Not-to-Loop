"""Spend guard (design doc §5.9).

Before an experiment starts, L1 estimates its total cost; the runner refuses to
start if the estimate exceeds ``max_spend_usd``. During the run a
:class:`SpendMeter` accumulates actual spend and raises :class:`BudgetExceeded`
when the cap is hit (partial results are kept and flagged).
"""

from __future__ import annotations

from loopbounds.harness.llm.base import Usage
from loopbounds.model import baseline, cost, reliability
from loopbounds.params import Params


class BudgetExceeded(RuntimeError):
    pass


def usage_dollars(params: Params, u: Usage) -> float:
    return float(cost.dollars(params, u.input_new, u.cache_read, u.cache_write, u.output))


class SpendMeter:
    def __init__(self, params: Params, cap_usd: float | None):
        self.params = params
        self.cap = cap_usd
        self.spent = 0.0

    def add(self, u: Usage) -> None:
        self.spent += usage_dollars(self.params, u)
        if self.cap is not None and self.spent > self.cap:
            raise BudgetExceeded(f"spend cap ${self.cap:.2f} exceeded (${self.spent:.4f})")


def estimate_cell_cost(params: Params, arm: str) -> float:
    """L1 expected cost of one run of this cell (agent arms or 'single_shot')."""
    if arm == "single_shot":
        ss = baseline.single_shot(params)
        return ss.cost if ss.feasible else 0.0
    res = reliability.solve(params, arm)
    return float(cost.expected_cost(arm, params, res.end_dist))


def estimate_experiment_cost(cells: list[tuple[Params, str, int]]) -> float:
    """cells: (params, arm, n_runs)."""
    return float(sum(estimate_cell_cost(p, a) * n for p, a, n in cells))
