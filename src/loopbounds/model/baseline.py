"""Single-shot baseline (design doc §3.6).

One call containing every file plus the task. Success q(n, T_ss) is *empirical*:
a logistic model fitted on single-shot runs (E8) — the weakest modeling
assumption, stated openly in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loopbounds.params import Params

FILE_NAME_OVERHEAD = 8  # tokens per file for the "=== name ===" header


def n_files(params: Params, n=None) -> np.ndarray:
    n = params.depth if n is None else n
    return np.asarray(n) * (1 + params.n_decoy_chains)


def t_ss(params: Params, n=None, file_tokens=None):
    ft = params.file_tokens if file_tokens is None else file_tokens
    return params.t0 + n_files(params, n) * (np.asarray(ft) + FILE_NAME_OVERHEAD)


def q_single_shot(params: Params, n=None, file_tokens=None):
    n = params.depth if n is None else n
    T = t_ss(params, n, file_tokens)
    z = params.ss_b0 + params.ss_b1 * np.asarray(n) + params.ss_b2 * T / 1000.0
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class SingleShot:
    context: float
    feasible: bool
    cost: float
    p_success: float
    latency: float

    @property
    def cost_per_success(self) -> float:
        return self.cost / self.p_success if (self.feasible and self.p_success > 0) else np.inf


def single_shot(params: Params, n=None, file_tokens=None) -> SingleShot:
    T = float(t_ss(params, n, file_tokens))
    feasible = T <= params.window
    cost = params.price_in * T + params.price_out * params.a_ss
    q = float(q_single_shot(params, n, file_tokens)) if feasible else 0.0
    lat = params.ttft_a0 + params.ttft_a1 * T + params.a_ss / params.tps
    return SingleShot(T, feasible, cost if feasible else 0.0, q, lat)
