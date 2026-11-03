"""Reliability models (design doc §3.3).

* :func:`p_naive` — independent steps, Eq. 6: pⁿ.
* :func:`markov_dp` — exact dynamic program over (call, advances, state) for the
  ON/DER recovery model, with optional step-dependent p (coupling, Eq. 8), tool
  delivery outcomes (retry model §3.4), a call budget M and the context wall W.

Transitions at each LLM call (``Pc, Ps, Pe`` = clean / silent / exhausted
delivery probabilities of a true-chain read):

ON  (k < n):  advance           p·Pc         → ON, k+1
              derail            (1-p) + p·Ps → DER
              error surfaced    p·Pe         → ON (no progress)
DER (k < n):  recover           r·Pc         → ON, k+1
              wrong answer      e            → FAIL
              stay derailed     1 - r·Pc - e
ON  (k = n):  answer call       1            → SUCCESS   (if ``answer_call``)

A run also fails if it reaches the call budget M, or if the next call's context
T(t) would exceed the window W (context overflow).

The DP is vectorized over a leading batch axis so a whole feasible-region grid
column is solved in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loopbounds.model.context import context_trajectory
from loopbounds.model.retry import delivery
from loopbounds.params import Params


def p_naive(p, n):
    """Eq. 6."""
    return np.asarray(p, dtype=float) ** np.asarray(n)


def coupled_p(T, p0: float, lam: float, form: str = "exp"):
    """Eq. 8: p(T) = p0·exp(-λT) (exp) or clip(p0 - λT, 0, 1) (linear)."""
    T = np.asarray(T, dtype=float)
    if form == "exp":
        return np.clip(p0 * np.exp(-lam * T), 0.0, 1.0)
    if form == "linear":
        return np.clip(p0 - lam * T, 0.0, 1.0)
    raise ValueError(form)


def step_p(params: Params, arm: str, M: int) -> np.ndarray:
    """Per-call advance probability p(t), t = 0..M-1."""
    if params.coupling in (None, "none"):
        return np.full(M, float(params.p))
    T = context_trajectory(arm, params, M)
    return coupled_p(T, params.p0, params.lam, params.coupling)


@dataclass(frozen=True)
class MarkovResult:
    """Termination distributions over the number of LLM calls m = 0..M.

    Arrays have shape (M+1,) or (B, M+1) for batched solves.
    """
    success_dist: np.ndarray
    wrong_dist: np.ndarray
    budget_dist: np.ndarray
    overflow_dist: np.ndarray

    @property
    def p_success(self):
        s = self.success_dist.sum(axis=-1)
        return float(s) if np.ndim(s) == 0 else s

    @property
    def end_dist(self) -> np.ndarray:
        return self.success_dist + self.wrong_dist + self.budget_dist + self.overflow_dist

    @property
    def expected_calls(self):
        m = np.arange(self.success_dist.shape[-1])
        v = (self.end_dist * m).sum(axis=-1)
        return float(v) if np.ndim(v) == 0 else v

    @property
    def expected_calls_given_success(self):
        m = np.arange(self.success_dist.shape[-1])
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (self.success_dist * m).sum(axis=-1) / self.success_dist.sum(axis=-1)
        return float(v) if np.ndim(v) == 0 else v

    @property
    def p_fail(self) -> dict:
        f = lambda a: (float(a.sum()) if a.ndim == 1 else a.sum(axis=-1))  # noqa: E731
        return {"answer_wrong": f(self.wrong_dist), "budget": f(self.budget_dist),
                "context_overflow": f(self.overflow_dist)}

    def row(self, b: int) -> "MarkovResult":
        return MarkovResult(self.success_dist[b], self.wrong_dist[b], self.budget_dist[b], self.overflow_dist[b])


def markov_dp(p, r, e, n: int, M: int, *, answer_call: bool = True,
              deliver=(1.0, 0.0, 0.0), context=None, window: float | None = None) -> MarkovResult:
    """Exact DP, O(M·n) per batch element.

    p        : scalar, (M,) per-call array, or (B, M) batch (use p[:, None] for a
               batch of constant probabilities).
    r, e     : scalars or (B,) arrays.
    deliver  : (P_clean, P_silent, P_exhaust) for true-chain reads.
    context  : optional (M,) or (B, M) input-context per call; with ``window``,
               a call is only made if context <= window, else the run overflows.
    """
    n, M = int(n), int(M)
    pc, ps, pe = (float(x) for x in deliver)
    p_in = np.asarray(p, dtype=float)
    batched = p_in.ndim == 2 or np.ndim(r) == 1 or np.ndim(e) == 1 \
        or (context is not None and np.ndim(context) == 2)
    # normalise everything to (B, ·). 1-D p is always a per-call array.
    if p_in.ndim == 0:
        p2 = np.full((1, M), float(p_in))
    elif p_in.ndim == 1:
        p2 = p_in[None, :M]
    else:
        p2 = p_in[:, :M]
    B = max(p2.shape[0], np.size(r) if np.ndim(r) else 1, np.size(e) if np.ndim(e) else 1,
            np.shape(context)[0] if context is not None and np.ndim(context) == 2 else 1)
    p2 = np.broadcast_to(p2, (B, M))
    rb = np.broadcast_to(np.asarray(r, float), (B,)).copy()
    eb = np.broadcast_to(np.asarray(e, float), (B,)).copy()
    if np.any(rb * pc + eb > 1 + 1e-12):
        raise ValueError("r·P_clean + e must be <= 1")
    ctx = None
    if context is not None and window is not None:
        ctx = np.broadcast_to(np.asarray(context, float)[..., :M], (B, M))

    succ = np.zeros((B, M + 1))
    wrong = np.zeros((B, M + 1))
    budget = np.zeros((B, M + 1))
    over = np.zeros((B, M + 1))
    on = np.zeros((B, n + 1))
    der = np.zeros((B, max(n, 1)))[:, :n]

    if n == 0 and not answer_call:
        succ[:, 0] = 1.0
    else:
        on[:, 0] = 1.0
        for t in range(M):
            if ctx is not None:
                ovf = ctx[:, t] > window
                if ovf.any():
                    over[ovf, t] += on[ovf].sum(axis=1) + der[ovf].sum(axis=1)
                    on[ovf] = 0.0
                    der[ovf] = 0.0
            pt = p2[:, t][:, None]
            new_on = np.zeros_like(on)
            new_der = np.zeros_like(der)
            if answer_call:
                succ[:, t + 1] += on[:, n]
            if n > 0:
                adv = on[:, :n] * pt * pc + der * (rb * pc)[:, None]
                if answer_call:
                    new_on[:, 1:] += adv
                else:
                    new_on[:, 1:n] += adv[:, : n - 1]
                    succ[:, t + 1] += adv[:, n - 1]
                new_on[:, :n] += on[:, :n] * pt * pe
                new_der += on[:, :n] * ((1.0 - pt) + pt * ps)
                wrong[:, t + 1] += (der * eb[:, None]).sum(axis=1)
                new_der += der * (1.0 - rb * pc - eb)[:, None]
            on, der = new_on, new_der
            if not (on.any() or der.any()):
                break
        budget[:, M] += on.sum(axis=1) + der.sum(axis=1)

    res = MarkovResult(succ, wrong, budget, over)
    return res if batched else res.row(0)


def p_success_markov(p, r, e, n, M, **kw):
    return markov_dp(p, r, e, n, M, **kw).p_success


def solve(params: Params, arm: str = "naive", *, use_window: bool = True) -> MarkovResult:
    """Full L1 reliability for a parameter set and context arm."""
    M = params.M
    d = delivery(params.f_inject, params.d_detect, params.retries)
    ctx = context_trajectory(arm, params, M) if use_window else None
    return markov_dp(step_p(params, arm, M), params.r, params.e, params.depth, M,
                     answer_call=params.answer_call, deliver=(d.clean, d.silent, d.exhaust),
                     context=ctx, window=params.window if use_window else None)
