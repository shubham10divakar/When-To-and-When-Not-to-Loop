"""Layer 1 unit tests (design doc Â§9, Model)."""

import numpy as np
import pytest

from loopbounds.params import Params, load_params
from loopbounds.model import baseline, context, cost, feasible, latency, reliability, retry


P = Params(t0=1500, delta=1100, a_out=60, window=200_000, price_in=1e-6, price_out=5e-6)


# --- params -------------------------------------------------------------------

def test_load_default_params():
    p = load_params()
    assert p.depth > 0 and p.price_in > 0
    assert p.M == 3 * p.depth


# --- context ------------------------------------------------------------------

def test_context_wall():
    nm = context.n_max(P.window, P.t0, P.delta)
    assert context.context_at_step(nm, P.t0, P.delta) <= P.window
    assert context.context_at_step(nm + 1, P.t0, P.delta) > P.window


def test_summarized_recurrence_converges_to_fixed_point():
    Q = P.with_(summ_every=4, gamma=0.25)
    s = context.summarized_schedule(Q, 400)
    # history right after a summary converges to h* = Î³kÎ´/(1-Î³)
    k = Q.summ_every
    post = s.context[k * 50] - Q.t0
    assert post == pytest.approx(s.h_star, rel=1e-9)
    assert s.context.max() <= Q.t0 + s.h_star + k * Q.delta


# --- cost ---------------------------------------------------------------------

@pytest.mark.parametrize("m", [1, 2, 7, 40])
def test_naive_cost_matches_bruteforce(m):
    brute = sum(P.t0 + i * P.delta for i in range(m))
    assert cost.cost_breakdown("naive", P, m).input_new == pytest.approx(brute)
    assert cost.naive_input_tokens(m, P.t0, P.delta) == pytest.approx(brute)


@pytest.mark.parametrize("m", [1, 3, 25])
def test_cached_closed_form(m):
    b = cost.cost_breakdown("cached", P, m)
    in_dollars = P.price_in * (P.kappa * b.cache_read + (1 + P.omega) * b.cache_write)
    assert in_dollars == pytest.approx(float(cost.cached_input_dollars(m, P)))


def test_cached_leading_coefficient():
    ms = np.array([2000, 4000, 6000])
    Q = P.with_(window=10**12)
    c = np.array([cost.cached_input_dollars(m, Q) for m in ms])
    # second difference / (Î”mÂ²) -> 2Â·coef
    coef = (c[2] - 2 * c[1] + c[0]) / (2 * 2000.0**2)
    assert coef == pytest.approx(cost.leading_coeff("cached", Q), rel=1e-6)
    assert cost.leading_coeff("cached", Q) == pytest.approx(Q.price_in * Q.kappa * Q.delta / 2)


def test_summarized_cost_bounded_per_step_and_linear():
    Q = P.with_(summ_every=5, gamma=0.2)
    per = cost.per_call_dollars("summarized", Q, 2000)
    assert per[1000:].max() <= per[:200].max() * 1.0001  # bounded
    c1 = cost.cost_breakdown("summarized", Q, 1000).dollars
    c2 = cost.cost_breakdown("summarized", Q, 2000).dollars
    assert c2 / c1 == pytest.approx(2.0, rel=0.02)       # O(n)


def test_breakeven_exists():
    Q = P.with_(summ_every=5, gamma=0.2)
    nstar = cost.breakeven_n(Q, "naive", "summarized")
    assert nstar is not None and nstar > Q.summ_every


# --- reliability ------------------------------------------------------------------

@pytest.mark.parametrize("p,n", [(0.9, 5), (0.97, 20), (0.5, 3)])
def test_markov_reduces_to_pn(p, n):
    no_answer = reliability.markov_dp(p, 0.0, 0.0, n, n, answer_call=False)
    assert no_answer.p_success == pytest.approx(p**n, abs=1e-12)
    with_answer = reliability.markov_dp(p, 0.0, 0.0, n, n + 1, answer_call=True)
    assert with_answer.p_success == pytest.approx(p**n, abs=1e-12)


def test_markov_mass_conserved_and_recovery_helps():
    r0 = reliability.markov_dp(0.9, 0.0, 0.05, 10, 30)
    r1 = reliability.markov_dp(0.9, 0.4, 0.05, 10, 30)
    for res in (r0, r1):
        assert res.end_dist.sum() == pytest.approx(1.0, abs=1e-12)
    assert r1.p_success > r0.p_success


def test_markov_batch_matches_scalar():
    ps = np.array([0.8, 0.9, 0.99])
    M = 30
    b = reliability.markov_dp(np.repeat(ps[:, None], M, axis=1), 0.3, 0.05, 10, M)
    for j, pv in enumerate(ps):
        assert b.p_success[j] == pytest.approx(reliability.markov_dp(pv, 0.3, 0.05, 10, M).p_success)


def test_context_overflow_fails_run():
    Q = P.with_(depth=40, window=1500 + 1100 * 10)
    res = reliability.solve(Q, "naive")
    assert res.p_success == 0.0
    assert res.p_fail["context_overflow"] > 0.9
    # summarization bounds context -> no overflow
    assert reliability.solve(Q.with_(summ_every=3, gamma=0.2), "summarized").p_fail["context_overflow"] == 0.0


def test_coupling_reduces_success():
    Q = P.with_(depth=20, coupling="exp", p0=0.98, lam=2e-6)
    assert reliability.solve(Q, "naive").p_success < reliability.solve(Q.with_(lam=0.0), "naive").p_success


# --- retry ----------------------------------------------------------------------

@pytest.mark.parametrize("f,d,R", [(0.2, 0.5, 0), (0.2, 0.9, 3), (0.5, 0.0, 6), (0.0, 1.0, 2), (0.3, 1.0, 4)])
def test_retry_probabilities_sum_to_one(f, d, R):
    dl = retry.delivery(f, d, R)
    assert dl.clean + dl.silent + dl.exhaust == pytest.approx(1.0)


def test_retry_R0_and_limit():
    f, d = 0.2, 0.5
    assert retry.p_clean(f, d, 0) == pytest.approx(1 - f)
    assert retry.p_silent(f, d, 0) == pytest.approx(f * (1 - d))
    assert retry.p_exhaust(f, d, 0) == pytest.approx(f * d)
    assert retry.expected_attempts(f, d, 0) == pytest.approx(1.0)
    assert retry.p_clean(f, d, 200) == pytest.approx(retry.clean_ceiling(f, d))
    assert retry.clean_ceiling(f, d) < 1.0


# --- latency / baseline -------------------------------------------------------

def test_latency_expected_matches_deterministic_for_perfect_agent():
    Q = P.with_(depth=10, p=1.0, e=0.0, r=0.0)
    res = reliability.solve(Q, "naive")
    assert latency.expected_latency("naive", Q, res) == pytest.approx(latency.run_latency("naive", Q, 11))


def test_cached_ttft_lower_than_naive():
    a = latency.per_call_llm_latency("naive", P, 30)
    b = latency.per_call_llm_latency("cached", P, 30)
    assert b[-1] < a[-1]


def test_single_shot_infeasible_when_too_big():
    Q = P.with_(depth=200, file_tokens=3000, window=200_000)
    assert not baseline.single_shot(Q).feasible


# --- feasible region ----------------------------------------------------------

def _grid(Q, arm="naive"):
    return feasible.feasible_grid_np(Q, arm, np.arange(2, 41, 2), np.linspace(0.8, 1.0, 11))


def test_feasible_monotone_in_budget():
    Q = P.with_(budget=0.02, slo=1e6, s_min=0.5)
    small = _grid(Q).feasible
    big = _grid(Q.with_(budget=0.2)).feasible
    assert np.all(big >= small)
    assert big.sum() > small.sum()


def test_feasible_monotone_in_smin_and_p():
    Q = P.with_(budget=10.0, slo=1e6, s_min=0.6)
    g = _grid(Q)
    assert np.all(_grid(Q.with_(s_min=0.9)).feasible <= g.feasible)
    # along p (columns), feasibility never turns off as p increases
    assert np.all(np.diff(g.feasible.astype(int), axis=1) >= 0)


def test_binding_labels():
    Q = P.with_(budget=0.01, slo=1e6, s_min=0.5)
    g = _grid(Q)
    assert set(np.unique(g.binding)) <= {"feasible", "success", "cost", "latency", "context"}
    assert "cost" in set(np.unique(g.binding))


def test_ndelta_grid_runs():
    Q = P.with_(budget=0.05, slo=120, s_min=0.6)
    g = feasible.feasible_grid_ndelta(Q, "cached", [5, 10, 20], [500, 1000, 3000])
    assert g.feasible.shape == (3, 3)
