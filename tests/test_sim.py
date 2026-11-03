"""Layer 2 tests: L1 ↔ L2 agreement at fixed seeds (design doc §7.3 thresholds)."""

import numpy as np
import pytest

from loopbounds.model import cost, latency, reliability, retry
from loopbounds.params import Params
from loopbounds.sim import run_sim, summarize

BASE = Params(depth=8, t0=1500, delta=1100, a_out=60, price_in=1e-6, price_out=5e-6, p=0.9, r=0.35, e=0.05)
N = 20_000


def _l1(P, arm):
    res = reliability.solve(P, arm)
    return res, cost.expected_cost(arm, P, res.end_dist), latency.expected_latency(arm, P, res)


@pytest.mark.parametrize("arm", ["naive", "cached", "summarized"])
def test_l1_l2_success_cost_latency(arm):
    P = BASE.with_(summ_every=3, gamma=0.25)
    res, c1, lat1 = _l1(P, arm)
    s = summarize(run_sim(P, arm, N, seed=1))
    assert abs(s["p_success"] - res.p_success) < 3 * s["p_success_se"] + 1e-9
    assert abs(s["cost_mean"] - c1) < 3 * s["cost_se"]
    assert s["latency_mean"] == pytest.approx(lat1, rel=0.02)


def test_l1_l2_coupled():
    P = BASE.with_(depth=12, coupling="exp", p0=0.98, lam=1.5e-5)
    res, c1, _ = _l1(P, "naive")
    s = summarize(run_sim(P, "naive", N, seed=2))
    assert abs(s["p_success"] - res.p_success) < 3 * s["p_success_se"]


def test_l1_l2_retries():
    P = BASE.with_(f_inject=0.2, d_detect=0.5, retries=2)
    res, c1, _ = _l1(P, "naive")
    df = run_sim(P, "naive", N, seed=3)
    s = summarize(df)
    assert abs(s["p_success"] - res.p_success) < 3 * s["p_success_se"]


def test_l1_l2_context_overflow():
    P = BASE.with_(depth=30, window=1500 + 1100 * 12)
    res, _, _ = _l1(P, "naive")
    df = run_sim(P, "naive", 5000, seed=4)
    obs = (df["termination"] == "context_overflow").mean()
    assert obs == pytest.approx(res.p_fail["context_overflow"], abs=0.03)


def test_concurrency_and_rate_limit_increase_tail_latency():
    P = BASE.with_(depth=5)
    one = summarize(run_sim(P, "naive", 400, seed=5, concurrency=1))
    many = summarize(run_sim(P, "naive", 400, seed=5, concurrency=16, tokens_per_min=3_000_000))
    assert many["e2e_p95"] > one["e2e_p95"]
    assert many["throughput_runs_per_s"] > one["throughput_runs_per_s"]


def test_deterministic_seed():
    a = run_sim(BASE, "cached", 300, seed=7)
    b = run_sim(BASE, "cached", 300, seed=7)
    assert a.equals(b)


@pytest.mark.slow
@pytest.mark.parametrize("arm", ["naive", "cached", "summarized"])
def test_l1_l2_cost_1pct_at_1e5(arm):
    P = BASE.with_(summ_every=3, gamma=0.25)
    _, c1, _ = _l1(P, arm)
    s = summarize(run_sim(P, arm, 100_000, seed=11))
    assert abs(s["cost_mean"] - c1) / c1 < 0.01
