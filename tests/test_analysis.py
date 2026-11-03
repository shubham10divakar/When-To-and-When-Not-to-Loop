"""Analysis tests: estimator recovery on mock data, stats helpers, prediction guard."""

import numpy as np
import pandas as pd
import pytest

from loopbounds.analysis import agreement, fit, predict, stats
from loopbounds.harness.logging import read_logs
from loopbounds.harness.runner import run_experiment
from loopbounds.model import reliability
from loopbounds.params import Params


# --- stats ---------------------------------------------------------------------

def test_wilson_known_values():
    p, lo, hi = stats.wilson(8, 10)
    assert p == pytest.approx(0.8)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)


def test_ols_hc3_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 2000)
    y = 1.5 + 2.0 * x + 0.3 * x**2 + rng.normal(0, 1, x.size)
    f = stats.ols_hc3(np.column_stack([np.ones_like(x), x, x**2]), y)
    assert f["coef"] == pytest.approx([1.5, 2.0, 0.3], abs=0.15)
    assert f["r2"] > 0.99


def test_logistic_fit():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = rng.random(5000) < 1 / (1 + np.exp(-(0.5 + 1.5 * x)))
    f = stats.logistic_fit(np.column_stack([np.ones_like(x), x]), y)
    assert f["coef"] == pytest.approx([0.5, 1.5], abs=0.15)


def test_brier_logloss():
    assert stats.brier([1, 0], [1, 0]) == 0
    assert stats.log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))


# --- estimator recovery ----------------------------------------------------------

@pytest.fixture(scope="module")
def markov_logs(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    cfg = {"name": "MK", "phase": "validation",
           "client": {"kind": "markov_mock", "p": 0.88, "r": 0.35, "e": 0.08, "seed": 3},
           "task": {"file_tokens": 50, "n_fake_clues": 0},
           "grid": {"depth": [5, 8], "arm": ["naive"]}, "runs_per_cell": 1000}
    run_experiment(cfg, out_root=root)
    return root


def test_markov_estimators_recover_truth(markov_logs):
    steps, runs = fit.load_experiments(["MK"], markov_logs)
    mk = fit.fit_markov(steps, n_boot=300)
    for k, truth in (("p", 0.88), ("r", 0.35), ("e", 0.08)):
        lo, hi = mk["wilson"][k]
        assert lo <= truth <= hi, (k, mk[k], lo, hi)


def test_fit_all_and_predict_markov_cells(markov_logs, tmp_path):
    res = fit.fit_all(["MK"], markov_logs, n_boot=200)
    assert res.params.p == pytest.approx(0.88, abs=0.02)
    yaml_p, json_p = fit.save_fit(res, tmp_path)
    loaded = fit.load_fit(tmp_path)
    assert loaded.params.p == pytest.approx(res.params.p)
    # held-out prediction (n=12) using the fitted parameters vs fresh mock data
    cfg = {"name": "MKP", "phase": "prediction",
           "client": {"kind": "markov_mock", "p": 0.88, "r": 0.35, "e": 0.08, "seed": 4},
           "task": {"file_tokens": 50, "n_fake_clues": 0}, "grid": {"depth": [12], "arm": ["naive"]},
           "runs_per_cell": 400}
    pred = predict.predict_experiment(cfg, res, n_sim=2000, raw_root=markov_logs, out=tmp_path)
    run_experiment(cfg, out_root=markov_logs)
    with pytest.raises(predict.PredictionAfterData):
        predict.predict_experiment(cfg, res, n_sim=0, raw_root=markov_logs, out=tmp_path)
    _, runs = fit.load_experiments(["MKP"], markov_logs)
    table = agreement.l3_cell_table(pred, runs)
    assert table["success_in_ci"].all()
    assert table["cost_rel_err"].iloc[0] < 0.10
    mvn = agreement.markov_vs_naive(pred, runs)
    assert mvn["brier_markov"] < mvn["brier_naive"]


def test_tokens_fit_matches_construction(markov_logs):
    steps, _ = fit.load_experiments(["MK"], markov_logs)
    tk = fit.fit_tokens(steps)
    assert tk["a_out"] == 60
    assert tk["delta_by_file_tokens"][50] == pytest.approx(110, abs=1)


def test_coupling_estimator_recovers_lambda(tmp_path):
    lam = 2.0e-5
    cfg = {"name": "CP", "client": {"kind": "coupled_mock", "p0": 0.98, "lam": lam, "r": 0.4, "e": 0.0, "seed": 5},
           "task": {"file_tokens": 50, "n_fake_clues": 0},
           "grid": {"depth": [8], "t0_pad": [0, 8000, 20000], "arm": ["naive"]}, "runs_per_cell": 400}
    run_experiment(cfg, out_root=tmp_path)
    steps, _ = fit.load_experiments(["CP"], tmp_path)
    cp = fit.fit_coupling(steps)
    assert cp["exp"]["lam"] == pytest.approx(lam, rel=0.3)
    assert cp["exp"]["p0"] == pytest.approx(0.98, abs=0.03)


def test_retry_mc_agrees():
    r = agreement.retry_delivery_mc(0.2, 0.5, 3, n=50_000)
    assert r["pass"]


def test_l1_l2_table_small():
    P = Params(depth=6, p=0.9, r=0.3, e=0.05, price_in=1e-6, price_out=5e-6)
    t = agreement.l1_l2_table([(P, "naive"), (P, "cached")], n_runs=20_000,
                              thresholds={"l1_l2": {"cost_rel_err": 0.02, "success_se_mult": 3, "retry_se_mult": 3}})
    assert t[t["metric"] == "p_success"]["pass"].all()
