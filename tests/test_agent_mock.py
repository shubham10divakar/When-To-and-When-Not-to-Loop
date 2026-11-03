"""End-to-end harness tests with mock clients (design doc §9, Harness)."""

import json

import numpy as np
import pytest

from loopbounds.harness import taskgen
from loopbounds.harness.agent import PointerChaseEnv, run_agent
from loopbounds.harness.arms import make_arm
from loopbounds.harness.budget import BudgetExceeded, SpendMeter
from loopbounds.harness.injection import Injector
from loopbounds.harness.llm import CoupledMock, MarkovMock, PerfectMock
from loopbounds.harness.llm.base import Usage
from loopbounds.harness.logging import RunRecord, StepRecord, read_logs
from loopbounds.harness.runner import run_experiment
from loopbounds.model import cost, reliability
from loopbounds.params import Params


def _run(task, llm, arm="naive", M=None, inj=None, k=5, window_cap=None):
    env = PointerChaseEnv(task, inj or Injector(0, 1, 0))
    return run_agent(env, llm, make_arm(arm, k), max_steps=M or 3 * task.depth, run_id="t",
                     window_cap=window_cap)


@pytest.mark.parametrize("difficulty", taskgen.DIFFICULTIES)
@pytest.mark.parametrize("arm", ["naive", "cached"])
def test_perfect_mock_success_and_exact_tokens(difficulty, arm):
    n, ft, a = 7, 300, 60
    task = taskgen.generate(taskgen.TaskConfig(depth=n, file_tokens=ft, difficulty=difficulty, seed=9))
    res = _run(task, PerfectMock(a_out=a), arm)
    assert res.success and res.termination == "answer_correct"
    assert len(res.steps) == n + 1                     # n reads + answer call
    P = Params(t0=res.t0_tokens, delta=a + ft, a_out=a)
    b = cost.cost_breakdown(arm, P, n + 1)
    assert res.usage.input_new == b.input_new
    assert res.usage.cache_read == b.cache_read
    assert res.usage.cache_write == b.cache_write
    assert res.usage.output == b.output
    assert [s.label for s in res.steps] == ["ADVANCE"] * n + ["ANSWER_CORRECT"]


def test_perfect_mock_summarized_close_to_L1():
    n, ft, a, k = 20, 300, 60, 4
    task = taskgen.generate(taskgen.TaskConfig(depth=n, file_tokens=ft, seed=2))
    llm = PerfectMock(a_out=a, gamma=0.25)
    res = _run(task, llm, "summarized", k=k)
    assert res.success
    from loopbounds.harness.arms import SUMMARY_SYSTEM
    from loopbounds.harness.tokens import count_tokens
    P = Params(t0=res.t0_tokens, delta=a + ft, a_out=a, summ_every=k, gamma=0.25,
               t0_summ=count_tokens(SUMMARY_SYSTEM))
    b = cost.cost_breakdown("summarized", P, len(res.steps))
    total_in = res.usage.input_new + res.summary_usage.input_new
    assert total_in == pytest.approx(b.input_new, rel=0.03)
    assert res.summary_usage.output == pytest.approx(b.summary_output, rel=0.03)


def test_silent_injection_misleads_perfect_mock():
    task = taskgen.generate(taskgen.TaskConfig(depth=5, file_tokens=200, seed=1))
    res = _run(task, PerfectMock(), inj=Injector(1.0, 0.0, 0))
    assert not res.success and res.termination == "answer_wrong"
    assert res.steps[0].label == "CORRUPTED"


def test_detectable_injection_with_retries_is_transparent():
    task = taskgen.generate(taskgen.TaskConfig(depth=5, file_tokens=200, seed=1))
    res = _run(task, PerfectMock(), inj=Injector(0.3, 1.0, 50))
    assert res.success
    assert any(s.inject.attempts > 1 for s in res.steps)


def test_window_cap_overflow():
    task = taskgen.generate(taskgen.TaskConfig(depth=30, file_tokens=1000, seed=1))
    res = _run(task, PerfectMock(), window_cap=8000)
    assert res.termination == "context_overflow"
    assert max(s.context_tokens for s in res.steps) <= 8000


def test_markov_mock_matches_dp():
    n, M = 6, 18
    task = taskgen.generate(taskgen.TaskConfig(depth=n, file_tokens=50, n_fake_clues=0, seed=3))
    llm = MarkovMock(p=0.85, r=0.4, e=0.1, seed=1)
    N = 1500
    succ = np.mean([_run(task, llm, M=M).success for _ in range(N)])
    pred = reliability.markov_dp(0.85, 0.4, 0.1, n, M).p_success
    se = np.sqrt(pred * (1 - pred) / N)
    assert abs(succ - pred) < 3.5 * se, (succ, pred)


def test_markov_mock_with_injection_matches_dp():
    from loopbounds.model.retry import delivery
    n, M = 5, 15
    task = taskgen.generate(taskgen.TaskConfig(depth=n, file_tokens=50, n_fake_clues=0, seed=3))
    llm = MarkovMock(p=0.9, r=0.4, e=0.05, seed=2)
    f, d, R = 0.2, 0.5, 1
    N = 1500
    rng_seed = iter(range(10**6))
    succ = np.mean([_run(task, llm, M=M, inj=Injector(f, d, R, seed=next(rng_seed))).success for _ in range(N)])
    dl = delivery(f, d, R)
    pred = reliability.markov_dp(0.9, 0.4, 0.05, n, M, deliver=(dl.clean, dl.silent, dl.exhaust)).p_success
    se = np.sqrt(pred * (1 - pred) / N)
    assert abs(succ - pred) < 3.5 * se, (succ, pred)


def test_coupled_mock_runs():
    task = taskgen.generate(taskgen.TaskConfig(depth=5, file_tokens=100, seed=3, t0_pad=2000))
    res = _run(task, CoupledMock(p0=0.99, lam=1e-6))
    assert res.termination in ("answer_correct", "answer_wrong", "budget")


def test_budget_meter_aborts():
    m = SpendMeter(Params(price_in=1e-3, price_out=1e-3), cap_usd=0.01)
    m.add(Usage(5, 0, 0, 0))
    with pytest.raises(BudgetExceeded):
        m.add(Usage(100, 0, 0, 0))


def test_runner_end_to_end_and_schema(tmp_path):
    cfg = {"name": "T", "phase": "validation", "client": {"kind": "perfect_mock"},
           "task": {"file_tokens": 100, "difficulty": "L2"},
           "grid": {"depth": [3, 4], "arm": ["naive", "cached", "summarized"]},
           "runs_per_cell": 2, "summ_every": 2}
    summ = run_experiment(cfg, out_root=tmp_path)
    assert summ.n_runs == 12 and not summ.aborted
    files = list((tmp_path / "T").glob("*.jsonl"))
    assert len(files) == 12
    for f in files:
        lines = f.read_text().splitlines()
        for ln in lines[:-1]:
            StepRecord.model_validate_json(ln)
        RunRecord.model_validate_json(lines[-1])
    steps, runs = read_logs(tmp_path / "T")
    assert runs["success"].all()
    # paired tasks: same seed across arms within a cell's task definition
    assert runs.groupby(["depth"])["seed"].nunique().max() == 2
    # resume skips existing
    assert run_experiment(cfg, out_root=tmp_path).n_skipped == 12


def test_runner_single_shot(tmp_path):
    cfg = {"name": "S", "mode": "single_shot", "client": {"kind": "perfect_mock"},
           "task": {"file_tokens": 100}, "grid": {"depth": [3, 50]}, "runs_per_cell": 3}
    run_experiment(cfg, out_root=tmp_path)
    _, runs = read_logs(tmp_path / "S")
    assert set(runs["arm"]) == {"single_shot"}
    assert len(runs) == 6
