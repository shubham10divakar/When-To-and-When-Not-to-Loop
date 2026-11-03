"""E10 coding environment, figure rendering without data, and pipeline guards."""

from loopbounds import pipeline
from loopbounds.figures import make_all
from loopbounds.harness.agent import run_agent
from loopbounds.harness.arms import make_arm
from loopbounds.harness.logging import read_logs
from loopbounds.tasks_coding import TASKS, CodingEnv, CodingMock, run_coding_experiment, run_hidden_tests


def test_hidden_tests_detect_bug():
    t = TASKS[0]
    assert all(x["ok"] for x in run_hidden_tests(t, t.reference)["results"])
    bad = run_hidden_tests(t, f"def {t.func}(*a):\n    return None\n")
    assert not all(x["ok"] for x in bad["results"])
    assert "error" in run_hidden_tests(t, "this is not python")


def test_coding_mock_agent_succeeds_with_fix_loop():
    env = CodingEnv(TASKS[1])
    res = run_agent(env, CodingMock(p_bug=1.0, seed=0), make_arm("cached"), max_steps=12, run_id="c")
    assert res.success
    labels = [s.label for s in res.steps]
    assert labels[:3] == ["READ_SPEC", "WRITE", "TEST_FAIL"] and labels[-1] == "SUBMIT_CORRECT"


def test_coding_experiment_logs(tmp_path):
    cfg = {"name": "C", "arms": ["naive"], "runs_per_task": 1, "max_steps": 12}
    n = run_coding_experiment(cfg, out_root=tmp_path, mock=True, tasks=TASKS[:3])
    assert n == 3
    _, runs = read_logs(tmp_path / "C")
    assert runs["success"].all()


def test_figures_render_without_data(tmp_path):
    (tmp_path / "processed").mkdir()
    made = make_all(tmp_path / "processed", tmp_path / "figs", tmp_path / "tabs")
    assert len([p for p in made if p.suffix == ".png"]) == 7
    assert len([p for p in made if p.suffix == ".tex"]) == 4


def test_prereg_check_reports_missing_tag():
    problems = pipeline.prereg_check()
    assert isinstance(problems, list)
    # until the user freezes and tags the pre-registration, the guard must block
    assert any("prereg-v1" in p or "not committed" in p for p in problems)
