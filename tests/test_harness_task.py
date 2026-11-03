"""Layer 3 harness tests: taskgen, injection, classifier (design doc §9, Harness)."""

import re

import numpy as np
import pytest

from loopbounds.harness import taskgen
from loopbounds.harness.classify import Classifier
from loopbounds.harness.injection import Injector, TRANSIENT_ERROR
from loopbounds.harness.tokens import count_tokens
from loopbounds.harness.tools import Toolset
from loopbounds.model import retry


def _valid_targets(task, content):
    key = re.escape(task.key)
    return re.findall(rf"^\[{key}\] (NEXT FILE.*|ANSWER: .*)$", content, flags=re.M)


@pytest.mark.parametrize("difficulty", taskgen.DIFFICULTIES)
def test_chain_integrity(difficulty):
    cfg = taskgen.TaskConfig(depth=8, file_tokens=300, difficulty=difficulty, seed=3)
    t = taskgen.generate(cfg)
    assert len(t.chain) == 8 and t.start == t.chain[0]
    assert len(set(t.files)) == 8 * (1 + cfg.n_decoy_chains)
    for i, name in enumerate(t.chain):
        lines = _valid_targets(t, t.files[name])
        expected_lines = 2 if (difficulty == "L3" and i < 7) else 1
        assert len(lines) == expected_lines, (name, lines)  # exactly one valid clue
        joined = " ".join(lines)
        if i < 7:
            nxt = t.chain[i + 1]
            assert nxt in joined or nxt[::-1] in joined or (nxt[::-1][:3] in joined and nxt[::-1][3:] in joined)
        else:
            assert f"ANSWER: {t.answer}" in joined


def test_deterministic_under_seed():
    cfg = taskgen.TaskConfig(depth=5, seed=11)
    a, b = taskgen.generate(cfg), taskgen.generate(cfg)
    assert a.files == b.files and a.answer == b.answer
    assert taskgen.generate(taskgen.TaskConfig(depth=5, seed=12)).files != a.files


@pytest.mark.parametrize("ft", [200, 1000, 3000])
def test_padding_exact(ft):
    t = taskgen.generate(taskgen.TaskConfig(depth=4, file_tokens=ft, difficulty="L3", seed=1))
    for content in list(t.files.values()) + list(t.silent_files.values()):
        assert abs(count_tokens(content) - ft) <= 0.05 * ft
        assert count_tokens(content) == ft


def test_decoys_reachable_only_via_injection_or_wrong_read():
    t = taskgen.generate(taskgen.TaskConfig(depth=6, difficulty="L1", seed=5))
    decoy_set = {x for dc in t.decoys for x in dc}
    for name in t.chain:
        for line in _valid_targets(t, t.files[name]):
            assert not any(d in line for d in decoy_set)
        # silent version points into a decoy chain (or a fake answer)
        assert t.decoy_map[name] in decoy_set | set(t.decoy_answers)
        assert t.decoy_map[name] in t.silent_files[name]


def test_t0_pad():
    t = taskgen.generate(taskgen.TaskConfig(depth=3, t0_pad=4000, seed=0))
    assert count_tokens(taskgen.system_prompt(t)) > 4000


# --- injection ------------------------------------------------------------------

@pytest.mark.parametrize("f,d,R", [(0.2, 0.5, 0), (0.2, 0.9, 3), (0.3, 0.2, 1)])
def test_injection_rates(f, d, R):
    inj = Injector(f, d, R, seed=7)
    N = 100_000
    outs = [inj.draw() for _ in range(N)]
    dl = retry.delivery(f, d, R)
    for key, pexp in (("clean", dl.clean), ("silent", dl.silent), ("exhausted", dl.exhaust)):
        obs = np.mean([o.outcome == key for o in outs])
        se = np.sqrt(pexp * (1 - pexp) / N) + 1e-9
        assert abs(obs - pexp) < 3.5 * se, (key, obs, pexp)
    assert max(o.attempts for o in outs) <= R + 1
    assert np.mean([o.attempts for o in outs]) == pytest.approx(dl.attempts, rel=0.01)


def test_injection_only_on_chain_files():
    t = taskgen.generate(taskgen.TaskConfig(depth=4, seed=2))
    tools = Toolset(t, Injector(1.0, 1.0, 0, seed=0))
    assert tools.execute("read_file", {"name": t.chain[1]}).content == TRANSIENT_ERROR
    decoy = t.decoys[0][0]
    assert tools.execute("read_file", {"name": decoy}).content == t.files[decoy]
    assert tools.execute("read_file", {"name": "zzzzzz"}).content.startswith("ERROR: no such file")


# --- classifier ------------------------------------------------------------------

def test_classifier_trace():
    t = taskgen.generate(taskgen.TaskConfig(depth=3, seed=4))
    c = Classifier(t)
    trace = [
        (("list_files", {}, "none"), ("LIST", "ON", "ON", 0)),
        (("read_file", {"name": t.chain[0]}, "clean"), ("ADVANCE", "ON", "ON", 1)),
        (("read_file", {"name": t.decoys[0][1]}, "none"), ("WRONG", "ON", "DER", 1)),
        (("read_file", {"name": t.chain[0]}, "clean"), ("REDUNDANT", "DER", "DER", 1)),
        (("read_file", {"name": t.chain[1]}, "clean"), ("ADVANCE", "DER", "ON", 2)),
        (("read_file", {"name": t.chain[2]}, "exhausted"), ("ERROR", "ON", "ON", 2)),
        (("read_file", {"name": t.chain[2]}, "silent"), ("CORRUPTED", "ON", "DER", 2)),
        (("read_file", {"name": t.chain[2]}, "clean"), ("ADVANCE", "DER", "ON", 3)),
        (("answer", {"token": t.answer}, "none"), ("ANSWER_CORRECT", "ON", "ON", 3)),
    ]
    for (tool, args, outc), (lab, s0, s1, prog) in trace:
        sl = c.step(tool, args, outc)
        assert (sl.label, sl.state_from, sl.state_to, sl.progress) == (lab, s0, s1, prog)
    assert sl.phase == "answer"


def test_classifier_skip_ahead_is_wrong():
    t = taskgen.generate(taskgen.TaskConfig(depth=4, seed=4))
    c = Classifier(t)
    assert c.step("read_file", {"name": t.chain[2]}).label == "WRONG"
    assert c.step("answer", {"token": "NOPE"}).label == "ANSWER_WRONG"
