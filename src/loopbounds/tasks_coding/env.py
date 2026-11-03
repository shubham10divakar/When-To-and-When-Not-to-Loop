"""Coding-task environment and runner (design doc §5.10, E10).

Tools: ``read_spec``, ``write_file(path, content)``, ``run_tests``, ``submit``.
Depth is not controlled; it is measured (steps to submission). Solutions run in
a separate, isolated Python subprocess with a timeout.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from loopbounds.harness.agent import run_agent
from loopbounds.harness.arms import make_arm
from loopbounds.harness.budget import SpendMeter, usage_dollars
from loopbounds.harness.llm import make_client
from loopbounds.harness.llm.base import LLMResponse, ToolCall
from loopbounds.harness.llm.mock import MockClient
from loopbounds.harness.logging import RunLogger, RunRecord, git_sha, now_iso
from loopbounds.harness.tokens import count_tokens
from loopbounds.harness.tools import ToolResult
from loopbounds.tasks_coding.tasks import TASKS, CodingTask

SOLUTION_FILE = "solution.py"

TOOLS = [
    {"name": "read_spec", "description": "Return the task specification.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "write_file", "description": f"Write the full contents of {SOLUTION_FILE}.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"], "additionalProperties": False}},
    {"name": "run_tests", "description": "Run the hidden unit tests against solution.py and report results.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "submit", "description": "Submit solution.py as final. Ends the task.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]

_RUNNER = r"""
import json, sys, importlib.util
spec = importlib.util.spec_from_file_location("solution", sys.argv[1])
cases = json.loads(open(sys.argv[2]).read())
res = []
try:
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    fn = getattr(mod, sys.argv[3])
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"})); sys.exit(0)
for args, exp in cases:
    try:
        got = fn(*args)
        res.append({"ok": got == exp, "args": args, "expected": exp, "got": repr(got)[:200]})
    except Exception as e:
        res.append({"ok": False, "args": args, "expected": exp, "got": f"{type(e).__name__}: {e}"})
print(json.dumps({"results": res}))
"""


def run_hidden_tests(task: CodingTask, code: str, timeout: float = 10.0) -> dict:
    with tempfile.TemporaryDirectory() as d:
        sol = Path(d) / SOLUTION_FILE
        sol.write_text(code, encoding="utf-8")
        cases = Path(d) / "cases.json"
        cases.write_text(json.dumps([[list(a), e] for a, e in task.tests]))
        runner = Path(d) / "_runner.py"
        runner.write_text(_RUNNER)
        try:
            out = subprocess.run([sys.executable, "-I", str(runner), str(sol), str(cases), task.func],
                                 capture_output=True, text=True, timeout=timeout, cwd=d)
            return json.loads(out.stdout.strip().splitlines()[-1]) if out.stdout.strip() else \
                {"error": out.stderr[-500:]}
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}


class CodingEnv:
    def __init__(self, task: CodingTask):
        self.task = task
        self.tools = TOOLS
        self.code = ""
        self.best = 0

    def system_prompt(self) -> str:
        return ("You are a software engineer. Use the tools to read the specification, write solution.py, run "
                "the hidden tests, fix failures, and submit when all tests pass. One tool call per step.")

    def task_prompt(self) -> str:
        return f"Task {self.task.id}: implement the function described by read_spec in {SOLUTION_FILE}."

    def execute(self, name, args):
        t = time.perf_counter()
        args = args or {}
        if name == "read_spec":
            r = ToolResult(self.task.spec + f"\n\nFunction name: {self.task.func}")
        elif name == "write_file":
            self.code = str(args.get("content", ""))
            r = ToolResult(f"Wrote {len(self.code)} characters to {SOLUTION_FILE}.")
        elif name == "run_tests":
            res = run_hidden_tests(self.task, self.code)
            if "error" in res:
                r = ToolResult(f"ERROR: {res['error']}")
            else:
                ok = sum(x["ok"] for x in res["results"])
                fails = [x for x in res["results"] if not x["ok"]][:2]
                msg = f"{ok}/{len(res['results'])} tests passed."
                for f_ in fails:
                    msg += f"\nFAIL args={f_['args']} expected={f_['expected']!r} got={f_['got']}"
                r = ToolResult(msg)
                r.correct = ok == len(res["results"])
        elif name == "submit":
            res = run_hidden_tests(self.task, self.code)
            ok = "results" in res and all(x["ok"] for x in res["results"])
            r = ToolResult("Submitted.", done=True, correct=ok)
        else:
            r = ToolResult(f"ERROR: unknown tool {name}")
        r.latency_s = time.perf_counter() - t
        return r

    def label(self, name, args, result):
        lab = {"read_spec": "READ_SPEC", "write_file": "WRITE", "run_tests": "TEST", "submit": "SUBMIT"}.get(name, "NOOP")
        if name == "run_tests" and result is not None:
            lab = "TEST_PASS" if result.correct else "TEST_FAIL"
        if name == "submit" and result is not None:
            lab = "SUBMIT_CORRECT" if result.correct else "SUBMIT_WRONG"
        return {"label": lab, "markov_state": "NA", "state_to": "NA", "phase": "coding", "progress": 0}


class CodingMock(MockClient):
    """Scripted coding agent: read spec, write (possibly buggy) code, test, fix, submit."""
    kind = "coding_mock"

    def __init__(self, p_bug: float = 0.3, **kw):
        super().__init__(**kw)
        self.p_bug = p_bug

    def _reset(self):
        self.stage = 0

    def complete(self, system, messages, tools, *, cache=False, max_tokens=1024) -> LLMResponse:
        task: CodingTask = self.task
        last = ""
        for m in reversed(messages):
            if m["role"] == "user" and isinstance(m["content"], list):
                last = m["content"][0].get("content", "")
                break
        if self.stage == 0:
            name, args = "read_spec", {}
            self.stage = 1
        elif self.stage == 1:
            buggy = self.rng.random() < self.p_bug
            bug = f"\n\ndef {task.func}(*a, **k):\n    return None\n"
            code = task.reference + (bug if buggy else "")
            name, args = "write_file", {"path": SOLUTION_FILE, "content": code}
            self.stage = 2
        elif self.stage == 2:
            name, args = "run_tests", {}
            self.stage = 3
        else:
            if "tests passed" in last and last.split("/")[0] == last.split("/")[1].split(" ")[0]:
                name, args = "submit", {}
            else:
                name, args = "write_file", {"path": SOLUTION_FILE, "content": task.reference}
                self.stage = 2
        total = self.prompt_tokens(system, messages, tools)
        out = min(self.a_out + (count_tokens(args.get("content", "")) if args else 0), max_tokens)
        u = self._usage(total, cache, out)
        ttft, tot = self._latency(u)
        tid = f"toolu_code_{self.stage}_{len(messages)}"
        return LLMResponse(content=[{"type": "tool_use", "id": tid, "name": name, "input": args}],
                           tool_calls=[ToolCall(name, args, tid)], text="", usage=u, ttft_s=ttft, total_s=tot,
                           model_id=self.model_id, stop_reason="tool_use")


def run_coding_experiment(cfg: dict, *, out_root, resume: bool = True, mock: bool = False,
                          tasks: list[CodingTask] | None = None, verbose: bool = False):
    from loopbounds.harness.runner import price_params

    name = cfg["name"]
    tasks = tasks or TASKS
    spec = dict(cfg.get("client") or {})
    llm = CodingMock(seed=int(cfg.get("seed", 0))) if mock or spec.get("kind") != "anthropic" else make_client(spec)
    is_real = getattr(llm, "kind", "") == "anthropic"
    pp = price_params(spec.get("model", "claude-haiku-5-5") if is_real else "mock")
    meter = SpendMeter(pp, float(cfg["max_spend_usd"]) if (is_real and cfg.get("max_spend_usd")) else None)
    logger = RunLogger(out_root, name)
    n = 0
    for task in tasks:
        for arm_name in cfg.get("arms", ["naive", "cached"]):
            for rep in range(int(cfg.get("runs_per_task", 3))):
                run_id = f"{name}-{task.id}-{arm_name}-{rep:02d}"
                if resume and logger.exists(run_id):
                    continue
                env = CodingEnv(task)
                res = run_agent(env, llm, make_arm(arm_name), max_steps=int(cfg.get("max_steps", 20)),
                                run_id=run_id, max_tokens=int(cfg.get("max_tokens", 4096)), spend_hook=meter.add)
                total = res.usage + res.summary_usage
                run = RunRecord(run_id=run_id, experiment=name, phase=cfg.get("phase", "external"), cell_id=task.id,
                                seed=rep, task="coding", arm=arm_name, max_steps=int(cfg.get("max_steps", 20)),
                                success=res.success, termination=res.termination, steps=len(res.steps),
                                t0_tokens=res.t0_tokens,
                                tokens={"input_new": total.input_new, "cache_read": total.cache_read,
                                        "cache_write": total.cache_write, "output": total.output},
                                cost_usd=usage_dollars(pp, total), wall_s=res.wall_s,
                                rate_limit_wait_s=res.rate_limit_wait_s, model_id=res.model_id,
                                client_kind=getattr(llm, "kind", ""), harness_git_sha=git_sha(), started_at=now_iso())
                logger.write(res.steps, run)
                n += 1
                if verbose:
                    print(f"[{name}] {run_id} -> {run.termination}")
    return n


def self_test() -> list[str]:
    """Every reference solution passes its own hidden tests (returns failing task ids)."""
    bad = []
    for t in TASKS:
        r = run_hidden_tests(t, t.reference)
        if "error" in r or not all(x["ok"] for x in r["results"]):
            bad.append(t.id)
    return bad


__all__ = ["CodingEnv", "CodingMock", "run_coding_experiment", "run_hidden_tests", "self_test", "TASKS"]
