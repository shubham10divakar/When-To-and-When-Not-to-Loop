"""JSONL step + run records (design doc §5.7), validated with pydantic.

``data/raw/{experiment}/{run_id}.jsonl``: one ``step`` record per LLM call,
then one ``run`` record. Rate-limit backoff time is logged separately and
excluded from ``wall_s``.
"""

from __future__ import annotations

import datetime as _dt
import functools
import json
import subprocess
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from loopbounds.params import REPO_ROOT

TERMINATIONS = ("answer_correct", "answer_wrong", "budget", "context_overflow", "error", "infeasible",
                "spend_cap")


class InjectInfo(BaseModel):
    attempts: int = 0
    types: list[str] = Field(default_factory=list)
    outcome: str = "none"


class StepRecord(BaseModel):
    type: Literal["step"] = "step"
    run_id: str
    step: int
    arm: str
    context_tokens: int
    input_tokens_new: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    ttft_s: float
    llm_total_s: float
    tool_name: str | None = None
    tool_arg: str | None = None
    tool_latency_s: float = 0.0
    inject: InjectInfo = Field(default_factory=InjectInfo)
    label: str
    markov_state: str
    state_to: str
    phase: str
    progress: int
    summary_input_tokens: int = 0
    summary_output_tokens: int = 0
    summary_latency_s: float = 0.0
    rate_limit_wait_s: float = 0.0
    stop_reason: str | None = None
    model_id: str
    ts: str


class RunRecord(BaseModel):
    type: Literal["run"] = "run"
    run_id: str
    experiment: str
    phase: str = "unspecified"
    cell_id: str = ""
    config_hash: str = ""
    seed: int = 0
    task: str = "pointer_chase"
    depth: int = 0
    file_tokens: int = 0
    difficulty: str = ""
    t0_pad: int = 0
    arm: str
    f: float = 0.0
    d: float | None = None
    R: int = 0
    summ_every: int | None = None
    max_steps: int = 0
    success: bool
    termination: Literal[TERMINATIONS]  # type: ignore[valid-type]
    steps: int
    t0_tokens: int | None = None
    tokens: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    wall_s: float = 0.0
    rate_limit_wait_s: float = 0.0
    model_id: str = ""
    client_kind: str = ""
    harness_git_sha: str = ""
    started_at: str = ""
    flags: list[str] = Field(default_factory=list)
    extra: dict = Field(default_factory=dict)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


@functools.lru_cache(maxsize=1)
def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - not a git checkout
        return "unknown"


class RunLogger:
    def __init__(self, root: str | Path, experiment: str):
        self.dir = Path(root) / experiment
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.jsonl"

    def exists(self, run_id: str) -> bool:
        return self.path(run_id).exists()

    def write(self, steps: list[StepRecord], run: RunRecord) -> Path:
        p = self.path(run.run_id)
        tmp = p.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for s in steps:
                fh.write(s.model_dump_json() + "\n")
            fh.write(run.model_dump_json() + "\n")
        tmp.replace(p)
        return p


def read_logs(paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load step and run records from JSONL files/directories into DataFrames."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(p.rglob("*.jsonl")) if p.is_dir() else [p])
    steps, runs = [], []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                (steps if rec["type"] == "step" else runs).append(rec)
    sdf = pd.json_normalize(steps) if steps else pd.DataFrame()
    rdf = pd.json_normalize(runs) if runs else pd.DataFrame()
    return sdf, rdf
