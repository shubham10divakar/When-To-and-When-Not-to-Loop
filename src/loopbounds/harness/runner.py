"""Experiment runner: config → cells → runs → JSONL logs (design doc §6).

Experiment YAML schema (all keys optional except ``name``)::

    name: E1
    phase: calibration | prediction | validation | pilot | external
    client: {kind: anthropic | perfect_mock | markov_mock | coupled_mock, ...}
    mode: agent | single_shot
    task: {difficulty: L1, file_tokens: 1000, n_decoy_chains: 2, n_fake_clues: 2, key: K7, t0_pad: 0}
    injection: {f: 0.0, d: 1.0, R: 0}
    grid: {depth: [5, 10], arm: [naive, cached], ...}   # cartesian product
    cells: [{depth: 20, arm: naive, ...}, ...]          # explicit list (overrides grid)
    runs_per_cell: 10
    max_steps_factor: 3
    summ_every: 5
    window_cap: null          # artificial context window (tokens), see design doc §12.2
    max_tokens: 1024
    max_spend_usd: 25
    seed: 0

Task seeds depend only on the task-defining keys and the replicate index, so
different arms (and the single-shot baseline) see *the same task instances*.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from loopbounds.harness import taskgen
from loopbounds.harness.agent import PointerChaseEnv, run_agent
from loopbounds.harness.arms import make_arm
from loopbounds.harness.budget import BudgetExceeded, SpendMeter, estimate_experiment_cost, usage_dollars
from loopbounds.harness.injection import Injector
from loopbounds.harness.llm import make_client
from loopbounds.harness.logging import RunLogger, RunRecord, git_sha, now_iso
from loopbounds.harness.single_shot import run_single_shot
from loopbounds.harness.tokens import count_json_tokens, count_tokens
from loopbounds.harness.tools import TOOL_SCHEMAS
from loopbounds.params import CONFIG_DIR, REPO_ROOT, Params, load_params, load_prices

RAW_DIR = REPO_ROOT / "data" / "raw"
TASK_KEYS = ("depth", "file_tokens", "difficulty", "n_decoy_chains", "n_fake_clues", "key", "t0_pad")
DEFAULT_TASK = {"depth": 10, "file_tokens": 1000, "difficulty": "L1", "n_decoy_chains": 2, "n_fake_clues": 2,
                "key": "K7", "t0_pad": 0}


def _h(obj: Any, n: int = 10) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def load_config(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        return dict(path_or_dict)
    p = Path(path_or_dict)
    if not p.exists():
        cand = CONFIG_DIR / "experiments" / p.name
        p = cand if cand.exists() else CONFIG_DIR / "experiments" / f"{p.name}.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg.setdefault("name", p.stem)
    return cfg


def expand_cells(cfg: dict) -> list[dict]:
    base = {**DEFAULT_TASK, **(cfg.get("task") or {})}
    inj = {"f": 0.0, "d": 1.0, "R": 0, **(cfg.get("injection") or {})}
    base.update(inj)
    base.update({"arm": cfg.get("arm", "naive"), "mode": cfg.get("mode", "agent"),
                 "summ_every": cfg.get("summ_every", 5), "window_cap": cfg.get("window_cap"),
                 "max_steps_factor": cfg.get("max_steps_factor", 3)})
    if cfg.get("cells_file") and not cfg.get("cells"):
        cf = Path(cfg["cells_file"])
        cf = cf if cf.is_absolute() else CONFIG_DIR / "experiments" / cf
        cfg = {**cfg, "cells": yaml.safe_load(cf.read_text())["cells"]}
    if cfg.get("cells"):
        cells = [{**base, **c} for c in cfg["cells"]]
    else:
        grid = cfg.get("grid") or {}
        keys = list(grid)
        cells = [{**base, **dict(zip(keys, vals))} for vals in itertools.product(*(grid[k] for k in keys))]
    for c in cells:
        if c["mode"] == "single_shot":
            c["arm"] = "single_shot"
        c["cell_id"] = _h({k: v for k, v in c.items() if k != "cell_id"})
    return cells


def price_params(model_id: str) -> Params:
    """Params carrying the price table entry for this model (fallback: default model)."""
    import yaml as _y

    table = _y.safe_load((CONFIG_DIR / "prices.yaml").read_text())["models"]
    key = model_id if model_id in table else next((k for k in table if model_id.startswith(k)), None)
    return Params().with_(**{k: v for k, v in load_prices(key).items() if k != "window"})


def estimate_params(cell: dict, client_spec: dict, base: Params | None = None) -> Params:
    """L1 parameters for budgeting a cell before any data exists."""
    base = base or load_params(model=None)
    task = taskgen.generate(taskgen.TaskConfig(**{k: cell[k] for k in TASK_KEYS}, seed=0))
    t0 = count_tokens(taskgen.system_prompt(task)) + count_tokens(taskgen.task_prompt(task)) + \
        count_json_tokens(TOOL_SCHEMAS)
    model = client_spec.get("model") if client_spec.get("kind") == "anthropic" else "mock"
    pp = price_params(model or "mock")
    depth = int(cell["depth"])
    return base.with_(depth=depth, max_steps=int(cell["max_steps_factor"]) * depth, t0=float(t0),
                      delta=float(base.a_out + cell["file_tokens"] + 20), file_tokens=float(cell["file_tokens"]),
                      n_decoy_chains=int(cell["n_decoy_chains"]), summ_every=int(cell["summ_every"]),
                      f_inject=float(cell["f"]), d_detect=float(cell["d"]), retries=int(cell["R"]),
                      price_in=pp.price_in, price_out=pp.price_out, kappa=pp.kappa, omega=pp.omega,
                      window=int(cell["window_cap"] or base.window))


@dataclass
class ExperimentSummary:
    name: str
    n_runs: int
    n_skipped: int
    spent_usd: float
    estimate_usd: float
    aborted: bool
    out_dir: str


def run_experiment(cfg_or_path, *, out_root: str | Path = RAW_DIR, resume: bool = True,
                   runs_per_cell: int | None = None, dry_run: bool = False, verbose: bool = False,
                   client=None) -> ExperimentSummary:
    cfg = load_config(cfg_or_path)
    name = cfg["name"]
    client_spec = dict(cfg.get("client") or {"kind": "perfect_mock"})
    cells = expand_cells(cfg)
    reps = int(runs_per_cell if runs_per_cell is not None else cfg.get("runs_per_cell", 10))
    seed0 = int(cfg.get("seed", 0))
    is_real = client_spec.get("kind") == "anthropic"

    est = estimate_experiment_cost([(estimate_params(c, client_spec), c["arm"], reps) for c in cells])
    cap = cfg.get("max_spend_usd")
    if is_real and cap is not None and est > float(cap):
        raise BudgetExceeded(f"{name}: L1 cost estimate ${est:.2f} exceeds max_spend_usd ${float(cap):.2f}")
    logger = RunLogger(out_root, name)
    if dry_run:
        return ExperimentSummary(name, len(cells) * reps, 0, 0.0, est, False, str(logger.dir))

    llm = client or make_client(client_spec)
    model_for_price = client_spec.get("model", "claude-haiku-5-5") if is_real else "mock"
    pp = price_params(model_for_price)
    meter = SpendMeter(pp, float(cap) if (cap is not None and is_real) else None)
    n_done = n_skip = 0
    aborted = False
    max_tokens = int(cfg.get("max_tokens", 1024))

    base_keys = set(TASK_KEYS) | {"f", "d", "R", "arm", "mode", "summ_every", "window_cap", "max_steps_factor",
                                  "cell_id"}
    try:
        for cell in cells:
            for rep in range(reps):
                task_key = {k: cell[k] for k in TASK_KEYS}
                task_seed = int(_h([seed0, task_key, rep], 8), 16)
                run_id = f"{name}-{cell['cell_id']}-{rep:03d}"
                if resume and logger.exists(run_id):
                    n_skip += 1
                    continue
                task = taskgen.generate(taskgen.TaskConfig(**task_key, seed=task_seed))
                rec_common = dict(
                    run_id=run_id, experiment=name, phase=cfg.get("phase", "unspecified"),
                    cell_id=cell["cell_id"], config_hash=_h(cfg), seed=task_seed, depth=int(cell["depth"]),
                    file_tokens=int(cell["file_tokens"]), difficulty=str(cell["difficulty"]),
                    t0_pad=int(cell["t0_pad"]), arm=cell["arm"], f=float(cell["f"]),
                    d=float(cell["d"]) if cell["f"] else None, R=int(cell["R"]),
                    summ_every=int(cell["summ_every"]) if cell["arm"] == "summarized" else None,
                    client_kind=getattr(llm, "kind", ""), harness_git_sha=git_sha(), started_at=now_iso(),
                    extra={k: cell[k] for k in cell if k not in base_keys},
                )
                if cell["mode"] == "single_shot":
                    window = int(cell["window_cap"] or load_params(model=None).window)
                    ss = run_single_shot(task, llm, window=window, seed=task_seed)
                    meter.add(ss.usage)
                    run = RunRecord(**rec_common, success=ss.success,
                                    termination=("infeasible" if not ss.feasible else
                                                 ("answer_correct" if ss.success else "answer_wrong")),
                                    steps=1 if ss.feasible else 0, t0_tokens=ss.usage.context or None,
                                    tokens={"input_new": ss.usage.input_new, "cache_read": 0, "cache_write": 0,
                                            "output": ss.usage.output, "context_estimate": ss.context_estimate},
                                    cost_usd=usage_dollars(pp, ss.usage), wall_s=ss.total_s,
                                    rate_limit_wait_s=ss.rate_limit_wait_s, model_id=ss.model_id,
                                    max_steps=1)
                    run.extra.update({"ttft_s": ss.ttft_s, "answer": ss.answer})
                    logger.write([], run)
                else:
                    inj = Injector(cell["f"], cell["d"], cell["R"], seed=int(_h([run_id, "inj"], 8), 16))
                    env = PointerChaseEnv(task, inj)
                    arm = make_arm(cell["arm"], int(cell["summ_every"]))
                    M = int(cell["max_steps_factor"]) * int(cell["depth"])
                    flags: list[str] = []
                    try:
                        res = run_agent(env, llm, arm, max_steps=M, run_id=run_id,
                                        window_cap=cell["window_cap"], max_tokens=max_tokens,
                                        spend_hook=meter.add)
                    except BudgetExceeded:
                        aborted = True
                        raise
                    total = res.usage + res.summary_usage
                    if cell["arm"] == "naive" and res.usage.cache_read:
                        flags.append("naive_arm_saw_cache_reads")
                    run = RunRecord(**rec_common, max_steps=M, success=res.success, termination=res.termination,
                                    steps=len(res.steps), t0_tokens=res.t0_tokens,
                                    tokens={"input_new": total.input_new, "cache_read": total.cache_read,
                                            "cache_write": total.cache_write, "output": total.output,
                                            "summary_input": res.summary_usage.input_new,
                                            "summary_output": res.summary_usage.output},
                                    cost_usd=usage_dollars(pp, total), wall_s=res.wall_s,
                                    rate_limit_wait_s=res.rate_limit_wait_s, model_id=res.model_id, flags=flags)
                    logger.write(res.steps, run)
                n_done += 1
                if verbose:
                    print(f"[{name}] {run_id} arm={cell['arm']} n={cell['depth']} -> {run.termination}")
    except BudgetExceeded as exc:
        aborted = True
        (logger.dir / "ABORTED.txt").write_text(f"{now_iso()} {exc}\n")
    return ExperimentSummary(name, n_done, n_skip, meter.spent, est, aborted, str(logger.dir))
