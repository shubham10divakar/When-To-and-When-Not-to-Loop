"""L1/L2 predictions for held-out (prediction-phase) cells (design doc §7.2).

Predictions are generated *before* looking at prediction-phase data: the
script refuses to run if logs for the experiment already exist, unless
``post_hoc=True``, which watermarks every output row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from loopbounds.analysis.fit import PROCESSED, RAW, FitResult
from loopbounds.harness.runner import expand_cells, load_config
from loopbounds.model import baseline, cost, feasible, latency, reliability
from loopbounds.params import CONFIG_DIR, Params
from loopbounds.sim import run_sim, summarize


class PredictionAfterData(RuntimeError):
    pass


def cell_params(fit: FitResult, cell: dict, constraints: dict | None = None) -> Params:
    P = fit.params_for(cell["depth"], cell["file_tokens"], cell.get("t0_pad", 0) or 0)
    upd = dict(max_steps=int(cell["max_steps_factor"]) * int(cell["depth"]), f_inject=float(cell["f"]),
               d_detect=float(cell["d"]), retries=int(cell["R"]), summ_every=int(cell["summ_every"]),
               n_decoy_chains=int(cell["n_decoy_chains"]))
    if cell.get("window_cap"):
        upd["window"] = int(cell["window_cap"])
    return P.with_(**upd, **(constraints or {}))


def predict_cell(P: Params, arm: str, *, n_sim: int = 10_000, seed: int = 0, with_feasibility=False) -> dict:
    if arm == "single_shot":
        ss = baseline.single_shot(P)
        return {"L1_p_success": ss.p_success, "L1_cost": ss.cost, "L1_latency": ss.latency,
                "L1_feasible_ss": ss.feasible, "L1_context": ss.context}
    res = reliability.solve(P, arm)
    row = {
        "L1_p_success": res.p_success,
        "L1_p_naive": float(reliability.p_naive(P.p, P.depth)),
        "L1_cost": cost.expected_cost(arm, P, res.end_dist),
        "L1_latency": latency.expected_latency(arm, P, res),
        "L1_steps": res.expected_calls,
        "L1_p_overflow": res.p_fail["context_overflow"],
    }
    if n_sim:
        s = summarize(run_sim(P, arm, n_sim, seed=seed))
        row.update({"L2_p_success": s["p_success"], "L2_p_success_se": s["p_success_se"],
                    "L2_cost": s["cost_mean"], "L2_cost_se": s["cost_se"], "L2_latency": s["latency_mean"],
                    "L2_latency_p95": s["latency_p95"]})
    if with_feasibility:
        f = feasible.predict_feasible(P, arm)
        row.update({f"pred_{k}": v for k, v in f.items() if k.startswith("ok_") or k in ("feasible",
                                                                                           "agent_preferred")})
    return row


def _has_data(name: str, raw_root: Path) -> bool:
    d = Path(raw_root) / name
    return d.exists() and any(d.glob("*.jsonl"))


def predict_experiment(cfg_or_path, fit: FitResult, *, n_sim: int = 10_000, post_hoc: bool = False,
                       raw_root: Path = RAW, out: Path = PROCESSED, seed: int = 0) -> pd.DataFrame:
    cfg = load_config(cfg_or_path)
    name = cfg["name"]
    if _has_data(name, raw_root) and not post_hoc:
        raise PredictionAfterData(f"{name}: prediction-phase data already exists; use post_hoc=True")
    constraints = cfg.get("constraints") or {}
    rows = []
    for cell in expand_cells(cfg):
        P = cell_params(fit, cell, constraints)
        row = {"experiment": name, "cell_id": cell["cell_id"], "arm": cell["arm"], "depth": cell["depth"],
               "file_tokens": cell["file_tokens"], "t0_pad": cell["t0_pad"], "f": cell["f"], "d": cell["d"],
               "R": cell["R"], "summ_every": cell["summ_every"], "near_boundary": cell.get("near_boundary"),
               "delta": P.delta, "t0": P.t0}
        row.update(predict_cell(P, cell["arm"], n_sim=n_sim if cell["arm"] != "single_shot" else 0,
                                seed=seed, with_feasibility=bool(constraints)))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["post_hoc"] = post_hoc
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"predictions_{name}.csv", index=False)
    return df


def generate_e9_cells(fit: FitResult, *, constraints: dict, arms=("naive", "cached", "summarized"),
                      depths=range(4, 61, 2), file_tokens=(200, 1000, 3000), n_near: int = 10, n_far: int = 10,
                      near_tol: float = 0.15, far_tol: float = 0.4, max_steps_factor: int = 3,
                      summ_every: int = 5, seed: int = 0) -> list[dict]:
    """Sample E9 configs near and far from the predicted feasible-region boundary.

    The signed margin of a config is the smallest normalised slack across the four
    constraints (negative = predicted infeasible).
    """
    rng = np.random.default_rng(seed)
    cands = []
    for arm in arms:
        for n in depths:
            for ft in file_tokens:
                cell = {"depth": n, "file_tokens": ft, "t0_pad": 0, "max_steps_factor": max_steps_factor,
                        "f": 0.0, "d": 1.0, "R": 0, "summ_every": summ_every, "n_decoy_chains": 2}
                P = cell_params(fit, cell, constraints)
                c = feasible.evaluate_cell(P, arm)
                slack = [(c.p_success - P.s_min) / P.s_min, (P.budget - c.cost) / P.budget,
                         (P.slo - c.latency) / P.slo, (P.window - c.context) / P.window]
                cands.append({"arm": arm, "depth": n, "file_tokens": ft, "margin": float(min(slack))})
    df = pd.DataFrame(cands)
    near = df[df["margin"].abs() < near_tol]
    far = df[df["margin"].abs() > far_tol]
    pick_near = near.sample(min(n_near, len(near)), random_state=int(rng.integers(1 << 31)))
    # balance far points between predicted-feasible and predicted-infeasible
    far_pos, far_neg = far[far["margin"] > 0], far[far["margin"] < 0]
    k = n_far // 2
    pick_far = pd.concat([far_pos.sample(min(k, len(far_pos)), random_state=1),
                          far_neg.sample(min(n_far - k, len(far_neg)), random_state=2)])
    out = []
    for flag, part in ((True, pick_near), (False, pick_far)):
        for _, r in part.iterrows():
            out.append({"arm": r["arm"], "depth": int(r["depth"]), "file_tokens": int(r["file_tokens"]),
                        "near_boundary": flag, "predicted_margin": round(float(r["margin"]), 4)})
    return out


def write_e9_cells(cells: list[dict], path: Path | None = None) -> Path:
    path = path or (CONFIG_DIR / "experiments" / "E9_cells.generated.yaml")
    path.write_text(yaml.safe_dump({"cells": cells}, sort_keys=False))
    return path
