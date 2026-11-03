"""Agreement tests (design doc §7.3).

* L1 ↔ L2 (math check, simulation only)
* L1/L2 ↔ L3 (assumption check) — every failed check is reported with the
  assumption it implicates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from loopbounds.analysis import stats
from loopbounds.analysis.fit import PROCESSED
from loopbounds.model import cost, latency, reliability, retry
from loopbounds.params import REPO_ROOT, Params
from loopbounds.sim import run_sim, summarize

THRESHOLDS_PATH = REPO_ROOT / "preregistration" / "thresholds.yaml"

DIAGNOSES = {
    "cost": "token-growth assumption (constant δ per step, constant a) or cache accounting (κ, ω, min cacheable length)",
    "success": "Markov assumptions (stationary p, r, e; no memory beyond ON/DER) or the coupling form p(T)",
    "markov_vs_naive": "recovery dynamics: if naive pⁿ wins, derailments are rarely recovered (r ≈ 0)",
    "retry": "independence of injected failures or the tool-layer retry policy",
    "feasibility": "one or more of the constraint models near the boundary",
    "latency": "TTFT linear in new/cached tokens, constant decode speed",
}


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


# ---------------------------------------------------------------------------
# L1 <-> L2
# ---------------------------------------------------------------------------

def l1_l2_table(configs: list[tuple[Params, str]], n_runs: int = 100_000, seed: int = 0,
                thresholds: dict | None = None) -> pd.DataFrame:
    th = (thresholds or load_thresholds())["l1_l2"]
    rows = []
    for i, (P, arm) in enumerate(configs):
        res = reliability.solve(P, arm)
        c1 = cost.expected_cost(arm, P, res.end_dist)
        df = run_sim(P, arm, n_runs, seed=seed + i)
        s = summarize(df)
        base = {"arm": arm, "depth": P.depth, "p": P.p, "f": P.f_inject, "d": P.d_detect, "R": P.retries}
        rows.append({**base, "metric": "cost", "L1": c1, "L2": s["cost_mean"], "L2_se": s["cost_se"],
                     "rel_err": abs(s["cost_mean"] - c1) / c1,
                     "pass": abs(s["cost_mean"] - c1) / c1 < th["cost_rel_err"]})
        rows.append({**base, "metric": "p_success", "L1": res.p_success, "L2": s["p_success"],
                     "L2_se": s["p_success_se"], "rel_err": np.nan,
                     "pass": abs(s["p_success"] - res.p_success) <= th["success_se_mult"] * s["p_success_se"] + 1e-12})
        if P.f_inject > 0:
            # clean-delivery rate per *injected* read: E[attempts] check via tool_attempts is arm-agnostic
            dl = retry.delivery(P.f_inject, P.d_detect, P.retries)
            rows.append({**base, "metric": "p_clean_closed_form", "L1": dl.clean, "L2": np.nan, "L2_se": np.nan,
                         "rel_err": np.nan, "pass": True})
        lat1 = latency.expected_latency(arm, P, res)
        rows.append({**base, "metric": "latency_mean", "L1": lat1, "L2": s["latency_mean"], "L2_se": np.nan,
                     "rel_err": abs(s["latency_mean"] - lat1) / lat1, "pass": True})
    return pd.DataFrame(rows)


def retry_delivery_mc(f: float, d: float, R: int, n: int = 100_000, seed: int = 0,
                      thresholds: dict | None = None) -> dict:
    """Monte Carlo of the tool-layer retry policy vs the closed form (Eq. 9-12)."""
    from loopbounds.harness.injection import Injector

    th = (thresholds or load_thresholds())["l1_l2"]
    inj = Injector(f, d, R, seed=seed)
    outs = np.array([inj.draw().outcome == "clean" for _ in range(n)], float)
    p = retry.p_clean(f, d, R)
    se = float(np.sqrt(p * (1 - p) / n))
    return {"f": f, "d": d, "R": R, "closed_form": p, "mc": float(outs.mean()), "se": se,
            "pass": abs(outs.mean() - p) <= th["retry_se_mult"] * se + 1e-12}


# ---------------------------------------------------------------------------
# L1/L2 <-> L3
# ---------------------------------------------------------------------------

def observed_cells(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in runs.groupby("cell_id"):
        k, n = int(g["success"].sum()), len(g)
        p, lo, hi = stats.wilson(k, n)
        rows.append({"cell_id": cid, "n_runs": n, "obs_success": float(p), "obs_lo": float(lo), "obs_hi": float(hi),
                     "obs_cost_mean": float(g["cost_usd"].mean()), "obs_cost_median": float(g["cost_usd"].median()),
                     "obs_wall_mean": float(g["wall_s"].mean()), "obs_wall_p95": float(g["wall_s"].quantile(0.95)),
                     "obs_overflow_rate": float((g["termination"] == "context_overflow").mean()),
                     "obs_steps_mean": float(g["steps"].mean())})
    return pd.DataFrame(rows)


def l3_cell_table(pred: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    obs = observed_cells(runs)
    t = pred.merge(obs, on="cell_id", how="inner")
    t["success_in_ci"] = (t["L1_p_success"] >= t["obs_lo"]) & (t["L1_p_success"] <= t["obs_hi"])
    t["cost_rel_err"] = (t["obs_cost_mean"] - t["L1_cost"]).abs() / t["L1_cost"]
    if "L2_p_success" in t:
        t["L2_success_in_ci"] = (t["L2_p_success"] >= t["obs_lo"]) & (t["L2_p_success"] <= t["obs_hi"])
    return t


def markov_vs_naive(pred: pd.DataFrame, runs: pd.DataFrame) -> dict:
    m = runs[["cell_id", "success"]].merge(pred[["cell_id", "L1_p_success", "L1_p_naive"]], on="cell_id")
    if m.empty:
        return {}
    y = m["success"].astype(float)
    return {"brier_markov": stats.brier(y, m["L1_p_success"]), "brier_naive": stats.brier(y, m["L1_p_naive"]),
            "logloss_markov": stats.log_loss(y, m["L1_p_success"]), "logloss_naive": stats.log_loss(y, m["L1_p_naive"]),
            "n_runs": int(len(m))}


def retry_ceiling(steps: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per (f, d, R): observed clean-delivery rate of true-chain reads vs closed form & ceiling."""
    s = steps[steps["inject.outcome"].isin(["clean", "silent", "exhausted"])]
    rows = []
    for (f, d, R), g in s.groupby(["f", "d", "R"]):
        if f == 0:
            continue
        k, n = int((g["inject.outcome"] == "clean").sum()), len(g)
        p, lo, hi = stats.wilson(k, n)
        rows.append({"f": f, "d": d, "R": R, "n_reads": n, "obs_clean": float(p), "lo": float(lo), "hi": float(hi),
                     "pred_clean": retry.p_clean(f, d, int(R)), "ceiling": retry.clean_ceiling(f, d)})
    t = pd.DataFrame(rows)
    if not t.empty:
        t["pred_in_ci"] = (t["pred_clean"] >= t["lo"]) & (t["pred_clean"] <= t["hi"])
        top = t.loc[t.groupby(["f", "d"])["R"].idxmax()]
        t["plateau_in_ci"] = False
        t.loc[top.index, "plateau_in_ci"] = ((top["ceiling"] >= top["lo"] - 0.02) & (top["ceiling"] <= top["hi"] + 0.02))
    return t


def feasibility_accuracy(pred: pd.DataFrame, runs: pd.DataFrame, constraints: dict) -> dict:
    obs = observed_cells(runs)
    t = pred.merge(obs, on="cell_id")
    if t.empty or "pred_feasible" not in t:
        return {}
    ok = ((t["obs_success"] >= constraints["s_min"]) & (t["obs_cost_mean"] <= constraints["budget"])
          & (t["obs_wall_p95"] <= constraints["slo"]) & (t["obs_overflow_rate"] == 0))
    t["obs_feasible"] = ok
    t["correct"] = t["obs_feasible"] == t["pred_feasible"]
    res = {"accuracy": float(t["correct"].mean()), "n": int(len(t)),
           "confusion": pd.crosstab(t["pred_feasible"], t["obs_feasible"]).to_dict()}
    if "near_boundary" in t and t["near_boundary"].notna().any():
        nb = t["near_boundary"].astype(bool)
        res["accuracy_near"] = float(t.loc[nb, "correct"].mean()) if nb.any() else np.nan
        res["accuracy_far"] = float(t.loc[~nb, "correct"].mean()) if (~nb).any() else np.nan
    res["table"] = t
    return res


def latency_fit(table: pd.DataFrame) -> dict:
    if table.empty:
        return {}
    y, yhat = table["obs_wall_mean"].to_numpy(float), table["L1_latency"].to_numpy(float)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "median_rel_resid": float(np.median(np.abs(y - yhat) / yhat))}


def summarize_checks(cell_table: pd.DataFrame, mvn: dict, retry_t: pd.DataFrame, feas: dict, lat: dict,
                     thresholds: dict | None = None) -> dict:
    th = (thresholds or load_thresholds())["l3"]
    checks = {}
    if not cell_table.empty:
        agents = cell_table[cell_table["arm"] != "single_shot"]
        v = float(agents["cost_rel_err"].median())
        checks["cost"] = {"value": v, "threshold": th["cost_median_rel_err"], "pass": v < th["cost_median_rel_err"]}
        v = float(agents["success_in_ci"].mean())
        checks["success"] = {"value": v, "threshold": th["success_ci_coverage"], "pass": v >= th["success_ci_coverage"]}
    if mvn:
        checks["markov_vs_naive"] = {"value": mvn["brier_markov"] - mvn["brier_naive"], "threshold": 0.0,
                                     "pass": mvn["brier_markov"] < mvn["brier_naive"]}
    if retry_t is not None and not retry_t.empty:
        top = retry_t[retry_t["plateau_in_ci"]]
        checks["retry"] = {"value": float(retry_t["pred_in_ci"].mean()), "threshold": None,
                           "pass": bool(len(top) > 0 and retry_t.groupby(["f", "d"])["plateau_in_ci"].any().all())}
    if feas:
        checks["feasibility"] = {"value": feas["accuracy"], "threshold": th["feasibility_accuracy"],
                                 "pass": feas["accuracy"] >= th["feasibility_accuracy"],
                                 "near": feas.get("accuracy_near"), "far": feas.get("accuracy_far")}
    if lat:
        checks["latency"] = {"value": lat["r2"], "threshold": None, "pass": None}
    for k, c in checks.items():
        if c.get("pass") is False:
            c["diagnosis"] = DIAGNOSES.get(k, "")
    return checks


def write_report(checks: dict, out: Path = PROCESSED, name: str = "agreement_summary.json") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(json.dumps(checks, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return p
