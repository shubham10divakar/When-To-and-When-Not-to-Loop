"""Parameter estimation from run logs (design doc §7.1).

Output: ``data/processed/params_fitted.yaml`` (point estimates loadable as
:class:`Params`) plus ``data/processed/fit_report.json`` with CIs and
diagnostics. Token parameters are fitted as functions of the task knobs so
that predictions can be made for unseen cells:

    δ(file_tokens) = δ0 + δ1·file_tokens         (naive-arm consecutive context deltas)
    T0(t0_pad)     = T00 + T01·t0_pad            (context of the first call)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import optimize

from loopbounds.analysis import stats
from loopbounds.harness.logging import read_logs
from loopbounds.params import REPO_ROOT, Params, load_params

PROCESSED = REPO_ROOT / "data" / "processed"
RAW = REPO_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_experiments(names, raw_root: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [Path(raw_root) / n for n in names if (Path(raw_root) / n).exists()]
    if not paths:
        return pd.DataFrame(), pd.DataFrame()
    steps, runs = read_logs(paths)
    if not runs.empty and not steps.empty:
        meta = runs[["run_id", "experiment", "cell_id", "depth", "file_tokens", "difficulty", "t0_pad", "f", "d", "R"]]
        steps = steps.merge(meta, on="run_id", how="left")
    return steps, runs


def process(names, raw_root: Path = RAW, out: Path = PROCESSED) -> dict[str, Path]:
    """Write per-experiment parquet files (released with the paper)."""
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for n in names:
        s, r = load_experiments([n], raw_root)
        if r.empty:
            continue
        r.to_parquet(out / f"{n}_runs.parquet", index=False)
        if not s.empty:
            s.drop(columns=[c for c in s.columns if c.startswith("inject.types")], errors="ignore") \
             .to_parquet(out / f"{n}_steps.parquet", index=False)
        written[n] = out / f"{n}_runs.parquet"
    return written


# ---------------------------------------------------------------------------
# Token parameters
# ---------------------------------------------------------------------------

def fit_tokens(steps: pd.DataFrame) -> dict:
    s = steps.copy()
    first = s[s["step"] == 0]
    out: dict = {}
    if not first.empty:
        X = np.column_stack([np.ones(len(first)), first["t0_pad"].fillna(0).to_numpy(float)])
        if first["t0_pad"].nunique() > 1:
            fit = stats.ols_hc3(X, first["context_tokens"].to_numpy(float))
            out["t0"], out["t0_per_pad"] = float(fit["coef"][0]), float(fit["coef"][1])
        else:
            out["t0"], out["t0_per_pad"] = float(first["context_tokens"].median()), 1.0
        out["t0_iqr"] = [float(first["context_tokens"].quantile(q)) for q in (0.25, 0.75)]
    main = s[s["summary_input_tokens"] == 0]
    out["a_out"] = float(main["output_tokens"].median())
    out["a_out_iqr"] = [float(main["output_tokens"].quantile(q)) for q in (0.25, 0.75)]
    # δ from consecutive context deltas within runs (no summary in between), read_file steps only
    s = s.sort_values(["run_id", "step"])
    s["ctx_next"] = s.groupby("run_id")["context_tokens"].shift(-1)
    s["summ_next"] = s.groupby("run_id")["summary_input_tokens"].shift(-1)
    d = s[(s["tool_name"] == "read_file") & s["ctx_next"].notna() & (s["summ_next"] == 0)
          & ~s["tool_arg"].isna()].copy()
    d = d[~d["label"].isin(["ERROR"])]
    d["delta"] = d["ctx_next"] - d["context_tokens"]
    if not d.empty:
        if d["file_tokens"].nunique() > 1:
            fit = stats.ols_hc3(np.column_stack([np.ones(len(d)), d["file_tokens"].to_numpy(float)]),
                                d["delta"].to_numpy(float))
            out["delta0"], out["delta1"] = float(fit["coef"][0]), float(fit["coef"][1])
        else:
            ft = float(d["file_tokens"].iloc[0])
            med = float(d["delta"].median())
            out["delta1"] = med / ft if ft else 0.0
            out["delta0"] = 0.0
        out["delta_by_file_tokens"] = {int(k): float(v) for k, v in d.groupby("file_tokens")["delta"].median().items()}
    return out


def fit_quadratic(runs: pd.DataFrame, params: Params) -> pd.DataFrame:
    """OLS of price-weighted input tokens on [1, m, m²] per arm (m = LLM calls).

    Expected m² coefficient: δ/2 (naive) and κ·δ/2 (cached) in token units.
    """
    rows = []
    for (arm, ft), g in runs[runs["arm"].isin(["naive", "cached"])].groupby(["arm", "file_tokens"]):
        if g["steps"].nunique() < 3:
            continue
        m = g["steps"].to_numpy(float)
        w = (g["tokens.input_new"] + params.kappa * g["tokens.cache_read"]
             + (1 + params.omega) * g["tokens.cache_write"]).to_numpy(float)
        fit = stats.ols_hc3(np.column_stack([np.ones_like(m), m, m**2]), w)
        rows.append({"arm": arm, "file_tokens": ft, "coef_m2": fit["coef"][2], "se_m2": fit["se"][2],
                     "r2": fit["r2"], "n": fit["n"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------

def markov_counts(steps: pd.DataFrame) -> dict:
    c = steps[steps["phase"] == "chase"]
    on, der = c[c["markov_state"] == "ON"], c[c["markov_state"] == "DER"]
    return {"adv_on": int((on["label"] == "ADVANCE").sum()), "n_on": int(len(on)),
            "adv_der": int((der["label"] == "ADVANCE").sum()),
            "wrong_der": int((der["label"] == "ANSWER_WRONG").sum()), "n_der": int(len(der))}


def _pre(cnt: dict) -> np.ndarray:
    p = cnt["adv_on"] / cnt["n_on"] if cnt["n_on"] else np.nan
    r = cnt["adv_der"] / cnt["n_der"] if cnt["n_der"] else np.nan
    e = cnt["wrong_der"] / cnt["n_der"] if cnt["n_der"] else np.nan
    return np.array([p, r, e])


def fit_markov(steps: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """§5.5 estimators with Wilson CIs and a run-level bootstrap."""
    if steps.empty:
        return {}
    cnt = markov_counts(steps)
    est = _pre(cnt)
    wl = {
        "p": stats.wilson(cnt["adv_on"], cnt["n_on"]),
        "r": stats.wilson(cnt["adv_der"], cnt["n_der"]),
        "e": stats.wilson(cnt["wrong_der"], cnt["n_der"]),
    }
    per_run = [markov_counts(g) for _, g in steps.groupby("run_id")]

    def stat(groups):
        tot = {k: sum(g[k] for g in groups) for k in per_run[0]}
        return _pre(tot)

    _, lo, hi = stats.bootstrap(per_run, stat, n_boot=n_boot, seed=seed)
    return {
        "counts": cnt,
        **{k: float(v) for k, v in zip(("p", "r", "e"), est)},
        "wilson": {k: [float(v[1]), float(v[2])] for k, v in wl.items()},
        "bootstrap": {k: [float(a), float(b)] for k, a, b in zip(("p", "r", "e"), lo, hi)},
    }


def fit_coupling(steps: pd.DataFrame) -> dict:
    """MLE of p(T) on chase-phase ON steps; exponential vs linear by AIC."""
    c = steps[(steps["phase"] == "chase") & (steps["markov_state"] == "ON")]
    if len(c) < 20:
        return {}
    y = (c["label"] == "ADVANCE").to_numpy(float)
    T = c["context_tokens"].to_numpy(float)
    scale = max(T.max(), 1.0)
    x = T / scale
    eps = 1e-9

    def nll_exp(th):
        p0, lam = th
        p = np.clip(p0 * np.exp(-lam * x), eps, 1 - eps)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def nll_lin(th):
        p0, lam = th
        p = np.clip(p0 - lam * x, eps, 1 - eps)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    p_init = min(max(y.mean(), 0.05), 0.999)
    res_e = optimize.minimize(nll_exp, [p_init, 0.01], bounds=[(1e-3, 1.0), (0.0, 50.0)], method="L-BFGS-B")
    res_l = optimize.minimize(nll_lin, [p_init, 0.01], bounds=[(1e-3, 1.0), (0.0, 1.0)], method="L-BFGS-B")
    out = {
        "exp": {"p0": float(res_e.x[0]), "lam": float(res_e.x[1] / scale), "aic": stats.aic(-res_e.fun, 2)},
        "linear": {"p0": float(res_l.x[0]), "lam": float(res_l.x[1] / scale), "aic": stats.aic(-res_l.fun, 2)},
        "n": int(len(y)),
    }
    out["best"] = "exp" if out["exp"]["aic"] <= out["linear"]["aic"] else "linear"
    # profile-likelihood-free CI for λ via observed information (exp form)
    try:
        hinv = res_e.hess_inv.todense()
        out["exp"]["lam_se"] = float(np.sqrt(max(hinv[1, 1], 0)) / scale)
    except Exception:  # noqa: BLE001
        out["exp"]["lam_se"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Latency / summarization / single-shot
# ---------------------------------------------------------------------------

def fit_latency(steps: pd.DataFrame) -> dict:
    s = steps[(steps["ttft_s"] > 0) & (steps["llm_total_s"] > steps["ttft_s"])]
    if len(s) < 10:
        return {}
    t_new = (s["input_tokens_new"] + s["cache_write_tokens"]).to_numpy(float)
    t_cached = s["cache_read_tokens"].to_numpy(float)
    X = np.column_stack([np.ones(len(s)), t_new, t_cached])
    if np.allclose(t_cached, 0):
        X = X[:, :2]
    fit = stats.ols_hc3(X, s["ttft_s"].to_numpy(float))
    coef = list(fit["coef"]) + ([0.0] if X.shape[1] == 2 else [])
    tps = float(np.median(s["output_tokens"] / (s["llm_total_s"] - s["ttft_s"])))
    resid = np.log(s["llm_total_s"]) - np.log(coef[0] + coef[1] * t_new + coef[2] * t_cached
                                               + s["output_tokens"] / tps).to_numpy()
    return {"ttft_a0": float(coef[0]), "ttft_a1": float(coef[1]), "ttft_a2": float(coef[2]),
            "ttft_r2": float(fit["r2"]), "tps": tps, "llm_lat_sigma": float(np.nanstd(resid)),
            "tool_lat": float(steps["tool_latency_s"].mean()),
            "tool_lat_samples_n": int(steps["tool_latency_s"].notna().sum())}


def fit_summary(steps: pd.DataFrame) -> dict:
    s = steps[steps["summary_input_tokens"] > 0]
    if len(s) < 3:
        return {}
    fit = stats.ols_hc3(np.column_stack([np.ones(len(s)), s["summary_input_tokens"].to_numpy(float)]),
                        s["summary_output_tokens"].to_numpy(float))
    gamma = float(fit["coef"][1])
    t0_summ = float(-fit["coef"][0] / gamma) if gamma > 0 else 0.0
    return {"gamma": gamma, "t0_summ": max(t0_summ, 0.0), "gamma_se": float(fit["se"][1]), "n": int(len(s))}


def fit_single_shot(runs: pd.DataFrame) -> dict:
    ss = runs[(runs["arm"] == "single_shot") & (runs["termination"] != "infeasible")]
    if len(ss) < 10 or ss["success"].nunique() < 2:
        return {}
    T = ss["tokens.input_new"].to_numpy(float) / 1000.0
    X = np.column_stack([np.ones(len(ss)), ss["depth"].to_numpy(float), T])
    fit = stats.logistic_fit(X, ss["success"].to_numpy(float))
    b = fit["coef"]
    return {"ss_b0": float(b[0]), "ss_b1": float(b[1]), "ss_b2": float(b[2]), "se": [float(x) for x in fit["se"]],
            "n": fit["n"], "a_ss": float(ss["tokens.output"].median())}


# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    params: Params
    report: dict = field(default_factory=dict)

    def params_for(self, depth: int, file_tokens: float, t0_pad: float = 0.0, **kw) -> Params:
        tk = self.report.get("tokens", {})
        delta = tk.get("delta0", 0.0) + tk.get("delta1", 1.0) * file_tokens if tk else self.params.delta
        t0 = tk.get("t0", self.params.t0) + tk.get("t0_per_pad", 1.0) * t0_pad
        return self.params.with_(depth=int(depth), delta=float(delta), t0=float(t0),
                                 file_tokens=float(file_tokens), **kw)


def fit_all(cal_experiments, raw_root: Path = RAW, base: Params | None = None, *,
            markov_from=None, coupling_from=None, n_boot: int = 2000) -> FitResult:
    """Fit every parameter from calibration-phase logs.

    ``markov_from`` / ``coupling_from`` restrict the reliability fits to the
    given experiments (default: all calibration runs without injection).
    """
    base = base or load_params()
    steps, runs = load_experiments(cal_experiments, raw_root)
    if runs.empty:
        raise FileNotFoundError(f"no calibration logs found for {cal_experiments} under {raw_root}")
    agent_steps = steps[steps["arm"].isin(["naive", "cached", "summarized"])] if not steps.empty else steps
    clean_steps = agent_steps[(agent_steps["f"].fillna(0) == 0)] if not agent_steps.empty else agent_steps
    report: dict = {"experiments": list(cal_experiments), "n_runs": int(len(runs)), "n_steps": int(len(steps))}
    upd: dict = {}

    if not agent_steps.empty:
        tk = fit_tokens(agent_steps[agent_steps["arm"] == "naive"] if (agent_steps["arm"] == "naive").any()
                        else agent_steps)
        report["tokens"] = tk
        upd.update({k: tk[k] for k in ("t0", "a_out") if k in tk})
        if "delta1" in tk:
            ft = float(base.file_tokens)
            upd["delta"] = tk["delta0"] + tk["delta1"] * ft
        report["quadratic"] = fit_quadratic(runs, base.with_(**upd)).to_dict(orient="records")

        msteps = clean_steps
        if markov_from:
            msteps = clean_steps[clean_steps["experiment"].isin(markov_from)]
        mk = fit_markov(msteps, n_boot=n_boot)
        report["markov"] = mk
        for k in ("p", "r", "e"):
            if k in mk and np.isfinite(mk[k]):
                upd[k] = mk[k]
        if upd.get("r", 0) + upd.get("e", 0) > 1:
            upd["e"] = 1 - upd["r"]

        csteps = clean_steps if not coupling_from else clean_steps[clean_steps["experiment"].isin(coupling_from)]
        cp = fit_coupling(csteps)
        report["coupling"] = cp
        if cp:
            best = cp[cp["best"]]
            upd.update({"p0": best["p0"], "lam": best["lam"]})
            se = cp["exp"].get("lam_se", float("nan"))
            # enable coupling only if λ is distinguishable from zero (pre-registered rule)
            if np.isfinite(se) and cp["exp"]["lam"] > 2 * se:
                upd["coupling"] = cp["best"]
            report["coupling_enabled"] = upd.get("coupling", "none")

        lt = fit_latency(agent_steps)
        report["latency"] = lt
        upd.update({k: lt[k] for k in ("ttft_a0", "ttft_a1", "ttft_a2", "tps", "llm_lat_sigma", "tool_lat")
                    if k in lt and np.isfinite(lt[k])})
        sm = fit_summary(agent_steps)
        report["summary"] = sm
        upd.update({k: sm[k] for k in ("gamma", "t0_summ") if k in sm})

    ss = fit_single_shot(runs)
    report["single_shot"] = ss
    upd.update({k: ss[k] for k in ("ss_b0", "ss_b1", "ss_b2", "a_ss") if k in ss})
    report["model_ids"] = sorted(set(runs["model_id"].dropna()))
    return FitResult(base.with_(**upd), report)


def save_fit(fit: FitResult, out: Path = PROCESSED) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    p_yaml = out / "params_fitted.yaml"
    d = fit.params.to_dict()
    d.pop("meta", None)
    p_yaml.write_text(yaml.safe_dump(d, sort_keys=False))
    p_json = out / "fit_report.json"
    p_json.write_text(json.dumps(fit.report, indent=2, default=float))
    return p_yaml, p_json


def load_fit(out: Path = PROCESSED) -> FitResult:
    p = load_params(out / "params_fitted.yaml")
    rep_path = out / "fit_report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    return FitResult(p, report)
