"""Tables 1–4 as LaTeX (``paper/tables``), from processed data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from loopbounds.figures.common import FigContext
from loopbounds.params import REPO_ROOT


def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        a = abs(v)
        return f"{v:.2e}" if (a and (a < 1e-3 or a >= 1e5)) else f"{v:.{digits}g}"
    return str(v).replace("_", r"\_")


def latex_table(rows: list[list], header: list[str], caption: str, label: str) -> str:
    cols = "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(_fmt(v) for v in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table1(ctx: FigContext) -> str:
    rep, P = ctx.report, ctx.params
    mk = rep.get("markov", {})
    tk = rep.get("tokens", {})
    cp = rep.get("coupling", {})
    lt = rep.get("latency", {})
    sm = rep.get("summary", {})
    ci = lambda k: "[{:.3f}, {:.3f}]".format(*mk["wilson"][k]) if mk.get("wilson") else "--"  # noqa: E731
    rows = [
        ["p (advance | on-track)", P.p, ci("p")],
        ["r (recover | derailed)", P.r, ci("r")],
        ["e (wrong answer | derailed)", P.e, ci("e")],
        ["T0 (tokens)", tk.get("t0", P.t0), "IQR [{:.0f}, {:.0f}]".format(*tk["t0_iqr"]) if "t0_iqr" in tk else "--"],
        ["δ0 + δ1·file\\_tokens", f"{tk.get('delta0', 0):.0f} + {tk.get('delta1', 1):.3f}·ft", "--"],
        ["a (output tokens/step)", tk.get("a_out", P.a_out),
         "IQR [{:.0f}, {:.0f}]".format(*tk["a_out_iqr"]) if "a_out_iqr" in tk else "--"],
        ["λ (per token)", cp.get("exp", {}).get("lam", P.lam),
         f"± {cp['exp'].get('lam_se', float('nan')):.2e}" if cp else "--"],
        ["coupling form", rep.get("coupling_enabled", P.coupling), "--"],
        ["α0 (s)", lt.get("ttft_a0", P.ttft_a0), f"R²={lt.get('ttft_r2', float('nan')):.2f}" if lt else "--"],
        ["α1 (s/new token)", lt.get("ttft_a1", P.ttft_a1), "--"],
        ["α2 (s/cached token)", lt.get("ttft_a2", P.ttft_a2), "--"],
        ["τ (tokens/s)", lt.get("tps", P.tps), "--"],
        ["γ (summary ratio)", sm.get("gamma", P.gamma), f"± {sm['gamma_se']:.3f}" if sm else "--"],
        ["κ, ω (cache prices)", f"{P.kappa:g}, {P.omega:g}", "prices.yaml"],
    ]
    return latex_table(rows, ["parameter", "estimate", "95% CI / spread"], "Fitted parameters (calibration phase).",
                       "tab:params")


def table2(ctx: FigContext) -> str:
    t = ctx.csv("agreement_l3_cells.csv")
    if t.empty:
        return "% Table 2: no agreement data yet\n"
    rows = []
    for _, r in t.sort_values(["experiment", "arm", "depth"]).iterrows():
        rows.append([r["experiment"], r["arm"], int(r["depth"]), int(r["file_tokens"]), r["L1_p_success"],
                     r.get("L2_p_success", np.nan), f"{r['obs_success']:.2f} [{r['obs_lo']:.2f}, {r['obs_hi']:.2f}]",
                     r["L1_cost"], r["obs_cost_mean"], r["cost_rel_err"]])
    return latex_table(rows, ["exp.", "arm", "n", "file", "L1 P(s)", "L2 P(s)", "obs. P(s) [95% CI]", "L1 cost",
                              "obs. cost", "cost err"], "Prediction errors per held-out cell.", "tab:errors")


def table3(ctx: FigContext) -> str:
    e9 = ctx.csv("e9_table.csv")
    if e9.empty:
        return "% Table 3: no E9 data yet\n"
    cm = pd.crosstab(e9["pred_feasible"].astype(bool), e9["obs_feasible"].astype(bool)).reindex(
        index=[True, False], columns=[True, False], fill_value=0)
    rows = [["predicted feasible", cm.loc[True, True], cm.loc[True, False]],
            ["predicted infeasible", cm.loc[False, True], cm.loc[False, False]]]
    acc = e9["correct"].mean()
    return latex_table(rows, ["", "observed feasible", "observed infeasible"],
                       f"E9 feasible-region classification (accuracy {acc:.0%}, n={len(e9)}).", "tab:e9")


def table4(ctx: FigContext) -> str:
    runs, steps = ctx.runs("E10_coding"), ctx.steps("E10_coding")
    if runs.empty:
        return "% Table 4: no E10 data yet\n"
    rows = []
    for arm, g in runs.groupby("arm"):
        s = steps[steps["arm"] == arm].sort_values(["run_id", "step"]) if not steps.empty else steps
        d = s.groupby("run_id")["context_tokens"].diff().dropna() if not s.empty else pd.Series(dtype=float)
        rows.append([arm, len(g), g["success"].mean(), g["steps"].mean(), d.median() if len(d) else np.nan,
                     g["cost_usd"].mean()])
    return latex_table(rows, ["arm", "runs", "success", "mean steps", "median δ", "mean cost"],
                       "External validity: coding tasks (E10).", "tab:e10")


def make(ctx: FigContext, out: Path | None = None) -> list[Path]:
    out = out or REPO_ROOT / "paper" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in (("tab1_params", table1), ("tab2_prediction_errors", table2), ("tab3_e9_confusion", table3),
                     ("tab4_e10_external", table4)):
        p = out / f"{name}.tex"
        p.write_text(fn(ctx), encoding="utf-8")
        paths.append(p)
    return paths
