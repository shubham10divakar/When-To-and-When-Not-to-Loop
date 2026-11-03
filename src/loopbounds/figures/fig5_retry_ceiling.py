"""Figure 5 — retry ceiling: success vs R for each detectability d. L1 curves,
L2 points (hollow), L3 observed (filled, Wilson CI); dashed asymptotes R → ∞."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from loopbounds.analysis.stats import wilson
from loopbounds.figures.common import SERIES, FigContext, note_missing, ref_params
from loopbounds.model import reliability


def make(ctx: FigContext, f: float = 0.2, ds=(0.2, 0.5, 0.9), depth: int = 10):
    P = ref_params(ctx, 1000, depth).with_(max_steps=3 * depth, f_inject=f)
    runs = ctx.runs("E7_retry")
    pred = ctx.csv("predictions_E7_retry.csv")
    R_grid = np.arange(0, 9)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for i, d in enumerate(ds):
        c = SERIES[i]
        y = [reliability.solve(P.with_(d_detect=d, retries=int(R)), "naive").p_success for R in R_grid]
        ax.plot(R_grid, y, color=c, label=f"d = {d:g}")
        ax.axhline(reliability.solve(P.with_(d_detect=d, retries=200), "naive").p_success, color=c,
                   linestyle="--", linewidth=1)
        if not pred.empty and "L2_p_success" in pred:
            pp = pred[np.isclose(pred["d"], d)]
            ax.scatter(pp["R"], pp["L2_p_success"], facecolor="none", edgecolor=c, s=28, linewidth=1.2)
        if not runs.empty:
            g = runs[np.isclose(runs["d"].astype(float), d)]
            agg = g.groupby("R")["success"].agg(["sum", "count"]).reset_index()
            p, lo, hi = wilson(agg["sum"], agg["count"])
            ax.errorbar(agg["R"] + 0.08 * (i - 1), p, yerr=[np.maximum(p - lo, 0), np.maximum(hi - p, 0)], fmt="o", color=c, markersize=4,
                        capsize=2, linewidth=1)
    ax.plot([], [], "k--", linewidth=1, label="ceiling (R → ∞)")
    ax.scatter([], [], facecolor="none", edgecolor="k", s=28, label="L2 simulation")
    ax.errorbar([], [], yerr=[], fmt="o", color="k", markersize=4, label="observed (95% CI)")
    if runs.empty:
        note_missing(ax, "E7")
    ax.set_xlabel("tool-layer retry limit R")
    ax.set_ylabel("P(success)")
    ax.set_title(f"Retries cannot fix silent failures (f = {f:g}, n = {depth})", loc="left")
    ax.legend(loc="lower right", ncol=2)
    return ctx.save(fig, "fig5_retry_ceiling")
