"""Figure 3 — success vs n: observed (Wilson CIs), naive pⁿ and Markov DP, one
panel per file size; calibration (n ≤ 10) vs prediction region shaded."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from loopbounds.analysis.stats import wilson
from loopbounds.figures.common import GRID, INK, INK2, SERIES, FigContext, note_missing, ref_params
from loopbounds.model import reliability


def make(ctx: FigContext, file_sizes=(200, 1000, 3000), cal_max: int = 10):
    runs = pd.concat([ctx.runs("E3_reliability_cal"), ctx.runs("E4_reliability_pred")], ignore_index=True)
    fig, axes = plt.subplots(1, len(file_sizes), figsize=(10, 3.4), sharey=True)
    n_grid = np.arange(1, 45)
    for ax, ft in zip(np.atleast_1d(axes), file_sizes):
        P = ref_params(ctx, ft)
        markov = [reliability.solve(P.with_(depth=int(n), max_steps=3 * int(n)), "naive").p_success for n in n_grid]
        ax.axvspan(0, cal_max + 0.5, color=GRID, alpha=0.6, linewidth=0)
        ax.plot(n_grid, markov, color=SERIES[0], label="Markov DP (L1)")
        ax.plot(n_grid, reliability.p_naive(P.p, n_grid), color=SERIES[1], linestyle="--", label="naive pⁿ")
        g = runs[(runs["file_tokens"] == ft) & (runs["arm"] == "naive")] if not runs.empty else runs
        if not g.empty:
            agg = g.groupby("depth")["success"].agg(["sum", "count"]).reset_index()
            p, lo, hi = wilson(agg["sum"], agg["count"])
            ax.errorbar(agg["depth"], p, yerr=[np.maximum(p - lo, 0), np.maximum(hi - p, 0)], fmt="o", color=INK, markersize=4, capsize=2,
                        linewidth=1, label="observed (95% CI)")
        else:
            note_missing(ax, "E3/E4")
        ax.set_title(f"file size {ft} tokens", loc="left")
        ax.set_xlabel("loop depth n")
        ax.set_ylim(-0.02, 1.02)
        ax.text(cal_max / 2, 0.04, "calibration", ha="center", fontsize=7, color=INK2)
        ax.text((cal_max + n_grid.max()) / 2, 0.04, "prediction", ha="center", fontsize=7, color=INK2)
    np.atleast_1d(axes)[0].set_ylabel("P(success)")
    np.atleast_1d(axes)[0].legend(loc="upper right")
    return ctx.save(fig, "fig3_success_vs_n")
