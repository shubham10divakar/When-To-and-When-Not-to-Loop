"""Figure 2 — price-weighted input tokens vs number of calls for naive, cached and
summarized arms: L1 curves + measured runs (E1, E6); break-even n* marked."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from loopbounds.figures.common import ARM_COLOR, ARM_LABEL, INK2, FigContext, note_missing, ref_params
from loopbounds.model import cost


def weighted_input(P, df):
    return df["tokens.input_new"] + P.kappa * df["tokens.cache_read"] + (1 + P.omega) * df["tokens.cache_write"]


def make(ctx: FigContext, file_tokens: int = 1000, summ_every: int = 5):
    P = ref_params(ctx, file_tokens).with_(summ_every=summ_every)
    runs = ctx.runs("E1_cost")
    summ = ctx.runs("E6_summarization")
    m_max = 125
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    m = np.arange(1, m_max + 1)
    for arm in ("naive", "cached", "summarized"):
        t = cost.per_call_tokens(arm, P, m_max)
        w = np.cumsum(t["input_new"] + P.kappa * t["cache_read"] + (1 + P.omega) * t["cache_write"])
        ax.plot(m, w / 1e6, color=ARM_COLOR[arm], label=f"{ARM_LABEL[arm]} (L1)")
    pts = []
    if not runs.empty:
        pts.append(runs[(runs["file_tokens"] == file_tokens) & runs["arm"].isin(["naive", "cached"])])
    if not summ.empty:
        pts.append(summ[(summ["summ_every"] == summ_every) & (summ["file_tokens"] == file_tokens)])
    if pts:
        for d in pts:
            for arm, g in d.groupby("arm"):
                ax.scatter(g["steps"], weighted_input(P, g) / 1e6, s=14, color=ARM_COLOR[arm], alpha=0.7,
                           edgecolor="white", linewidth=0.5)
        ax.scatter([], [], s=14, color=INK2, label="measured runs")
    else:
        note_missing(ax, "E1/E6")
    nstar = cost.breakeven_n(P, "naive", "summarized", n_hi=m_max)
    if nstar:
        ax.axvline(nstar, color=INK2, linestyle=":", linewidth=1)
        ax.text(nstar + 1.5, ax.get_ylim()[1] * 0.45, f"break-even n* = {nstar}\n(summarized < naive)",
                fontsize=8, color=INK2)
    ax.set_xlabel("LLM calls per run")
    ax.set_ylabel("price-weighted input tokens (millions)")
    ax.set_title(f"Cost scaling (file size {file_tokens} tokens, k={summ_every})", loc="left")
    ax.legend(loc="upper left")
    return ctx.save(fig, "fig2_cost_scaling")
