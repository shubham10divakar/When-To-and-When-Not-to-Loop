"""Figure 7 — three-layer agreement: predicted (L1 filled, L2 hollow) vs observed
(L3) per prediction cell, one panel per metric."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from loopbounds.figures.common import INK2, SERIES, FigContext, note_missing

PANELS = (("success", "L1_p_success", "L2_p_success", "obs_success", "P(success)"),
          ("cost", "L1_cost", "L2_cost", "obs_cost_mean", "mean cost per run ($)"),
          ("latency", "L1_latency", "L2_latency", "obs_wall_mean", "mean latency per run (s)"))


def make(ctx: FigContext):
    t = ctx.csv("agreement_l3_cells.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    exps = sorted(t["experiment"].unique()) if not t.empty else []
    color = {e: SERIES[i % len(SERIES)] for i, e in enumerate(exps)}
    for ax, (name, l1, l2, obs, label) in zip(axes, PANELS):
        if t.empty or l1 not in t:
            note_missing(ax, "agreement")
            ax.set_title(name, loc="left")
            continue
        d = t[t["arm"] != "single_shot"] if name != "success" else t
        for e in exps:
            g = d[d["experiment"] == e]
            ax.scatter(g[l1], g[obs], s=22, color=color[e], edgecolor="white", linewidth=0.6,
                       label=e if name == "success" else None)
            if l2 in g:
                ax.scatter(g[l2], g[obs], s=22, facecolor="none", edgecolor=color[e], linewidth=1.0)
        vals = np.concatenate([d[l1].to_numpy(float), d[obs].to_numpy(float)])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = vals.min(), vals.max()
            pad = 0.05 * (hi - lo or 1)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, linewidth=1, linestyle="--")
        ax.set_xlabel(f"predicted {label}")
        ax.set_ylabel(f"observed {label}")
        ax.set_title(name, loc="left")
    if exps:
        axes[0].legend(loc="upper left", fontsize=7)
        axes[0].text(0.98, 0.02, "filled: L1, hollow: L2", transform=axes[0].transAxes, ha="right", fontsize=7,
                     color=INK2)
    return ctx.save(fig, "fig7_three_layer_agreement")
