"""Figure 4 — per-step advance probability vs context size: fitted coupling curve,
E5 binned points (naive arm) and the summarized arm (E6) overlaid."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from loopbounds.analysis.stats import wilson
from loopbounds.figures.common import ARM_COLOR, INK2, FigContext, note_missing
from loopbounds.model.reliability import coupled_p


def _binned(steps, n_bins=8):
    s = steps[(steps["phase"] == "chase") & (steps["markov_state"] == "ON")].copy()
    if len(s) < 20:
        return None
    s["bin"] = np.minimum((s["context_tokens"].rank(pct=True) * n_bins).astype(int), n_bins - 1)
    agg = s.groupby("bin").agg(x=("context_tokens", "median"), k=("label", lambda v: (v == "ADVANCE").sum()),
                               n=("label", "size"))
    p, lo, hi = wilson(agg["k"], agg["n"])
    return agg["x"].to_numpy(), p, lo, hi


def make(ctx: FigContext):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    cp = ctx.report.get("coupling", {})
    e5 = ctx.steps("E5_coupling")
    e6 = ctx.steps("E6_summarization")
    xmax = 1.0
    for steps, arm, label in ((e5, "naive", "E5 naive (binned)"), (e6, "summarized", "E6 summarized (binned)")):
        b = _binned(steps) if not steps.empty else None
        if b is not None:
            x, p, lo, hi = b
            ax.errorbar(x / 1000, p, yerr=[np.maximum(p - lo, 0), np.maximum(hi - p, 0)], fmt="o", color=ARM_COLOR[arm], markersize=4,
                        capsize=2, linewidth=1, label=label)
            xmax = max(xmax, x.max() / 1000)
    if cp:
        best = cp["best"]
        T = np.linspace(0, xmax * 1000 * 1.05, 200)
        ax.plot(T / 1000, coupled_p(T, cp[best]["p0"], cp[best]["lam"], best), color=INK2,
                label=f"fit: {best}, λ={cp[best]['lam']:.2e}/token")
        other = "linear" if best == "exp" else "exp"
        ax.plot(T / 1000, coupled_p(T, cp[other]["p0"], cp[other]["lam"], other), color=INK2, linestyle=":",
                linewidth=1, label=f"alt: {other} (ΔAIC={cp[other]['aic'] - cp[best]['aic']:.1f})")
    if e5.empty:
        note_missing(ax, "E5")
    ax.set_xlabel("context size at the step (thousand tokens)")
    ax.set_ylabel("P(advance | on-track)")
    ax.set_title("Coupling: reliability falls as context grows", loc="left")
    ax.legend(loc="lower left")
    return ctx.save(fig, "fig4_coupling")
