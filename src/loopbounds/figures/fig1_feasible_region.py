"""Figure 1 — the feasible region.

(a) (n, p) plane at the reference file size: shaded naive feasible region,
    boundary of each arm (naive / cached / summarized / naive + retries) and the
    binding wall labelled where the naive arm is infeasible.
(b) (n, δ) plane at the fitted p with the E9 configurations overlaid
    (correct vs misclassified feasibility predictions).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import yaml

from loopbounds.figures.common import ARM_COLOR, ARM_LABEL, INK, INK2, REGION_FILL, FigContext, ref_params
from loopbounds.model import feasible
from loopbounds.params import CONFIG_DIR

WALL_TEXT = {"success": "reliability wall", "cost": "cost wall", "latency": "latency wall", "context": "context wall"}
CRITICAL = "#d03b3b"


def constraints() -> dict:
    p = CONFIG_DIR / "experiments" / "E9_feasibility.yaml"
    try:
        return yaml.safe_load(p.read_text()).get("constraints", {})
    except FileNotFoundError:
        return {}


def _p_axis_param(P):
    return "p0" if P.coupling not in (None, "none") else "p"


def make(ctx: FigContext, n_max: int = 80):
    P = ref_params(ctx).with_(**constraints())
    n_vals = np.arange(2, n_max + 1, 2)
    p_vals = np.linspace(0.80, 0.995, 40)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.0))

    grids = {}
    for arm in ("naive", "cached", "summarized"):
        grids[arm] = feasible.feasible_grid_np(P, arm, n_vals, p_vals)
    grids["retries"] = feasible.feasible_grid_np(P.with_(f_inject=0.2, d_detect=0.5, retries=3), "naive",
                                                 n_vals, p_vals)
    g0 = grids["naive"]
    ax.contourf(n_vals, p_vals, g0.feasible.T.astype(float), levels=[0.5, 1.5], colors=[REGION_FILL])
    for arm, g in grids.items():
        if g.feasible.any() and not g.feasible.all():
            ax.contour(n_vals, p_vals, g.feasible.T.astype(float), levels=[0.5], colors=[ARM_COLOR[arm]],
                       linewidths=1.6)
        ax.plot([], [], color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    for wall, text in WALL_TEXT.items():
        mask = g0.binding == wall
        if mask.sum() >= 4:
            ii, jj = np.nonzero(mask)
            ax.text(np.median(n_vals[ii]), np.median(p_vals[jj]), text, ha="center", va="center", fontsize=8,
                    color=INK2)
    ax.text(n_vals[1], p_vals[-2], "feasible\n(naive)", fontsize=8, color=INK, va="top")
    ax.set_xlabel("loop depth n (steps)")
    ax.set_ylabel(f"per-step reliability {_p_axis_param(P)}")
    ax.set_title("(a) feasible region in (n, p)", loc="left")
    ax.legend(loc="lower right")

    # (b) n x delta with E9 overlay
    d_vals = np.linspace(300, 3300, 21)
    n_b = np.arange(2, 62, 3)
    for arm in ("naive", "cached", "summarized"):
        g = feasible.feasible_grid_ndelta(P, arm, n_b, d_vals)
        if arm == "naive":
            bx.contourf(n_b, d_vals, g.feasible.T.astype(float), levels=[0.5, 1.5], colors=[REGION_FILL])
        if g.feasible.any() and not g.feasible.all():
            bx.contour(n_b, d_vals, g.feasible.T.astype(float), levels=[0.5], colors=[ARM_COLOR[arm]],
                       linewidths=1.6)
        bx.plot([], [], color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    e9 = ctx.csv("e9_table.csv")
    if not e9.empty:
        tk = ctx.report.get("tokens", {})
        dl = tk.get("delta0", 0.0) + tk.get("delta1", 1.0) * e9["file_tokens"]
        ok = e9["correct"].astype(bool)
        bx.scatter(e9.loc[ok, "depth"], dl[ok], marker="o", s=30, facecolor=INK2, edgecolor="white",
                   linewidth=1.0, label="E9: correct", zorder=3)
        bx.scatter(e9.loc[~ok, "depth"], dl[~ok], marker="X", s=40, color=CRITICAL, edgecolor="white",
                   linewidth=0.8, label="E9: misclassified", zorder=3)
        acc = ok.mean()
        bx.set_title(f"(b) feasible region in (n, δ); E9 accuracy {acc:.0%}", loc="left")
    else:
        bx.set_title("(b) feasible region in (n, δ)", loc="left")
        bx.text(0.98, 0.02, "no E9 data", transform=bx.transAxes, ha="right", fontsize=7, color=INK2)
    bx.set_xlabel("loop depth n (steps)")
    bx.set_ylabel("tokens appended per step δ")
    bx.legend(loc="upper right")
    fig.suptitle(f"Constraints: s_min={P.s_min:g}, budget=${P.budget:g}/task, latency SLO={P.slo:g}s, "
                 f"W={P.window:,} tokens", fontsize=8, color=INK2, y=1.0)
    return ctx.save(fig, "fig1_feasible_region")
