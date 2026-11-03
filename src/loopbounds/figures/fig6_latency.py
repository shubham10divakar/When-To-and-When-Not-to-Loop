"""Figure 6 — latency. (a) per-call LLM latency vs context with the fitted model
(naive vs cached); (b) end-to-end p95/p99 vs number of concurrent agents sharing
a rate limit (L2 simulation)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from loopbounds.figures.common import ARM_COLOR, ARM_LABEL, SERIES, FigContext, note_missing, ref_params
from loopbounds.sim import run_sim, summarize

AGENT_EXPS = ("E1_cost", "E3_reliability_cal", "E4_reliability_pred", "E5_coupling")


def make(ctx: FigContext, concurrency=(1, 4, 16, 64), tokens_per_min: float = 4_000_000, n_runs: int = 400):
    from loopbounds.figures.fig1_feasible_region import constraints

    P = ref_params(ctx, 1000, 10).with_(**constraints())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6))
    steps = pd.concat([ctx.steps(e) for e in AGENT_EXPS], ignore_index=True)
    t_hi = float(steps["context_tokens"].max()) * 1.05 if not steps.empty else 140_000.0
    T = np.linspace(0, t_hi, 100)
    for arm in ("naive", "cached"):
        if not steps.empty:
            s = steps[steps["arm"] == arm]
            if len(s) > 3000:
                s = s.sample(3000, random_state=0)
            ax.scatter(s["context_tokens"] / 1000, s["llm_total_s"], s=5, color=ARM_COLOR[arm], alpha=0.35,
                       linewidth=0)
        if arm == "naive":
            fit = P.ttft_a0 + P.ttft_a1 * T + P.a_out / P.tps
        else:
            fit = P.ttft_a0 + P.ttft_a1 * P.delta + P.ttft_a2 * np.maximum(T - P.delta, 0) + P.a_out / P.tps
        ax.plot(T / 1000, fit, color=ARM_COLOR[arm], label=f"{ARM_LABEL[arm]}: fitted model")
    if steps.empty:
        note_missing(ax, "step")
    ax.set_xlabel("context size (thousand tokens)")
    ax.set_ylabel("LLM call latency (s)")
    ax.set_title("(a) per-call latency vs context", loc="left")
    ax.legend(loc="upper left")

    rows = []
    for c in concurrency:
        s = summarize(run_sim(P, "naive", max(n_runs, 2 * c), seed=c, concurrency=c, tokens_per_min=tokens_per_min))
        rows.append((c, s["e2e_p95"], s["e2e_p99"]))
    rows = np.array(rows)
    bx.plot(rows[:, 0], rows[:, 1], "o-", color=SERIES[0], label="p95")
    bx.plot(rows[:, 0], rows[:, 2], "s-", color=SERIES[1], label="p99")
    bx.axhline(P.slo, color="#52514e", linestyle=":", linewidth=1)
    bx.text(rows[0, 0], P.slo, " SLO", va="bottom", fontsize=7, color="#52514e")
    bx.set_xscale("log", base=2)
    bx.set_xticks(list(concurrency), [str(c) for c in concurrency])
    bx.set_xlabel("concurrent agents sharing the rate limit")
    bx.set_ylabel("end-to-end run latency (s)")
    bx.set_title(f"(b) tail latency, n={P.depth}, {tokens_per_min / 1e6:g}M tokens/min (L2)", loc="left")
    bx.legend(loc="upper left")
    return ctx.save(fig, "fig6_latency")
