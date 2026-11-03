"""Paper figures (Fig 1–7) and tables (Tab 1–4), from ``data/processed`` only."""

from __future__ import annotations

from pathlib import Path

from loopbounds.figures import (
    fig1_feasible_region, fig2_cost_scaling, fig3_success_vs_n, fig4_coupling, fig5_retry_ceiling, fig6_latency,
    fig7_three_layer_agreement, tables,
)
from loopbounds.figures.common import make_context

FIGURES = [fig1_feasible_region, fig2_cost_scaling, fig3_success_vs_n, fig4_coupling, fig5_retry_ceiling,
           fig6_latency, fig7_three_layer_agreement]


def make_all(processed: str | Path | None = None, out: str | Path | None = None,
             tables_out: str | Path | None = None, only: list[str] | None = None) -> list[Path]:
    ctx = make_context(processed, out)
    paths: list[Path] = []
    for mod in FIGURES:
        name = mod.__name__.rsplit(".", 1)[-1]
        if only and not any(name.startswith(o) for o in only):
            continue
        paths += mod.make(ctx)
    if not only or "tables" in only:
        paths += tables.make(ctx, Path(tables_out) if tables_out else None)
    return paths
