"""Shared figure context: data loading from ``data/processed`` and a consistent style.

Figure scripts read only processed data and ``params_fitted.yaml``; none needs
an API key. If a piece of L3 data is missing, the figure still renders its
model curves and says which data is absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from loopbounds.params import REPO_ROOT, Params, load_params  # noqa: E402

# Categorical slots in fixed order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ARM_COLOR = {"naive": SERIES[0], "cached": SERIES[1], "summarized": SERIES[2], "retries": SERIES[3],
             "single_shot": SERIES[4]}
ARM_LABEL = {"naive": "naive", "cached": "cached", "summarized": "summarized", "retries": "naive + retries",
             "single_shot": "single-shot"}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
REGION_FILL = "#cde2fb"   # sequential blue step 100 (light region shading)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.5, "lines.markersize": 5, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.dpi": 200,
    })


@dataclass
class FigContext:
    processed: Path
    out: Path
    params: Params
    report: dict = field(default_factory=dict)
    is_mock: bool = False
    _cache: dict = field(default_factory=dict)

    def runs(self, exp: str) -> pd.DataFrame:
        return self._load(f"{exp}_runs.parquet")

    def steps(self, exp: str) -> pd.DataFrame:
        return self._load(f"{exp}_steps.parquet")

    def csv(self, name: str) -> pd.DataFrame:
        p = self.processed / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def json(self, name: str) -> dict:
        p = self.processed / name
        return json.loads(p.read_text()) if p.exists() else {}

    def _load(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            p = self.processed / name
            self._cache[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        return self._cache[name]

    def save(self, fig, name: str) -> list[Path]:
        self.out.mkdir(parents=True, exist_ok=True)
        if self.is_mock:
            fig.text(0.995, 0.005, "MOCK DATA", ha="right", va="bottom", fontsize=7, color=INK2)
        paths = []
        for ext in ("pdf", "png"):
            p = self.out / f"{name}.{ext}"
            fig.savefig(p)
            paths.append(p)
        plt.close(fig)
        return paths


def make_context(processed: str | Path | None = None, out: str | Path | None = None) -> FigContext:
    apply_style()
    processed = Path(processed) if processed else REPO_ROOT / "data" / "processed"
    out = Path(out) if out else REPO_ROOT / "paper" / "figures"
    pf = processed / "params_fitted.yaml"
    params = load_params(pf) if pf.exists() else load_params()
    rep_p = processed / "fit_report.json"
    report = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    is_mock = (processed / "MOCK").exists()
    return FigContext(processed, out, params, report, is_mock)


def note_missing(ax, what: str) -> None:
    ax.text(0.98, 0.02, f"no {what} data", transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color=INK2)


def ref_params(ctx: FigContext, file_tokens: float = 1000.0, depth: int = 10) -> Params:
    """Fitted params evaluated at a reference cell (δ from the file-size fit)."""
    tk = ctx.report.get("tokens", {})
    delta = tk.get("delta0", 0.0) + tk.get("delta1", 1.0) * file_tokens if tk else ctx.params.delta
    return ctx.params.with_(depth=depth, delta=float(delta), file_tokens=float(file_tokens))
