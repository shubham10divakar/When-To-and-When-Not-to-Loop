"""Cross-model comparison of per-model studies: ``data/{ollama,openai}/<model>/processed``.

Writes ``data/models/comparison.csv`` (fitted parameters + agreement checks per
model) and ``data/models/models_success_vs_n.{pdf,png}`` (observed
success vs n with each model's own L1 Markov prediction).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from loopbounds.params import REPO_ROOT, load_params

DATA = REPO_ROOT / "data"
PROVIDERS = ("ollama", "openai")
OUT = DATA / "models"


def _studies(root: Path = DATA) -> list[Path]:
    return sorted(p for prov in PROVIDERS for p in (root / prov).glob("*/processed")
                  if (p / "params_fitted.yaml").exists())


def compare_models(root: Path = DATA, file_tokens: int = 1000, out: Path = OUT) -> pd.DataFrame:
    rows = []
    studies = _studies(root)
    for proc in studies:
        model = (proc / "MODEL").read_text().split()[0] if (proc / "MODEL").exists() else proc.parent.name
        rep = json.loads((proc / "fit_report.json").read_text())
        chk = json.loads((proc / "agreement_summary.json").read_text()) if (proc / "agreement_summary.json").exists() \
            else {}
        mk = rep.get("markov", {})
        cp = rep.get("coupling", {}).get("exp", {})
        lt = rep.get("latency", {})
        row = {"model": model, "p": mk.get("p"), "r": mk.get("r"), "e": mk.get("e"),
               "p_ci": mk.get("wilson", {}).get("p"), "lambda": cp.get("lam"), "lambda_se": cp.get("lam_se"),
               "coupling": rep.get("coupling_enabled"), "a_out": rep.get("tokens", {}).get("a_out"),
               "ttft_a1_s_per_token": lt.get("ttft_a1"), "decode_tok_s": lt.get("tps"),
               "n_runs_calibration": rep.get("n_runs")}
        for k, c in chk.items():
            row[f"check_{k}"] = c.get("pass")
            if "value" in c and not isinstance(c["value"], (dict, list)):
                row[f"value_{k}"] = c["value"]
        rows.append(row)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparison.csv", index=False)
    if studies:
        _figure(studies, out, file_tokens)
    return df


def _figure(studies: list[Path], out: Path, file_tokens: int) -> None:
    import matplotlib.pyplot as plt

    from loopbounds.analysis.stats import wilson
    from loopbounds.figures.common import INK2, SERIES, apply_style
    from loopbounds.model import reliability

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    n_grid = np.arange(1, 45)
    for i, proc in enumerate(studies):
        c = SERIES[i % len(SERIES)]
        name = (proc / "MODEL").read_text().split()[0].split("/", 1)[-1] if (proc / "MODEL").exists() \
            else proc.parent.name
        P = load_params(proc / "params_fitted.yaml")
        rep = json.loads((proc / "fit_report.json").read_text())
        tk = rep.get("tokens", {})
        P = P.with_(delta=tk.get("delta0", 0.0) + tk.get("delta1", 1.0) * file_tokens, file_tokens=file_tokens)
        y = [reliability.solve(P.with_(depth=int(n), max_steps=3 * int(n)), "naive").p_success for n in n_grid]
        ax.plot(n_grid, y, color=c, label=f"{name} (L1)")
        frames = [pd.read_parquet(f) for f in (proc / "E3_reliability_cal_runs.parquet",
                                               proc / "E4_reliability_pred_runs.parquet") if f.exists()]
        if frames:
            r = pd.concat(frames)
            r = r[(r["file_tokens"] == file_tokens) & (r["arm"] == "naive")]
            agg = r.groupby("depth")["success"].agg(["sum", "count"]).reset_index()
            p, lo, hi = wilson(agg["sum"], agg["count"])
            ax.errorbar(agg["depth"] + 0.4 * (i - 1), p, yerr=[np.maximum(p - lo, 0), np.maximum(hi - p, 0)],
                        fmt="o", color=c, markersize=4, capsize=2, linewidth=1)
    ax.axvspan(0, 10.5, color="#e4e3df", alpha=0.6, linewidth=0)
    ax.text(5, 0.03, "calibration", ha="center", fontsize=7, color=INK2)
    ax.set_xlabel("loop depth n")
    ax.set_ylabel("P(success)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Models: success vs n (file size {file_tokens} tokens)", loc="left")
    ax.legend(loc="lower left")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"models_success_vs_n.{ext}")
    plt.close(fig)
