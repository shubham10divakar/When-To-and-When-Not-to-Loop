"""Single source of truth for every model parameter (design doc §2).

Every layer (L1 model, L2 simulation, L3 harness budget estimates, analysis)
reads a :class:`Params` object. No layer hardcodes a parameter value.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_PARAMS_PATH = CONFIG_DIR / "params_default.yaml"
PRICES_PATH = CONFIG_DIR / "prices.yaml"

ARMS = ("naive", "cached", "summarized")


@dataclass(frozen=True)
class Params:
    # --- task shape -------------------------------------------------------
    depth: int = 10                 # n: chain advances the task requires
    max_steps: int | None = None    # M: LLM-call budget per run (None -> 3n)
    answer_call: bool = True        # an extra LLM call to submit the answer after n advances

    # --- tokens -----------------------------------------------------------
    t0: float = 1500.0              # T0: system + task + fixed padding tokens
    delta: float = 1100.0           # δ: tokens appended per step (action + observation)
    a_out: float = 60.0             # a: output tokens per step
    window: int = 200_000           # W: context window (tokens)
    file_tokens: float = 1000.0     # padding target per file (single-shot baseline)
    n_decoy_chains: int = 2         # decoy chains per task (single-shot context size)

    # --- prices -----------------------------------------------------------
    price_in: float = 1.0e-7        # c_in  ($/input token)
    price_out: float = 5.0e-7       # c_out ($/output token)
    kappa: float = 0.1              # κ: cache-read price multiplier
    omega: float = 0.25             # ω: cache-write premium
    tool_cost_usd: float = 0.0      # $ per tool attempt (0 for local tools)

    # --- summarization arm -----------------------------------------------
    summ_every: int = 5             # k
    gamma: float = 0.2              # γ: summary compression ratio
    t0_summ: float = 300.0          # summarization prompt overhead tokens

    # --- reliability -----------------------------------------------------
    p: float = 0.95                 # P(advance | on-track)
    r: float = 0.3                  # P(recover | derailed)
    e: float = 0.05                 # P(commit wrong answer | derailed)
    coupling: str = "none"          # none | exp | linear
    p0: float = 0.97                # coupled: p(T) = p0*exp(-λT) or p0 - λT
    lam: float = 0.0                # λ (per token)

    # --- tool failure injection / retries -------------------------------
    f_inject: float = 0.0           # f: per-attempt failure probability
    d_detect: float = 1.0           # d: fraction of failures that are detectable
    retries: int = 0                # R: tool-layer retry limit

    # --- latency ----------------------------------------------------------
    ttft_a0: float = 0.35           # α0 (s)
    ttft_a1: float = 2.0e-5         # α1 (s / uncached input token)
    ttft_a2: float = 2.0e-6         # α2 (s / cached input token)
    tps: float = 150.0              # τ: decode tokens / s
    tool_lat: float = 0.002         # mean tool latency (s)
    tool_lat_sigma: float = 0.5     # lognormal σ of tool latency (L2)
    llm_lat_sigma: float = 0.25     # lognormal σ of LLM service time noise (L2)

    # --- single-shot baseline -------------------------------------------
    a_ss: float = 30.0              # output tokens of a single-shot answer
    ss_b0: float = 4.0              # logit q = b0 + b1*n + b2*(T_ss/1000)
    ss_b1: float = -0.15
    ss_b2: float = -0.02

    # --- constraints ------------------------------------------------------
    budget: float = 0.05            # B: $ per task
    slo: float = 60.0               # latency SLO (s)
    s_min: float = 0.8              # minimum acceptable success rate

    # --- provenance (not used by the math) -----------------------------
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    # ------------------------------------------------------------------
    @property
    def M(self) -> int:
        return int(self.max_steps) if self.max_steps is not None else 3 * int(self.depth)

    def with_(self, **kw: Any) -> "Params":
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        if not d.get("meta"):
            d.pop("meta", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Params":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"Unknown parameter(s): {sorted(unknown)}")
        return cls(**d)


def _flatten(d: dict[str, Any]) -> dict[str, Any]:
    """YAML files may group parameters into sections; flatten one level."""
    out: dict[str, Any] = {}
    names = {f.name for f in fields(Params)}
    for k, v in d.items():
        if isinstance(v, dict) and k not in names:
            out.update(_flatten(v))
        else:
            out[k] = v
    return out


def load_prices(model: str | None = None, path: Path = PRICES_PATH) -> dict[str, float]:
    data = yaml.safe_load(Path(path).read_text())
    model = model or data["default_model"]
    entry = data["models"][model]
    per = 1.0e6
    return {
        "price_in": entry["input_per_mtok"] / per,
        "price_out": entry["output_per_mtok"] / per,
        "kappa": entry["cache_read_multiplier"],
        "omega": entry["cache_write_multiplier"] - 1.0,
        "window": int(entry["context_window"]),
    }


def load_params(path: str | Path | None = None, *, model: str | None = None,
                overrides: dict[str, Any] | None = None) -> Params:
    """Load params: defaults <- prices.yaml(model) <- YAML file <- overrides."""
    base: dict[str, Any] = {}
    try:
        base.update(load_prices(model))
    except (FileNotFoundError, KeyError):
        pass
    path = Path(path) if path is not None else DEFAULT_PARAMS_PATH
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        base.update(_flatten(raw))
    if overrides:
        base.update(overrides)
    return Params.from_dict(base)


def save_params(params: Params, path: str | Path, extra: dict[str, Any] | None = None) -> None:
    d = params.to_dict()
    if extra:
        d = {**d, **extra}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(d, sort_keys=False))
