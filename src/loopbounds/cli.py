"""Command-line entry point: ``loopbounds <command>`` (or ``python -m loopbounds``).

Commands mirror the Makefile targets; add ``--mock`` to run a stage offline with
mock clients into ``data/mock/``.
"""

from __future__ import annotations

import argparse
import json
import sys

from loopbounds import pipeline as pl
from loopbounds.params import REPO_ROOT


def load_dotenv(path=None) -> None:
    """Load KEY=VALUE lines from the repo's .env into os.environ (never overrides existing vars)."""
    import os

    p = path or REPO_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip(chr(34)).strip(chr(39)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="loopbounds", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, help_, mock=True):
        p = sub.add_parser(name, help=help_)
        if mock:
            p.add_argument("--mock", action="store_true", help="use mock clients, write to data/mock/")
            p.add_argument("--ollama", metavar="MODEL", default=None,
                           help="use a local Ollama model, write to data/ollama/<model>/")
            p.add_argument("--num-ctx", type=int, default=32768, help="Ollama context window (num_ctx)")
            p.add_argument("--openai", metavar="MODEL", default=None,
                           help="use an OpenAI model (OPENAI_API_KEY in .env), write to data/openai/<model>/")
            p.add_argument("--budget", type=float, default=None, help="study-wide spend cap in USD")
            p.add_argument("--difficulty", choices=["L0", "L1", "L2", "L3"], default=None,
                           help="task difficulty for every experiment (separate study folder)")
        p.add_argument("-v", "--verbose", action="store_true")
        return p

    p = add("run", "run one or more experiments by name")
    p.add_argument("experiments", nargs="+")
    p.add_argument("--scale", type=float, default=None, help="multiply runs_per_cell")
    p = add("estimate", "L1 cost estimate of experiments (no API calls)")
    p.add_argument("experiments", nargs="*", default=pl.CALIBRATION + pl.PREDICTION)
    p = add("validate", "E0 mock end-to-end + estimator recovery", mock=False)
    p.add_argument("--runs", type=int, default=None)
    p = add("calibrate", "run calibration experiments and fit parameters")
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--fit-only", action="store_true", help="refit from existing logs")
    add("prereg-check", "verify PREREG.md is committed and tagged", mock=False)
    p = add("predict", "write L1/L2 predictions (+ E9 cells) before prediction data exists")
    p.add_argument("--post-hoc", action="store_true")
    p.add_argument("--n-sim", type=int, default=10_000)
    p.add_argument("--regenerate-e9", action="store_true")
    p = add("run-predict", "run prediction-phase experiments (requires prereg-check)")
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--skip-prereg", action="store_true", help="DEBUG ONLY: bypass the prereg guard")
    add("external", "run E10 coding tasks")
    p = add("analyze", "process logs, run agreement tests")
    p.add_argument("--l2-runs", type=int, default=100_000)
    p = add("figures", "render all figures and tables from processed data")
    p.add_argument("--processed", default=None)
    p.add_argument("--out", default=None)
    p = add("mock-pipeline", "every stage offline with mock clients", mock=False)
    p.add_argument("--scale", type=float, default=0.3)
    p.add_argument("--l2-runs", type=int, default=20_000)
    p = add("local-pipeline", "every stage with local Ollama models, one study per model", mock=False)
    p.add_argument("--models", nargs="+", default=["llama3.1:8b", "qwen2.5:7b", "mistral:7b"])
    p.add_argument("--scale", type=float, default=0.2)
    p.add_argument("--num-ctx", type=int, default=32768)
    p.add_argument("--no-external", action="store_true", help="skip E10 coding tasks")
    p.add_argument("--openai", nargs="*", default=[], metavar="MODEL", help="also run these OpenAI models")
    p.add_argument("--budget", type=float, default=4.0, help="spend cap per OpenAI model study (USD)")
    p.add_argument("--openai-scale", type=float, default=None, help="run scale for OpenAI studies")
    p.add_argument("--difficulty", choices=["L0", "L1", "L2", "L3"], default=None,
                   help="task difficulty for every experiment (pick it with pilot-report)")
    add("compare-models", "cross-model table + figure from data/ollama/*/processed", mock=False)
    add("pilot-report", "per-difficulty success and per-step p from E0b pilot logs")

    load_dotenv()
    a = ap.parse_args(argv)
    if a.cmd == "local-pipeline":
        for m in a.models:
            pl.local_pipeline(m, scale=a.scale, num_ctx=a.num_ctx, external_tasks=not a.no_external,
                              difficulty=a.difficulty, verbose=a.verbose)
        for m in a.openai:
            pl.local_pipeline(m, provider="openai", scale=a.openai_scale or a.scale, budget_usd=a.budget,
                              external_tasks=not a.no_external, difficulty=a.difficulty, verbose=a.verbose)
        from loopbounds.analysis.compare import compare_models
        compare_models()
        return 0
    if a.cmd == "compare-models":
        from loopbounds.analysis.compare import compare_models
        print(compare_models().to_string())
        return 0
    paths = pl.Paths.make(mock=getattr(a, "mock", False), ollama=getattr(a, "ollama", None),
                          num_ctx=getattr(a, "num_ctx", 32768), openai=getattr(a, "openai", None),
                          budget_usd=getattr(a, "budget", None), difficulty=getattr(a, "difficulty", None))
    if a.cmd == "run":
        pl.run(a.experiments, paths, scale=a.scale, verbose=a.verbose)
    elif a.cmd == "estimate":
        for k, v in pl.estimate(a.experiments, paths).items():
            print(f"{k:24s} ${v:,.2f}")
    elif a.cmd == "validate":
        res = pl.validate(runs=a.runs)
        print(json.dumps(res, indent=2, default=float))
        return 0 if res["all_recovered"] else 1
    elif a.cmd == "pilot-report":
        print(pl.pilot_report(paths).round(3).to_string(index=False))
    elif a.cmd == "calibrate":
        pl.calibrate(paths, scale=a.scale, verbose=a.verbose, run_experiments=not a.fit_only)
    elif a.cmd == "prereg-check":
        problems = pl.prereg_check()
        if problems:
            print("prereg-check FAILED:\n  - " + "\n  - ".join(problems))
            return 1
        print("prereg-check OK")
    elif a.cmd == "predict":
        pl.predict_stage(paths, post_hoc=a.post_hoc, n_sim=a.n_sim, regenerate_e9=a.regenerate_e9)
    elif a.cmd == "run-predict":
        pl.run_predict(paths, scale=a.scale, skip_prereg=a.skip_prereg, verbose=a.verbose)
    elif a.cmd == "external":
        pl.external(paths, verbose=a.verbose)
    elif a.cmd == "analyze":
        pl.analyze(paths, l2_runs=a.l2_runs)
    elif a.cmd == "figures":
        if a.processed:
            from pathlib import Path

            from loopbounds.figures import make_all
            made = make_all(Path(a.processed), Path(a.out) if a.out else None)
            print(f"{len(made)} files")
        else:
            pl.figures(paths, out=None if a.out is None else __import__("pathlib").Path(a.out))
    elif a.cmd == "mock-pipeline":
        pl.mock_pipeline(scale=a.scale, l2_runs=a.l2_runs, verbose=a.verbose)  # checks are reported, not fatal
    return 0


if __name__ == "__main__":
    sys.exit(main())
