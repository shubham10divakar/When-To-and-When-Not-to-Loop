# When To and When Not to Loop

Companion code for the paper **"When To and When Not to Loop: Feasibility Bounds for Agentic LLM Pipelines."**

Agentic loops run into three limits that tighten as loop depth *n* grows:

1. **Context / cost.** Re-sending the history makes input tokens grow ~n² (prompt caching shrinks the constant but not the exponent).
2. **Compounding reliability.** Every step can derail the run.
3. **Serial latency.** Steps run one after another.

The limits are coupled: per-step reliability falls as context grows. Together
they define a **feasible region** of task shapes where looping beats a single
call under a cost budget and a latency SLO. Caching, summarization and retries
move specific boundaries in predictable ways. This repository builds the model,
checks it three ways and produces every figure and table in the paper.

| Layer | What it is | Role |
|---|---|---|
| **L1** analytical model (`loopbounds.model`) | closed forms + exact dynamic programming (numpy) | makes the predictions |
| **L2** simulation (`loopbounds.sim`) | SimPy discrete-event model with the same parameters | Monte Carlo check of L1; latency tails, rate limits, concurrency |
| **L3** real runs (`loopbounds.harness`) | instrumented tool-use agent against the Anthropic API on a controlled pointer-chase task | ground truth: calibrates parameters, then tests held-out predictions |

L1 ↔ L2 agreement verifies the maths; L1/L2 ↔ L3 agreement verifies the
assumptions. Parameters are fitted on short chains (n ∈ {5, 10}) and predictions
for longer chains and boundary points are **pre-registered** before any
prediction-phase data exists (`preregistration/`).

## Quickstart: offline, no API key

```bash
uv sync                          # 1. install the locked environment
make test                        # 2. unit + agreement tests (mocks only)
uv run loopbounds mock-pipeline  # 3. every stage end-to-end with mock clients
```

Step 3 writes logs, fitted parameters, agreement results, figures and tables
under `data/mock/` (≈2 minutes at the default `--scale 0.3`). Figures from
mock data carry a "MOCK DATA" mark.

To rebuild the paper's figures and tables from the released processed data
(`data/processed/`):

```bash
make figures        # -> paper/figures/*.pdf|png, paper/tables/*.tex
```

No make? Every target is `uv run loopbounds <command>`; see `uv run loopbounds --help`.

## Full rerun with real API calls

Requires `ANTHROPIC_API_KEY` (see `.env.example`). Real runs use
`claude-haiku-5-5`; prices are in `configs/prices.yaml` (dated). Every
experiment has a `max_spend_usd` cap. The runner refuses to start if the L1
cost estimate exceeds it, and aborts (keeping partial results, flagged) if
actual spend crosses it. `make estimate` prints the estimates (≈ $52 total
excluding E9, at prior parameter values).

```bash
make pilot          # E0b: choose the difficulty level -> lock it in PREREG.md
make calibrate      # E1, E2, E3, E5, E8a -> data/processed/params_fitted.yaml
make predict        # L1/L2 predictions + E9 boundary configs (refuses if prediction data exists)
# finalize preregistration/PREREG.md, commit it with the predictions, then:
git tag -a prereg-v1 -m "Pre-registration v1"
make prereg-check   # must pass before the prediction phase
make run-predict    # E4, E6, E7, E8b, E9
make external       # E10: coding tasks (external validity)
make analyze        # parquet export, agreement tests -> data/processed/agreement_*.{csv,json}
make figures
```

Runs are resumable: an interrupted experiment skips runs whose logs already exist.

## Open-source models (Ollama, free, local)

The same experiments run against local open-source models through
[Ollama](https://ollama.com). Each model gets its own study under
`data/ollama/<model>/` (logs, fitted parameters, predictions, agreement
checks, figures, tables). No API key is needed and the cost is $0. For local
models the binding wall is latency, and the logged token counts still give the
hosted-API cost of the same runs.

**One command for everything** (steps 2 and 3 below, then a summary):

```bash
uv run loopbounds study --models llama3.1:8b qwen2.5:7b mistral:7b --scale 0.2 -v
```

It runs the E0b pilot on every model, picks one difficulty for all of them (the
hardest level where every model's per-step p̂ lies in 0.85–0.97), runs the full
study per model at that level, and prints a summary that is also saved to
`data/models/study_summary.md`. Pass `--difficulty L0` to skip the pilot. The
command is resumable, so re-running it skips finished runs.

**1. Prerequisites**

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama run llama3.1:8b "hi"   # load once, then:
ollama ps                     # PROCESSOR should read "100% GPU" (CPU-only is ~10x slower)
```

**Note: making sure Ollama uses the GPU.** On the development machine
(RTX 3060, 12 GB), Ollama 0.40.13 ran models on the CPU only: `ollama ps`
showed `100% CPU`, and `%LOCALAPPDATA%\Ollama\server.log` listed only
`inference compute id=cpu`, although the NVIDIA driver was installed. Each
step took about 6 s. Updating Ollama to 0.40.14 and restarting it fixed this.
The log now shows `inference compute ... library=CUDA ... name="NVIDIA GeForce RTX 3060"`
and `ollama ps` shows `100% GPU`. To check after any reinstall or driver update:

```bash
ollama run llama3.1:8b "hi"   # loads the model
ollama ps                     # PROCESSOR must read "100% GPU"
```

If it still says CPU, look for CUDA errors near "discovering available GPUs" in
`server.log`; updating Ollama and/or the NVIDIA driver usually resolves it.
Models larger than VRAM (e.g. `qwen2.5:32b`, 19 GB) spill onto the CPU and run
far slower, so stick to ~7–8B models on a 12 GB card.

**2. Pick the task difficulty (pilot).** The default difficulty (L1, with fake
clues tagged with other keys) is too hard for 7–8B models. A first llama3.1:8b
check reached the answer in only 2 of 18 runs, because it followed fake `[K5]`
clues into decoy chains. The E0b pilot measures per-step reliability p̂ at
every level; choose the hardest level whose p̂ sits in the target band
0.85–0.97 for all models, so per-step errors are rare but measurable:

```bash
uv run loopbounds run E0b_pilot --ollama llama3.1:8b --scale 0.25 -v
uv run loopbounds run E0b_pilot --ollama qwen2.5:7b  --scale 0.25 -v
uv run loopbounds run E0b_pilot --ollama mistral:7b  --scale 0.25 -v
uv run loopbounds pilot-report --ollama llama3.1:8b     # repeat for each model
```

**3. Run all three models end to end** at the chosen level (calibrate → predict → prediction runs → E10 coding tasks → analyze → figures), then compare them:

```bash
uv run loopbounds local-pipeline --models llama3.1:8b qwen2.5:7b mistral:7b --difficulty L0 --scale 0.2
```

With `--difficulty`, each study goes to `data/ollama/<model>-<level>/`, so
runs at different difficulty levels never mix.

- `--scale 0.2` runs 20% of each experiment's planned runs (at least 3 per cell). Raise it for tighter error bars; on CPU, lower it.
- `--num-ctx 32768` sets Ollama's context window (the default). Runs stop at 85% of it, because Ollama silently truncates longer prompts. Chains that outgrow it end as `context_overflow`, which is the context wall.
- `--no-external` skips the E10 coding tasks.
- Runs are resumable: re-running the command skips runs whose logs already exist.

**4. Outputs**

| Path | Content |
|---|---|
| `data/ollama/<model>[-<level>]/processed/` | fitted parameters, predictions, agreement summary, parquet exports |
| `data/ollama/<model>[-<level>]/figures/`, `tables/` | Fig 1–7 and Tab 1–4 for that model |
| `data/models/comparison.csv` | fitted p, r, e, λ, speed and check results per model |
| `data/models/models_success_vs_n.png` | observed success vs n with each model's L1 prediction |

**One model, one stage at a time** (every stage accepts `--ollama MODEL`):

```bash
uv run loopbounds calibrate   --ollama qwen2.5:7b --scale 0.2
uv run loopbounds predict     --ollama qwen2.5:7b
uv run loopbounds run-predict --ollama qwen2.5:7b --scale 0.2
uv run loopbounds external    --ollama qwen2.5:7b
uv run loopbounds analyze     --ollama qwen2.5:7b --l2-runs 20000
uv run loopbounds figures     --ollama qwen2.5:7b
uv run loopbounds compare-models
```

Each model's predictions are written before its prediction-phase data exists,
and `run-predict` refuses to start without them. This is exploratory, so no
git pre-registration tag is required for local models.

**Adding an OpenAI model** (optional, paid): put `OPENAI_API_KEY=...` in `.env`
and append `--openai gpt-4.1-nano --budget 4 --openai-scale 0.12` to the
`local-pipeline` command. The study is hard-capped at `--budget` dollars of
logged spend.

## Experiments

| ID | Phase | Purpose |
|---|---|---|
| E0 | validation | mock end-to-end; estimator recovery |
| E0b | pilot | difficulty level; δ, a, γ, TTFT |
| E1 | calibration | cost scaling, naive vs cached |
| E2 | calibration | context wall (artificial 64K cap + one real-window run) |
| E3 | calibration | fit p, r, e |
| E4 | **prediction** | out-of-sample reliability at n ∈ {20, 40} |
| E5 | calibration | coupling λ (vary T0 at fixed n) |
| E6 | prediction | summarization cost + reliability |
| E7 | prediction | retry ceiling |
| E8a / E8b | calibration / prediction | single-shot baseline |
| E9 | **prediction (headline)** | feasible-region classification near/far from the boundary |
| E10 | external validity | small Python coding tasks with hidden tests |

Configs live in `configs/experiments/`. Different arms and the single-shot
baseline see the same task instances (paired seeds).

## Figures and tables

| | Content |
|---|---|
| Fig 1 | feasible region in (n, p) and (n, δ); arm boundaries, binding walls, E9 overlay |
| Fig 2 | price-weighted input tokens vs calls: naive / cached / summarized, break-even n* |
| Fig 3 | success vs n: observed, naive pⁿ, Markov DP; calibration vs prediction region |
| Fig 4 | advance probability vs context size: coupling fit (exp vs linear) |
| Fig 5 | retry ceiling: success vs R per detectability d |
| Fig 6 | per-call latency vs context; tail latency vs concurrency (L2) |
| Fig 7 | three-layer parity plot |
| Tab 1–4 | fitted parameters, per-cell prediction errors, E9 confusion matrix, E10 summary |

## Repository map

```
configs/                 params_default.yaml (priors), prices.yaml, mock_truth.yaml, experiments/E*.yaml
preregistration/         PREREG.md, thresholds.yaml (frozen + tagged before the prediction phase)
src/loopbounds/
  params.py              single source of truth for all parameters
  model/                 L1: context, cost, reliability (Markov DP), retry, latency, baseline, feasible
  sim/                   L2: SimPy LLM/tool servers, agent process, Monte Carlo driver
  harness/               L3: taskgen, tools, injection, classify, agent, arms, llm/, logging, budget, runner
  tasks_coding/          E10 tasks, environment, scripted mock
  analysis/              fit, predict, agreement, stats
  figures/               fig1 … fig7, tables
  pipeline.py, cli.py    stages and the `loopbounds` command
tests/                   pytest (`-m "not slow and not api"` by default)
data/raw/                per-run JSONL logs (gitignored)
data/processed/          released processed data (parquet/csv/json/yaml)
paper/figures, tables/   generated outputs
```

The full build specification is in `agentic-loop-bounds-design.md`.

## Parameter glossary

| Symbol | Code | Meaning |
|---|---|---|
| n | `depth` | chain advances the task requires |
| M | `max_steps` | LLM-call budget per run (default 3n) |
| T₀ | `t0` | system prompt + tools + task tokens |
| δ | `delta` | tokens appended per step (action + observation) |
| a | `a_out` | output tokens per step |
| W | `window` | context window |
| κ, ω | `kappa`, `omega` | cache-read multiplier, cache-write premium |
| c_in, c_out | `price_in`, `price_out` | $/token |
| k, γ | `summ_every`, `gamma` | summarize every k calls; summary compression ratio |
| p, r, e | `p`, `r`, `e` | P(advance \| on-track), P(recover \| derailed), P(wrong answer \| derailed) |
| p₀, λ | `p0`, `lam` | coupled reliability p(T) = p₀·exp(−λT) (or p₀ − λT) |
| f, d, R | `f_inject`, `d_detect`, `retries` | injected failure rate, detectable fraction, retry limit |
| α₀, α₁, α₂, τ | `ttft_a0..a2`, `tps` | TTFT = α₀ + α₁·T_new + α₂·T_cached; decode tokens/s |
| B, SLO, s_min | `budget`, `slo`, `s_min` | $ per task, latency target, minimum success rate |

## Notes on the real-API setup

- Current Claude models reject sampling parameters (`temperature`, …), so
  determinism is controlled with a fixed `effort` and thinking setting instead
  of temperature 0. Model sampling non-determinism is documented, not removed.
- The cached arm places one `cache_control` breakpoint on the last message
  every step (incremental prefix caching); the naive arm logs cache reads and
  flags any non-zero value.
- Rate-limit backoff is logged separately and excluded from latency analysis.
- Known simplification: costs use the ≤100K-token price tier throughout.

## Citation

See `CITATION.cff`.

## License

Code: MIT. Data (`data/processed/` and released datasets): CC-BY-4.0. See `LICENSE`.
