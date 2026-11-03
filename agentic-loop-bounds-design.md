# When To and When Not to Loop — Implementation Design Doc

**Purpose.** This document is the complete build specification for the companion repository of the paper *"When To and When Not to Loop: Feasibility Bounds for Agentic LLM Pipelines."* An implementer (human or coding agent) should be able to build the entire repo from this file alone. Everything in the paper — every formula, experiment, figure, and table — maps to a component defined here.

**Core claim being tested.** Agentic loops face three walls that tighten with loop depth *n*: (1) context/cost, (2) compounding reliability, (3) serial latency. These walls are coupled (per-step reliability falls as context grows). Together they define a *feasible region* of task shapes where looping beats a single-shot call under a cost budget and latency SLO. Standard interventions (prompt caching, summarization, retries) move specific boundaries in predictable ways.

**Verification strategy: three layers that must agree.**

| Layer | What it is | Role |
|---|---|---|
| L1 — Analytical model | numpy; closed forms + exact dynamic programming | Makes the predictions |
| L2 — SimPy simulation | Stochastic discrete-event model, same parameters | Monte Carlo cross-check of L1; handles what L1 can't (latency tails, rate limits, concurrency) |
| L3 — Real runs | Instrumented agent against a real LLM API on a controlled synthetic task | Ground truth; calibrates parameters, then tests out-of-sample predictions |

L1 ↔ L2 agreement verifies the math. L2/L1 ↔ L3 agreement verifies the assumptions. Disagreement in the second check is a finding, not a bug — report which assumption broke.

**Method: calibrate-then-predict.** Parameters are fitted on short chains (n ∈ {5, 10}), then L1/L2 predict outcomes at longer chains (n ∈ {20, 40}) and at feasible-region boundary points. Pass criteria are pre-registered before the prediction phase runs.

---

## 1. Repository layout

```
agentic-loop-bounds/
├── README.md                    # paper link, quickstart, reproduce-figures instructions
├── LICENSE                      # MIT (code); CC-BY-4.0 (data)
├── CITATION.cff
├── pyproject.toml               # managed with uv; lockfile committed
├── uv.lock
├── Makefile                     # all entry points (§10)
├── .env.example                 # ANTHROPIC_API_KEY=...
├── configs/
│   ├── params_default.yaml      # single source of truth for model parameters
│   ├── prices.yaml              # $/token table, dated; used for cost computation
│   └── experiments/
│       ├── E0_mock.yaml … E10_coding.yaml
├── preregistration/
│   └── PREREG.md                # committed + git-tagged BEFORE prediction phase
├── src/loopbounds/
│   ├── params.py                # dataclasses for all parameters; YAML load/save
│   ├── model/                   # ── Layer 1 ──
│   │   ├── cost.py              # token-cost formulas: naive, cached, summarized
│   │   ├── context.py           # context wall n_max, context trajectory T(i)
│   │   ├── reliability.py       # naive p^n, Markov DP, coupled p(T) DP
│   │   ├── retry.py             # retry/detectability closed forms
│   │   ├── latency.py           # per-step and total latency model
│   │   ├── baseline.py          # single-shot cost/feasibility/success model
│   │   └── feasible.py          # feasible-region grid + agent-vs-single-shot dominance
│   ├── sim/                     # ── Layer 2 ──
│   │   ├── agent_sim.py         # SimPy agent process
│   │   ├── environment.py       # LLM server w/ rate limit, tool server
│   │   └── run_sim.py           # Monte Carlo driver
│   ├── harness/                 # ── Layer 3 ──
│   │   ├── taskgen.py           # pointer-chase task generator
│   │   ├── tools.py             # list_files, read_file, answer
│   │   ├── injection.py         # failure injection + tool-layer retry policy
│   │   ├── agent.py             # instrumented ReAct loop
│   │   ├── arms.py              # naive / cached / summarized context strategies
│   │   ├── single_shot.py       # single-shot baseline runner
│   │   ├── classify.py          # step labelling (advance / redundant / wrong)
│   │   ├── llm/
│   │   │   ├── base.py          # LLMClient interface
│   │   │   ├── anthropic_client.py
│   │   │   └── mock.py          # scripted + stochastic mock policies
│   │   ├── budget.py            # spend guard
│   │   ├── logging.py           # JSONL step + run records
│   │   └── runner.py            # experiment runner (config → runs → logs)
│   ├── tasks_coding/            # secondary realistic task (E10)
│   ├── analysis/
│   │   ├── fit.py               # parameter estimation from logs
│   │   ├── predict.py           # L1/L2 predictions for held-out configs
│   │   ├── agreement.py         # L1↔L2, L1/L2↔L3 tests
│   │   └── stats.py             # Wilson CIs, bootstrap, regressions
│   └── figures/
│       ├── fig1_feasible_region.py … fig7_three_layer_agreement.py
│       └── tables.py
├── tests/                       # pytest (§9)
├── data/
│   ├── raw/                     # gitignored; per-run JSONL
│   └── processed/               # released with the paper (parquet)
└── paper/
    ├── figures/                 # generated PDFs/PNGs
    └── tables/                  # generated LaTeX tables
```

---

## 2. Notation and parameters (single source of truth)

All parameters live in `params.py` as frozen dataclasses, loaded from YAML. Every layer reads the same object. **No layer hardcodes a parameter.**

| Symbol | Name in code | Meaning | Source |
|---|---|---|---|
| n | `depth` | steps (chain length) the task requires | set |
| M | `max_steps` | step budget per run (default `3n`) | set |
| T₀ | `t0` | system prompt + task + fixed padding tokens | measured |
| δ | `delta` | tokens appended to context per step (action + observation) | measured |
| a | `a_out` | output tokens per step | measured |
| W | `window` | context window | model spec |
| κ | `kappa` | price multiplier for cache-read tokens | `prices.yaml` |
| ω | `omega` | price premium for cache-write tokens | `prices.yaml` |
| c_in, c_out | `price_in`, `price_out` | $/token | `prices.yaml` |
| k | `summ_every` | summarize every k steps | set |
| γ | `gamma` | summary compression ratio | measured |
| p | `p` | P(advance \| on-track) | fitted |
| r | `r` | P(recover \| derailed) | fitted |
| e | `e` | P(commit wrong answer \| derailed) | fitted |
| p₀, λ | `p0`, `lam` | coupled reliability p(T) = p₀·exp(−λT) | fitted |
| f | `f_inject` | per-attempt injected failure probability | set |
| d | `d_detect` | fraction of injected failures that are detectable | set |
| R | `retries` | tool-layer retry limit for detectable failures | set |
| α₀, α₁, α₂ | `ttft_*` | TTFT = α₀ + α₁·T_new + α₂·T_cached | fitted |
| τ | `tps` | decode tokens/sec | fitted |
| ℓ_tool | `tool_lat` | tool latency distribution | measured |
| B | `budget` | $ per task | set |
| SLO | `slo` | latency target (p95, seconds) | set |
| s_min | `s_min` | minimum acceptable success rate | set |

---

## 3. Layer 1 — Analytical model (`src/loopbounds/model/`)

All functions are pure, vectorized over numpy arrays where sensible, and documented with the equation number used in the paper.

### 3.1 Context trajectory and context wall (`context.py`)

- `context_at_step(i, t0, delta) -> T(i) = t0 + i*delta` (naive arm)
- `n_max(window, t0, delta) -> floor((window - t0) / delta)` — first step at which the naive loop overflows.
- `context_trajectory(arm, params, n) -> array[T(i)]` for naive / cached (same as naive) / summarized (§3.2.3).

### 3.2 Token cost (`cost.py`)

Return **expected dollar cost** and **token breakdown** (`input_new`, `cache_read`, `cache_write`, `output`).

**3.2.1 Naive (no caching).**
C_in(n) = Σ_{i=0}^{n−1} (T₀ + iδ) = n·T₀ + δ·n(n−1)/2
Cost = c_in·C_in + c_out·n·a. *Leading term: c_in·δ·n²/2.*

**3.2.2 Cached (incremental prefix caching).** At step i ≥ 1 the previous context T(i−1) is a cache hit; the δ new tokens are written to cache.
Cost_in = c_in·[T₀(1+ω) + Σ_{i=1}^{n−1} (κ·T(i−1) + (1+ω)·δ)]
*Leading term: c_in·κ·δ·n²/2 — still quadratic, smaller constant.* This is a headline claim; the function must expose the leading-term coefficient so a figure can show it.

**3.2.3 Summarized.** History h (excluding T₀) grows by δ per step; every k steps it is replaced by a summary of size γ·(h + kδ). Fixed point after the first cycle:
h* = γ·k·δ / (1 − γ)
Each summary call costs input (T₀_summ + h* + kδ) and output γ·(h* + kδ). Implement by **direct step simulation of the deterministic recurrence** (not an approximation), so cost is exact for any n, including the first partial cycle. Per-step cost is bounded ⇒ total cost is O(n).

**3.2.4 Summarization break-even.**
`breakeven_n(params, arm_a, arm_b) -> n*` = smallest n where cost(arm_b) ≤ cost(arm_a). Computed numerically (scan n = 1…n_max). Report also as a function of (γ, k, δ) for a sensitivity figure.

### 3.3 Reliability (`reliability.py`)

**3.3.1 Naive:** `p_naive(p, n) = p**n`.

**3.3.2 Markov recovery model (exact DP).** States: ON (on-track), DER (derailed), plus absorbing SUCCESS and FAIL.

- From ON: advance w.p. p (stay ON, advances += 1); else → DER.
- From DER: advance w.p. r (→ ON, advances += 1); commit wrong answer w.p. e (→ FAIL); else stay DER.
- SUCCESS when advances == n; FAIL if step budget M is exhausted.

`p_success_markov(p, r, e, n, M)` computes the DP over (step, advances, state). Complexity O(M·n). Also returns the distribution of steps-to-success (needed for the cost-per-success metric).

**Consistency check (unit test):** r = 0, e = 0, M = n ⇒ result equals pⁿ exactly.

**3.3.3 Coupled reliability.** p becomes step-dependent: p(i) = p₀·exp(−λ·T(i)), with T(i) from the arm's context trajectory. Same DP with p indexed by step. This is how summarization's *reliability* benefit enters the model, not just its cost benefit. (A linear form p₀ − λT is also implemented; model selection by AIC in `fit.py`.)

### 3.4 Retries and detectability (`retry.py`)

Per tool attempt: failure w.p. f; a failure is detectable w.p. d. Detectable failures are retried up to R times; silent failures pass through as corrupted content.

- Clean delivery: P_clean = (1−f)·(1 − (fd)^{R+1}) / (1 − fd)
- Silent corruption: P_silent = f(1−d)·(1 − (fd)^{R+1}) / (1 − fd)
- Surfaced error (retries exhausted): P_exhaust = (fd)^{R+1}
- Expected attempts: E[A] = (1 − (fd)^{R+1}) / (1 − fd)
- **Ceiling as R → ∞:** P_clean → (1−f)/(1−fd) < 1 whenever d < 1.

Composition with agent reliability: effective per-step advance probability p_eff = p·P_clean; silent corruption routes the agent into DER (decoy chain). Exhausted errors are surfaced to the agent, which may retry on its own (captured by r). Cost multiplies tool-call cost by E[A]; latency adds E[A]·ℓ_tool.

### 3.5 Latency (`latency.py`)

Per step: L(i) = TTFT(i) + a/τ + ℓ_tool, with TTFT(i) = α₀ + α₁·T_new(i) + α₂·T_cached(i).
Total: L(n) = Σ L(i). L1 returns the **mean**; tail percentiles come from L2 (§4).

### 3.6 Single-shot baseline (`baseline.py`)

- Context: T_ss = T₀ + Σ(all file tokens). Feasible only if T_ss ≤ W.
- Cost: c_in·T_ss + c_out·a_ss.
- Success q(n, file_tokens): **empirical, not derived.** Fitted as a logistic model on single-shot runs (E8). State this openly in the paper as the weakest modeling assumption.

### 3.7 Feasible region (`feasible.py`)

For a grid over (n, p) — and a second grid over (n, δ):

Constraints for the agentic arm:
1. P_success ≥ s_min
2. E[cost] ≤ B
3. latency (p95 from L2, or mean from L1 with a flag) ≤ SLO
4. max context ≤ W (or summarization arm used)

**Dominance rule.** Agent is preferred iff it satisfies all constraints AND (single-shot is infeasible, OR single-shot success < s_min, OR agent cost-per-success < single-shot cost-per-success), where cost-per-success = E[cost] / P(success).

Output: boolean masks per constraint, a "binding constraint" label per cell (which wall fails first), and region polygons for each arm (naive / cached / summarized / with retries) so Figure 1 can overlay boundary shifts.

---

## 4. Layer 2 — SimPy simulation (`src/loopbounds/sim/`)

**Role (honest scope).** The exact DP in L1 already handles step-varying p(i). L2 exists to (a) independently Monte Carlo-check L1's cost, reliability, and retry math, and (b) model the **time dimension** L1 can't do in closed form: latency distributions and tails, API rate limits, and multiple concurrent agents sharing a rate limit.

### 4.1 Entities

- `LLMServer` — SimPy `Resource` with capacity = concurrent-request limit. Optional token-bucket rate limiter (tokens/min). Service time = TTFT(T_new, T_cached) + a/τ, with lognormal noise fitted from L3 residuals.
- `ToolServer` — service time drawn from the empirical ℓ_tool distribution (bootstrap from L3 logs). Applies failure injection (f, d) and the retry policy (R).
- `AgentProcess` — runs the same state machine as §3.3.2 (ON/DER/SUCCESS/FAIL) using the same parameters, tracking context with the same arm logic as §3.2 (naive / cached / summarized), and accumulating tokens, cost, and wall-time.

### 4.2 Interface

`run_sim(params, arm, n_runs, seed, concurrency=1) -> DataFrame` with one row per simulated run, same columns as the L3 run record (§5.7), so L2 and L3 outputs are directly comparable by the same analysis code.

### 4.3 Required capabilities

- Arms: naive, cached, summarized.
- Coupled p(T) on/off.
- Retry policy with (f, d, R).
- Concurrency sweep (1, 4, 16, 64 agents) under a shared rate limit → throughput and tail-latency curves (supports the latency wall and a short "multi-tenant" discussion paragraph).
- Deterministic seeding; 10⁴–10⁵ runs per config in seconds to minutes.

---

## 5. Layer 3 — Real-run harness (`src/loopbounds/harness/`)

### 5.1 Task: pointer-chase over files (`taskgen.py`)

A task instance is a virtual directory (in memory; no disk I/O needed) of files.

- **True chain:** files F₀ → F₁ → … → F_n. F₀ is named in the task prompt. Each Fᵢ contains a clue identifying F_{i+1}. F_n contains `ANSWER: <random 8-char token>`.
- **File names:** random 6-char strings (no ordering leakage).
- **Padding:** each file holds filler text sized to a target token count (`file_tokens` ∈ {200, 1000, 3000}). Filler is generated from a fixed seeded corpus (e.g. shuffled public-domain sentences), so it is natural-looking but contentless. The clue is inserted at a uniformly random position.
- **Decoys:** `n_decoy_chains` decoy chains (same length distribution) ending in fake ANSWER tokens. Silent failure injection redirects the agent into a decoy chain.
- **Distractor clues:** each file may contain `n_fake_clues` additional clue-like lines that are invalid under a rule stated in the system prompt (e.g. "only follow clues tagged with the key `K7`"; fake clues carry other keys).
- **Difficulty knob (critical).** Strong models may score p ≈ 1 on a trivial chase, leaving nothing to measure. Difficulty levels:
  - L0: clue states the file name directly.
  - L1: valid clue among fake clues with wrong keys.
  - L2: L1 + indirection (e.g. "next file is the one whose name, reversed, is `xq3kfa`").
  - L3: L2 + clue split across two lines separated by filler.
  A **pilot (E0b)** picks the level giving p in roughly 0.85–0.97 for the chosen model. Lock it in PREREG.md.
- Fixed `T0_pad`: an optional filler block appended to the system prompt, to vary context size without changing step count (used by E5 for λ).
- Output: `TaskInstance(files: dict[name, str], start: str, answer: str, chain: list[str], decoy_map, meta)`. Fully determined by `(seed, config)`.

### 5.2 Tools (`tools.py`)

- `list_files() -> str` — names only.
- `read_file(name) -> str` — routed through `injection.py`.
- `answer(token) -> ends run`.

Tool schemas are passed through the provider's native tool-use API.

### 5.3 Agent loop (`agent.py`, `arms.py`)

A standard ReAct-style tool-use loop:

```
messages = [system, task]
for step in range(M):
    resp = llm.complete(messages, tools, arm_options)   # streaming
    log step metrics
    if resp.tool == "answer": end
    obs = tools.execute(resp.tool_call)
    messages = arm.update(messages, resp, obs, step)     # naive / cached / summarized
```

Arms:
- **naive** — append everything; no cache markers. Confirm caching is actually off for the provider (log cached-token counts; assert zero).
- **cached** — append everything; place a cache breakpoint on the last message each step (incremental prefix caching). Verify the provider's current caching mechanics and minimum cacheable length in its docs before implementing.
- **summarized** — every k steps, call the LLM with a fixed summarization prompt over the history since T₀, instructing it to preserve the current chain position and the last valid clue. Replace the history with the summary. Log summary input/output tokens (S) and the achieved γ.

Settings: temperature 0, fixed max output tokens, pinned model ID (exact version string recorded from every API response).

### 5.4 Failure injection and retry policy (`injection.py`)

Applied only to `read_file` calls on true-chain files. Per attempt, an independent draw:
- w.p. (1−f): return the true content.
- w.p. f·d: detectable failure → `ERROR: transient read failure`. The tool layer retries up to R times automatically; if exhausted, the error string is returned to the agent.
- w.p. f·(1−d): silent failure → return the content with the valid clue rewritten to point into a decoy chain.

Every attempt is logged (type, attempt index). The RNG stream is separate from the task-generation RNG, so injection is reproducible independently.

### 5.5 Step classification (`classify.py`)

Each `read_file` action is labelled against the ground-truth chain:
- `ADVANCE` — reads F_{progress+1} (progress += 1)
- `REDUNDANT` — re-reads an already-visited chain file
- `WRONG` — reads a non-chain file or skips ahead
- `LIST` — a `list_files` call (neutral)
- `ANSWER_CORRECT` / `ANSWER_WRONG`

Markov state assignment for fitting: the agent is DER after any `WRONG` action until its next `ADVANCE`, and ON otherwise. Estimators:
- p̂ = #ADVANCE-from-ON / #steps-from-ON
- r̂ = #ADVANCE-from-DER / #steps-from-DER
- ê = #ANSWER_WRONG-from-DER / #steps-from-DER

`REDUNDANT` and `LIST` count as non-advancing steps in their current state. This definition is fixed in PREREG.md.

### 5.6 Single-shot baseline (`single_shot.py`)

One call containing every file (name + content) plus the task, asking for the answer token. If T_ss > W, record `infeasible` and skip the call. Log tokens, cost, latency, correctness.

### 5.7 Logging schema (`logging.py`)

`data/raw/{experiment}/{run_id}.jsonl`: one `step` record per step, then one `run` record.

**Step record:**
```json
{"type":"step","run_id":"…","step":7,"arm":"cached",
 "context_tokens":5230,"input_tokens_new":412,"cache_read_tokens":4818,"cache_write_tokens":412,
 "output_tokens":61,"ttft_s":0.83,"llm_total_s":1.42,"tool_name":"read_file","tool_arg":"xq3kfa",
 "tool_latency_s":0.002,"inject":{"attempts":2,"types":["detectable","clean"]},
 "label":"ADVANCE","markov_state":"ON","progress":5,
 "rate_limit_wait_s":0.0,"model_id":"<exact id from response>","ts":"…"}
```

**Run record:**
```json
{"type":"run","run_id":"…","experiment":"E4","config_hash":"…","seed":1234,
 "depth":20,"file_tokens":1000,"difficulty":"L2","arm":"naive","f":0.0,"d":null,"R":0,
 "success":true,"termination":"answer_correct|answer_wrong|budget|context_overflow|error",
 "steps":23,"tokens":{…},"cost_usd":0.0312,"wall_s":31.4,"model_id":"…","harness_git_sha":"…"}
```

Rate-limit backoff time is logged separately and **excluded** from latency analysis.

### 5.8 LLM clients (`llm/`)

`LLMClient.complete(messages, tools, cache: bool, stream=True) -> LLMResponse(tool_call|text, usage{input_new, cache_read, cache_write, output}, ttft_s, total_s, model_id)`

- `anthropic_client.py` — real client; streaming to measure TTFT; exponential backoff on 429/5xx, with the wait time recorded.
- `mock.py` — **essential for zero-cost validation**:
  - `PerfectMock`: always follows the valid clue. Success must be 100%, and token counts must exactly match §3.2.
  - `MarkovMock(p, r, e)`: behaves stochastically per the Markov model, with a fake token-usage generator. Used to verify that `fit.py` recovers known p, r, e (estimator validation).
  - `CoupledMock(p0, lam)`: verifies the λ estimator.
  Mocks emit realistic usage fields so the whole pipeline, including figures, runs end-to-end without an API key.

### 5.9 Budget guard (`budget.py`)

Pre-run cost estimate from L1 for the whole experiment config; refuse to start if it exceeds `max_spend_usd` in the config. A running spend meter aborts the experiment when the cap is hit (the partial results are kept and flagged).

### 5.10 Secondary realistic task (`tasks_coding/`, E10)

Small, self-contained Python function-writing tasks with hidden unit tests. Tools: `read_spec`, `write_file`, `run_tests`. Depth is not controlled exactly; it's measured. Purpose: external validity — check that per-step p, δ, and the cached-cost quadratic behave as the synthetic task predicts. 20–30 tasks, kept deliberately simple. If timeline pressure hits, this is the first thing to cut (to a smaller pilot).

---

## 6. Experiments (`configs/experiments/`)

Each experiment is one YAML file. The runner executes it, writes logs, and records the git SHA. Run counts are defaults; the budget guard enforces spend.

| ID | Phase | Purpose | Grid | Runs/cell |
|---|---|---|---|---|
| E0 | Validation | Mock end-to-end; estimator recovery | all mocks | 1,000 (free) |
| E0b | Pilot | Pick difficulty level; measure δ, a, γ, TTFT | n=10, all difficulty levels, file_tokens=1000 | 20 |
| E1 | Calibration | Cost scaling, naive vs cached | n∈{5,10,20,40} × file_tokens∈{200,1000,3000} × {naive, cached} | 10 |
| E2 | Calibration | Context wall | file_tokens=3000, n large enough to overflow | 10 |
| E3 | **Calibration** | Fit p, r, e | n∈{5,10} × file_tokens∈{200,1000,3000}, naive | 50 |
| E4 | **Prediction** | Out-of-sample reliability | n∈{20,40} × same | 50 |
| E5 | Calibration | Fit coupling λ | n=10, T0_pad∈{0, 4k, 16k, 64k} | 50 |
| E6 | Prediction | Summarization cost + reliability | n∈{10,20,40} × k∈{3,5} | 30 |
| E7 | Prediction | Retry ceiling | n=10, f=0.2, d∈{0.2,0.5,0.9}, R∈{0,1,3,6} | 30 |
| E8 | Both | Single-shot baseline | all (n, file_tokens) cells from E1/E3/E4 | 30 |
| E9 | **Prediction (headline)** | Feasible-region classification | ~20 configs sampled near and far from predicted boundaries | 30 |
| E10 | External validity | Realistic coding task | 20–30 tasks × {naive, cached} | 3 |

**Ordering:** E0 → E0b → calibration (E1, E2, E3, E5, calibration half of E8) → fit parameters → **freeze PREREG.md, tag the repo `prereg-v1`** → prediction (E4, E6, E7, prediction half of E8, E9) → E10.

---

## 7. Analysis (`src/loopbounds/analysis/`)

### 7.1 Parameter fitting (`fit.py`)
- δ, a, T₀: from step-record token counts (median + IQR).
- κ check: measured cached-token pricing vs `prices.yaml`.
- Quadratic fit of total input tokens against n per arm (OLS), with the coefficient compared to δ/2 and κδ/2.
- p, r, e: §5.5 estimators with Wilson 95% CIs; bootstrap over runs for parameter CIs.
- λ (and p₀): maximum likelihood on per-step ADVANCE outcomes against context size; exponential vs linear compared by AIC.
- TTFT coefficients α: OLS of TTFT on (T_new, T_cached); τ from decode timing.
- γ, S: from summary-call logs.
- Output: `data/processed/params_fitted.yaml` (plus CIs) → loaded by L1/L2.

### 7.2 Predictions (`predict.py`)
Given fitted params, generate L1 and L2 predictions for every prediction-phase cell **before** looking at prediction-phase data (the script refuses to run if prediction data already exists, unless `--post-hoc` is set, which watermarks the outputs).

### 7.3 Agreement tests (`agreement.py`)

**L1 ↔ L2** (math check, simulation only):
- Mean cost: relative error < 1% at 10⁵ runs.
- Success probability: |L1 − L2| < 3 standard errors.
- Retry clean-delivery rate: matches the closed form within 3 SE.

**L1/L2 ↔ L3** (assumption check). Proposed thresholds, to be finalized in PREREG.md:
- Token cost per run: median relative error < 10%.
- Success probability: predicted value inside the observed Wilson 95% CI in ≥ 80% of prediction cells.
- Markov vs naive pⁿ: Markov has lower Brier score / log-loss on E4.
- Retry ceiling: observed plateau within CI of (1−f)/(1−fd).
- Feasible-region (E9): classification accuracy ≥ 80% overall; also reported separately for points near the boundary.
- Latency: report R² and residual distribution; no hard threshold (API noise).

Every failed check is reported in the paper with a diagnosis of which assumption broke.

### 7.4 Statistics (`stats.py`)
Wilson intervals, bootstrap (10k resamples, seeded), Brier score, log-loss, OLS with HC3 standard errors, and AIC.

---

## 8. Figures and tables (paper mapping)

| ID | Content | Layers |
|---|---|---|
| **Fig 1** | **Feasible region** in (n, p): constraint curves, shaded region, binding-wall labels, boundary shifts for cached / summarized / retries; E9 points overlaid (✓/✗ correct classification) | L1 + L3 |
| Fig 2 | Total input tokens vs n — naive, cached, summarized: model curves + measured points; break-even n* marked | L1 + L3 |
| Fig 3 | Success vs n — observed (CIs), naive pⁿ, Markov DP; calibration vs prediction region shaded | L1 + L3 |
| Fig 4 | p vs context size — fitted coupling curve + E5 points; summarized arm overlaid | L1 + L3 |
| Fig 5 | Retry ceiling: success vs R for each d; closed-form asymptotes dashed | L1 + L2 + L3 |
| Fig 6 | Latency: per-step latency vs context (fit); p95/p99 vs concurrency from simulation | L2 + L3 |
| Fig 7 | Three-layer agreement parity plot (predicted vs observed, all metrics) | L1 + L2 + L3 |
| Tab 1 | Fitted parameters with CIs | — |
| Tab 2 | Prediction errors per cell (L1, L2 vs L3) | — |
| Tab 3 | E9 classification confusion matrix | — |
| Tab 4 | E10 external-validity summary | — |

Each figure script reads only `data/processed/` and `params_fitted.yaml`. **All figures are reproducible without an API key.**

---

## 9. Tests (`tests/`, pytest)

**Model (L1):**
- Markov DP with r=0, e=0, M=n equals pⁿ (to 1e-12).
- Naive cost formula matches brute-force summation.
- Cached cost leading coefficient → κδ/2 as n grows.
- Summarized cost is bounded per step; the recurrence converges to h*.
- Retry formulas: R=0 cases; R→∞ limit; probabilities sum to 1.
- Feasible-region masks are monotone in the expected directions (e.g. larger B never shrinks the region).

**Simulation (L2):** agreement with L1 within tolerance at fixed seeds (§7.3 thresholds).

**Harness (L3):**
- Taskgen: chain integrity, deterministic under a seed, padding within ±5% of target tokens, exactly one valid clue per chain file, decoys reachable only via injection or a wrong read.
- Injection: empirical rates over 10⁵ draws within 3 SE of (f, d); retry count ≤ R.
- Classifier: hand-built traces → expected labels and states.
- `PerfectMock` end-to-end: 100% success; logged tokens exactly equal the L1 naive/cached formulas.
- `MarkovMock` end-to-end: `fit.py` recovers p, r, e within CI on 2,000 mock runs.
- Budget guard aborts on cap.
- Logging schema validated with pydantic.

CI: GitHub Actions runs everything except real-API tests (marked `@pytest.mark.api`, skipped without a key).

---

## 10. Reproducibility and artifact packaging

**Makefile targets:**
```
make setup          # uv sync
make test           # full test suite (no API)
make validate       # E0 mock end-to-end + estimator recovery
make pilot          # E0b (needs API key)
make calibrate      # E1, E2, E3, E5, E8a → fit params
make prereg-check   # refuses prediction runs unless PREREG.md is committed + tagged
make predict        # generate L1/L2 predictions (before prediction data exists)
make run-predict    # E4, E6, E7, E8b, E9
make external       # E10
make analyze        # fitting + agreement tests → data/processed/
make figures        # all figures + tables from data/processed (no API)
make all-offline    # test + validate + figures from released data
```

**Reproducibility requirements:**
- Pinned model ID; the exact ID returned by the API is logged on every call. Run date range recorded in the README.
- Every run is reproducible from (config, seed) except for model sampling nondeterminism, which is documented.
- `prices.yaml` is dated; costs are recomputable under any price table.
- Git SHA recorded in every run record.
- `data/processed/` released on Zenodo with a DOI; the README links it. Raw JSONL is released too if size permits.
- README sections: overview, paper citation, quickstart (offline figure reproduction in 3 commands), full rerun (with API cost estimate), repo map, parameter glossary (§2 table), license.
- `CITATION.cff` with paper metadata.

---

## 11. Build order and milestones

| Day | Deliverable |
|---|---|
| 1 | Repo skeleton, `params.py`, L1 `context.py` / `cost.py` / `reliability.py` / `retry.py` + tests |
| 2 | L1 `latency.py` / `baseline.py` / `feasible.py`; a provisional Fig 1 from assumed parameters |
| 3 | `taskgen.py`, `tools.py`, `injection.py`, `classify.py` + tests |
| 4 | `agent.py`, `arms.py`, mock clients, logging, runner; E0 passes end-to-end |
| 5 | L2 SimPy model; L1↔L2 agreement tests pass |
| 6 | Anthropic client, budget guard; E0b pilot; lock difficulty level |
| 7–8 | Calibration experiments; `fit.py`; freeze PREREG.md, tag `prereg-v1` |
| 9–10 | Prediction experiments (E4, E6, E7, E8b, E9) |
| 11 | E10 external validity (or a reduced pilot) |
| 12–13 | Agreement analysis, all figures and tables |
| 14 | README, Zenodo packaging, final checks |

---

## 12. Open decisions (resolve before Day 6)

1. **Model choice.** A cheap, fast model (e.g. a Haiku-tier model) keeps costs low but may need a higher difficulty level. Pin the exact version.
2. **Window W for the context-wall experiment.** Either use the model's real window (expensive at large δ) or impose an artificial window in the harness (cheaper, but must be disclosed). Recommended: an artificial cap with one real-window confirmation run.
3. **Final pass thresholds** in PREREG.md (§7.3 values are proposals).
4. **Single-shot success model form** (logistic in n and T_ss vs a simpler per-cell empirical rate).
5. **Whether E10 stays** in the main paper or moves to an appendix.
6. **Novelty check.** Complete a focused literature search before Day 1 to confirm the coupled-walls feasible region hasn't been published.
