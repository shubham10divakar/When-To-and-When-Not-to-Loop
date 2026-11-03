# Pre-registration — "When To and When Not to Loop"

**Status:** DRAFT. Finalize every `TBD` below after calibration, then commit this
file, `thresholds.yaml`, `configs/experiments/` (including
`E9_cells.generated.yaml`) and `data/processed/predictions_*.csv`, and tag:

```bash
git add preregistration configs/experiments data/processed/predictions_*.csv data/processed/params_fitted.yaml
git commit -m "Pre-registration v1"
git tag -a prereg-v1 -m "Pre-registration v1"
git push && git push origin prereg-v1
```

`make run-predict` refuses to start unless `make prereg-check` passes: the tag
exists, and nothing under `preregistration/` or `configs/experiments/` changed
after it.

## 1. Hypotheses

- **H1 (cost wall).** Under naive and cached context strategies, total
  price-weighted input tokens grow quadratically in the number of calls, with
  leading coefficients δ/2 and κ·δ/2; under summarization, growth is linear.
- **H2 (reliability wall).** P(success) of the agent loop is predicted by the
  ON/DER Markov model (with coupling p(T) if λ is significant) better than by
  independent-step pⁿ.
- **H3 (coupling).** Per-step advance probability decreases with context size.
- **H4 (retry ceiling).** Tool-layer retries raise clean delivery only to
  (1−f)/(1−fd); success plateaus accordingly.
- **H5 (feasible region).** The L1 model classifies held-out configurations as
  inside/outside the agent's feasible region (cost budget, latency SLO,
  minimum success rate, context window) with ≥ 80% accuracy.

## 2. Model, task and settings (locked)

| Item | Value |
|---|---|
| Model | `claude-haiku-5-5` (exact id returned by the API is logged on every call) |
| Thinking / effort | `adaptive` / `low` (sampling parameters such as temperature are not accepted by current models) |
| Max output tokens per step | 1024 |
| Difficulty level (from E0b) | TBD (target p ∈ [0.85, 0.97]) |
| Decoy chains / fake clues | 2 / 2 |
| Step budget | M = 3n |
| Artificial window for E2 | 64,000 tokens (one real-window confirmation run, E2_context_wall_real) |
| E9 constraints | budget $0.02/task, latency SLO 120 s (p95 of run wall time), s_min 0.8 — TBD after calibration |

## 3. Estimators (locked)

Step labels and the Markov state assignment are defined in
`src/loopbounds/harness/classify.py`. Over chase-phase steps (progress < n):

- p̂ = #ADVANCE-from-ON / #steps-from-ON
- r̂ = #ADVANCE-from-DER / #steps-from-DER
- ê = #ANSWER_WRONG-from-DER / #steps-from-DER

REDUNDANT, LIST and NOOP count as non-advancing steps in their current state.
CORRUPTED (silent injected failure on the correct file) moves the agent to DER.
p, r, e are fitted on E3 only (no injection). λ, p0 are fitted on E5 by maximum
likelihood. Exponential vs linear coupling is chosen by AIC, and coupling is enabled in
predictions only if λ̂ > 2·SE(λ̂). δ and T0 are fitted as linear functions of
file size and T0 padding.

## 4. Pass criteria

See `thresholds.yaml` (values copied here at freeze time):

| Check | Criterion |
|---|---|
| L1 ↔ L2 cost | relative error < 1% at 10⁵ simulated runs |
| L1 ↔ L2 success | \|L1 − L2\| < 3 SE |
| L3 cost | median over prediction cells of relative error < 10% |
| L3 success | prediction inside the observed Wilson 95% CI in ≥ 80% of prediction cells |
| Markov vs pⁿ | Markov Brier score < pⁿ Brier score on E4 |
| Retry ceiling | observed clean-delivery plateau within CI of (1−f)/(1−fd) |
| Feasible region (E9) | overall accuracy ≥ 80%; also reported near vs far from the boundary |
| Latency | R² and residuals reported; no hard threshold |

Every failed check is reported in the paper with a diagnosis of the assumption
that broke.

## 5. Prediction-phase experiments

E4, E6, E7, E8b, E9. Predictions are generated with `make predict` **before**
any prediction-phase data exists (the script refuses otherwise) and committed
with this file.

## 6. Deviations

Any deviation from this plan after the tag is listed here with its date and reason.
