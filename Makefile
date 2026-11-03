# Entry points (design doc §10). Every target is a thin wrapper over the
# `loopbounds` CLI, so they also work without make: `uv run loopbounds <cmd>`.
LB := uv run loopbounds

.PHONY: setup test test-all validate pilot calibrate prereg-check predict run-predict external analyze \
        figures all-offline mock-pipeline estimate

setup:            ## install the locked environment
	uv sync

test:             ## full test suite without API calls (slow Monte Carlo tests skipped)
	uv run pytest -q -m "not slow and not api"

test-all:         ## including slow Monte Carlo tests
	uv run pytest -q -m "not api"

validate:         ## E0 mock end-to-end + estimator recovery
	$(LB) validate

estimate:         ## L1 cost estimate of every experiment (no API calls)
	$(LB) estimate

pilot:            ## E0b (needs ANTHROPIC_API_KEY)
	$(LB) run E0b_pilot -v

calibrate:        ## E1, E2, E3, E5, E8a -> fit params (needs API key)
	$(LB) calibrate -v

prereg-check:     ## refuses prediction runs unless PREREG.md is committed + tagged
	$(LB) prereg-check

predict:          ## L1/L2 predictions (before prediction data exists) + E9 cells
	$(LB) predict

run-predict:      ## E4, E6, E7, E8b, E9 (needs API key and prereg-check)
	$(LB) run-predict -v

external:         ## E10 coding tasks (needs API key)
	$(LB) external -v

analyze:          ## fitting + agreement tests -> data/processed/
	$(LB) analyze

figures:          ## all figures + tables from data/processed (no API)
	$(LB) figures

mock-pipeline:    ## every stage offline with mock clients -> data/mock/
	$(LB) mock-pipeline

all-offline: test validate figures
