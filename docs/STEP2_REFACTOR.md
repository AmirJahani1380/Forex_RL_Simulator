# Step 2: package refactor

## Architecture

Reusable implementation now lives under `src/forex_rl/`:

- `config.py` provides an explicit immutable experiment configuration.
- `data.py`, `features.py`, and `preprocessing.py` contain data acquisition,
  indicators, and train-only scaling.
- `environment.py` contains `ForexEnv`, its Gymnasium/SB3 flattening wrapper,
  and train-only-scaled vector environment builders.
- `walk_forward.py` constructs chronological fold dictionaries.
- `evaluation.py` contains the R and daily-MTM metric helpers.
- `models.py` preserves model-family hyperparameters and R-based validation
  selection behind an optional Stable-Baselines dependency boundary.
- `research.py` provides deterministic backtesting, baselines, and opt-in
  walk-forward orchestration; `reporting.py` provides inspectable report files.

`src/forex_rl_simulator/baseline.py` is a compatibility facade for the Step 1
test/import path. It contains no duplicated implementation. The notebook now
only configures and composes package components; it no longer embeds core
implementations, while retaining explicit opt-in training/walk-forward hooks.

## Behavioral preservation

No quantitative methodology was changed. In particular, the moved logic keeps
the existing indicator formulas, sample-standard-deviation scaling, train-only
scaler fitting, DateOffset fold boundaries, index-reset environment logging,
action timing, position sizing, SL-before-TP-before-time ordering, slippage,
and currently inert `transaction_cost` parameter. The existing caveats in the
Step 1 map remain intentionally documented rather than silently corrected.

## Deliberately not validated

No market-data download, RL training, notebook execution, full walk-forward
run, or former 400,000-step "smoke" run was executed. Those require external
data and/or materially expensive training and are not refactor validation.
