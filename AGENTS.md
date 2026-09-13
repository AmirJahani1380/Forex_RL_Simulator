# AGENTS.md

## Project purpose

This repository is a quantitative research and reinforcement-learning trading project. Refactors must improve structure, maintainability, reproducibility, and testability without silently changing the research methodology or trading semantics.

## Core safety rule: do not run expensive training

- Do NOT run full RL training.
- Do NOT run the complete historical walk-forward pipeline.
- Do NOT execute long notebook runs over the full dataset.
- Do NOT launch large DQN, QR-DQN, PPO, or RecurrentPPO training jobs.
- Do NOT use expensive validation merely to prove that a refactor works.

Use lightweight validation instead.

## Allowed validation

Prefer the smallest checks that give useful evidence:

- `python -m compileall` on project source files.
- Import checks for all project modules.
- Unit tests with `pytest`.
- Static analysis, formatting, linting, and type checking when configured.
- Tiny synthetic datasets.
- Small deterministic fixtures.
- Environment reset/step tests.
- Very short model smoke tests only when required to verify interfaces or execution paths.
- Mock or stub model training when practical.

A smoke test must be deliberately small and must not become a real training run.

Never claim that full training or full walk-forward behavior was verified unless it was actually executed.

## Preserve quantitative behavior

Refactor first; change methodology only when a specific bug is demonstrated.

Preserve, unless an explicitly documented bug requires a correction:

- feature definitions;
- observation construction;
- action semantics;
- reward calculations;
- position sizing;
- PnL/accounting logic;
- stop-loss and take-profit behavior;
- exit priority/order;
- transaction-cost and slippage semantics;
- timestamps and bar alignment;
- train/validation/test boundaries;
- walk-forward window semantics;
- scaler fitting behavior;
- model-selection logic;
- evaluation metric definitions.

If a refactor would alter any of these, stop and document the proposed behavioral change before implementing it.

## Quant research correctness

Treat data leakage and unrealistic execution assumptions as critical defects.

In particular:

- Never fit scalers, encoders, feature transforms, or learned preprocessing on validation/test data.
- Never use future observations when constructing current features or rewards.
- Preserve chronological train/validation/test separation.
- Avoid look-ahead bias in indicators, labels, exits, and evaluation.
- Keep transaction costs and slippage explicit and configurable.
- Apply the same evaluation assumptions to strategy and baseline comparisons.
- Keep in-sample and out-of-sample results clearly separated.

If an existing methodological problem is discovered, document it clearly. Do not silently rewrite the research logic while performing a structural refactor.

## Refactoring expectations

Prefer a conventional Python package under `src/` with reusable modules rather than business logic embedded in notebooks.

The notebook should become a thin research/demo interface that imports reusable project code.

Avoid:

- duplicated core logic;
- hidden global state;
- hard-coded machine-specific paths;
- unnecessary abstractions;
- large unrelated rewrites;
- changing public behavior purely for style.

Keep configuration explicit and reproducible.

## Testing priorities

Tests should focus on financially and methodologically sensitive behavior, especially:

- reward and PnL calculations;
- long/short accounting;
- SL/TP/time-exit ordering;
- transaction costs and slippage;
- environment reset/step behavior;
- feature alignment;
- walk-forward split boundaries;
- train-only scaler fitting;
- deterministic configuration/seed handling;
- metric calculations.

Use tiny fixtures or synthetic OHLC data so tests remain fast.

## Dependency and packaging rules

- Keep dependencies minimal and explicit.
- Prefer `pyproject.toml` for package configuration.
- Do not introduce a dependency solely to save a few lines of code.
- Keep the project importable after a fresh installation.
- Avoid committing generated model artifacts, large datasets, caches, or experiment outputs.

## Git workflow

- Work on a separate branch.
- Do not commit directly to `main` or `master`.
- Preserve unrelated existing changes.
- Keep commits logically scoped.
- Before finalizing, inspect the complete diff for accidental behavioral changes.
- Open a PR rather than merging automatically unless the user explicitly requests otherwise.

## Required validation report

For every substantial refactor, report:

1. files created, modified, and deleted;
2. architectural changes;
3. any behavior or methodology changes;
4. exact validation commands executed;
5. whether each command passed or failed;
6. what was intentionally not run because it would require expensive training;
7. remaining risks or areas that still require a real research run.

## Acceptance standard

A refactor is acceptable only when:

- source files compile/import successfully;
- relevant lightweight tests pass;
- no known train/validation/test leakage is introduced;
- expensive training was not run unless explicitly requested;
- original trading/research semantics are preserved unless a documented bug was fixed;
- notebook logic has been reduced where appropriate in favor of reusable modules;
- changes are reviewable and limited to the requested scope;
- documentation accurately distinguishes verified behavior from behavior that still requires full training to validate.
