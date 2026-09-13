# Forex RL Simulator

> A reproducible research scaffold for testing whether an RL policy can add
> value over simple FX trading baselines under the same risk, execution, and
> chronological-evaluation assumptions. Research and education only—not
> investment advice.

## What this project asks

The project studies EUR/USD directional trading with a discrete-action agent:
flat, long, or short. Trades use ATR-derived stops and targets, R-multiple
rewards, fixed fractional risk sizing, and a bounded holding period. The
purpose is not to claim a live or profitable strategy; it is to make the
research path inspectable and repeatable.

No performance result is included in this repository. Full data acquisition,
model training, and historical walk-forward experiments are deliberate
research runs that have not been executed as part of the lightweight checks.

## Architecture

Reusable code lives in `src/forex_rl/`; the notebook is an experiment surface,
not the source of business logic.

- `data.py`, `features.py` — data retrieval and indicator construction.
- `preprocessing.py` — train-fitted z-score scaling.
- `environment.py` — Gym-compatible execution, R rewards, position accounting,
  and SL → TP → time-exit priority.
- `research.py` — deterministic policy backtest, flat/random/trend baselines,
  and opt-in walk-forward orchestration.
- `evaluation.py` and `reporting.py` — shared equity curves, drawdowns, risk
  metrics, turnover/trade counts, and CSV/JSON/HTML artifacts.
- `walk_forward.py` — chronological fold boundaries.

The optional model layer supports DQN, QR-DQN, and recurrent PPO when the
research dependencies are installed.

## Quant methodology and controls

Features are formed from OHLCV history; the current feature row does not use a
future bar. For every walk-forward fold, preprocessing is fitted on the train
slice only and then applied to validation and test slices. Fold summaries retain
train, validation, and test boundaries so in-sample and out-of-sample results
can be reported separately, although they are not fully independent because
adjacent slices share an endpoint. The repository deliberately preserves the established
DateOffset window semantics and execution ordering. Those boundaries are later
selected with inclusive pandas `.loc[start:end]`, so adjacent train/validation/
test slices share endpoint rows; this legacy limitation is disclosed rather
than silently changed.

RL policies and the bundled flat, seeded-random, and EMA-crossover baselines
receive the same ATR stop/target, position-risk fraction, maximum holding
period, `transaction_cost`, and `slippage` configuration. `transaction_cost`
is a proportional **per-side executed-notional** fee; `slippage` is the legacy
absolute adverse adjustment at entry. Both default to zero. Costs are included
in net PnL, realized R, and entry-bar MTM equity. All actions use the same
canonical sequence: open at the current close, then evaluate that bar's SL,
TP, and time exit in order. Make cost calibration an explicit experiment input.

Reports calculate annualized daily-MTM Sharpe and Sortino (252 periods/year,
zero risk-free rate unless supplied), maximum fractional drawdown from running
equity peaks, entry-plus-exit notional turnover, and closed-trade count. These
figures are descriptive evaluation outputs, not model-selection guarantees.

## Installation and lightweight verification

The core package needs only NumPy and pandas. The dev extra includes the test
and lint tools; the research extra adds Gymnasium, market-data, and SB3 stacks.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m compileall -q src tests
python -c "import forex_rl; import forex_rl_simulator"
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest
```

The tests use tiny deterministic in-memory OHLCV fixtures. They cover feature
alignment, train-only scaling, execution ordering, friction, baseline/policy
evaluation consistency, equity/drawdown reporting, and walk-forward boundary
construction. They do not download prices, run the notebook, train an RL
agent, or execute a full historical walk-forward evaluation.

## Running an experiment independently

Install research dependencies with `python -m pip install -e ".[research]"`,
open `Forex_RL_Agent_Final.ipynb`, and set an explicit `ExperimentConfig`.
Record `config.to_dict()` next to artifacts and call `seed_everything()` before
optional model construction. Build folds, train only on each train slice,
select only using its validation slice, and report the held-out test slice
alongside the corresponding baselines. `RUN_FULL_WALKFORWARD` is intentionally
opt-in: it may download data and run substantial model training.

`generate_trade_report()` writes metrics, trade, and equity/drawdown CSVs plus
an HTML summary. `generate_walkforward_report()` writes per-fold CSV/HTML and
aggregate JSON. Store artifacts outside version control.

## Limitations and next research steps

This is a single-instrument, bar-based simulator. Intrabar fill assumptions,
spread/cost calibration, liquidity, data quality, market regimes, and model
instability require sensitivity analysis before any conclusion. The baseline
generators are deliberately simple comparators, not production strategies.
Walk-forward results can still be over-interpreted if researchers tune heavily
against repeated validation/test feedback. No live execution, portfolio
allocation, or financial recommendation is implemented.

See `docs/STEP1_CURRENT_IMPLEMENTATION_MAP.md` for preserved historical
caveats and `docs/STEP2_REFACTOR.md` for the package-refactor map.
