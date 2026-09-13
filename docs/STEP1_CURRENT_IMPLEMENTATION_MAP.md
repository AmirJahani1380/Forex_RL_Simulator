# Step 1: current implementation map and characterization scope

## Source of truth and scope

`Forex_RL_Agent_Final.ipynb` is the current executable research pipeline. It is
not executed in this step. `src/forex_rl_simulator/baseline.py` is a
dependency-light, formula-preserving characterization extract of notebook cells
6, 8, 9, and the metric helpers in cell 11. It exists solely to make tiny,
deterministic tests possible without downloading market data or training models;
the notebook remains the training entry point.

## Pipeline map

| Concern | Current implementation | Observed semantics |
| --- | --- | --- |
| Data and features | Cells 6 (`download_eurusd`, `compute_indicators`) | yfinance EUR/USD OHLCV; normalize possible MultiIndex columns; sort and drop null rows. Features are EMA20, RSI14, MACD(12,26,9), Bollinger(20,2), ATR14/close, and close return. `dropna()` establishes the feature/timestamp alignment. |
| Scaling | Cells 8, 10, 13 (`ZScoreScaler`) | Column mean and sample standard deviation (`pandas.std`, `ddof=1`) are fitted to `train_df[feature_cols]`; zero std is replaced with `1e-8`; validation/test are transformed using that same scaler. |
| Walk-forward | Cell 8 (`build_walkforward_folds`) | DateOffset boundaries: train ends/validation starts at `test_start - val_months`; validation ends/test starts at `test_start`; fold excluded when `test_end > df.index.max()`. Current config defaults to a rolling 36-month train history, with start clipped to data start. Pandas `.loc[start:end]` later makes adjacent endpoint rows inclusive in train/val/test slices. |
| Observation/action | Cell 9 (`ForexEnv`, `ForexEnvSB3`) | Initial observation is `feat.iloc[0:window_size]`, float32, shape `(window, features)`; `t=window_size`. Actions are 0 hold, 1 open long, 2 open short, and there is no manual close. SB3 wrapper flattens to `window*features`. |
| Position/PnL/reward | Cell 9 | Entry uses current bar close plus slippage for long / minus for short. Units target `net_worth * position_frac` risk at the ATR stop. Exit reward is realized PnL divided by stop-defined risk (R), then configured drawdown/turnover penalties. Net worth changes by absolute `profit_value`. |
| Exit event order | Cell 9; Step 4 package execution | The original environment handles an action then checks SL, TP, then time on that same bar; a bar touching both records SL. Step 4 routes baselines through that same canonical ordering. This remains an intrabar assumption: a close-priced entry can see that bar's high/low, including prices that can precede the close. |
| Costs/slippage | Step 4 package execution | `slippage` changes entry price. `transaction_cost` is now a proportional per-side executed-notional fee applied at entry and exit to policy and baseline accounting; default config sets both to 0. |
| Models/selection | Cells 12–13 | DQN/QRDQN/RecurrentPPO interfaces are constructed with seeded SB3 configs. Each fold fits only its training scaler. Evaluation is deterministic and best model selection uses validation R metrics with PF/trade/DD gates. |
| Evaluation | Cells 11–14; Step 4 package reporting | R totals, win rate, profit factor, R drawdown/MAR, daily close mark-to-market, Sharpe, Sortino, max drawdown, and extended report metrics. Baselines are flat, deterministic seeded random signals, and EMA crossover signals. Step 4 reporting resolves preserved positional environment timestamps and records entry/exit fees in MTM equity. |

## Characterization coverage

`tests/test_baseline_characterization.py` uses four-to-25-row synthetic frames
to cover reset/observations/actions, long and short PnL, SL/TP/time/EOD ordering,
slippage and transaction-cost accounting, scaler fit provenance,
walk-forward boundary construction, indicator index alignment, deterministic
environment RNG construction, and R/drawdown/Sharpe metrics.

## Suspected methodology issues documented but not changed

1. Fold dictionary boundaries touch, while downstream inclusive `.loc` slicing
   includes the shared endpoint in adjacent train/validation/test frames. This
   could duplicate boundary observations across phases. It is preserved because
   changing it would alter the current research methodology.
2. The notebook's `SMOKE_TEST=True` uses 400,000 timesteps despite its comment
   calling that small. It is not run here and must not be treated as lightweight.
3. Same-bar open/SL/TP/time evaluation is an intrabar execution assumption: a
   close-priced entry can be tested against that bar's high/low, including prices
   that may have occurred before the close. Step 4 makes this shared rather than
   a baseline/environment inconsistency.
4. Random and trend baseline generators set an internal `in_pos` on first entry
   and never clear it, so each emits at most one entry signal.
5. Configured `MIN_TRADES_VAL=4` is unused: `RMetricsEvalCallback` applies a
   hard-coded `trades >= 6` selection gate instead.
