# Forex RL — Refined R‑Reward

> End‑to‑end reinforcement learning (RL) pipeline for EUR/USD with **R‑multiple rewards**, ATR‑based risk, and **walk‑forward cross‑validation**.  
> Built around a custom `gymnasium` environment + Stable‑Baselines3 (DQN/QR‑DQN/RPPO), robust feature engineering, and rich reporting.

<p align="center">
  <em>Research only — not financial advice.</em>
</p>

---

## Highlights

- **R‑multiple reward**: returns measured in risk units (R) defined by ATR‑based stop size.
- **Custom `ForexEnv`** (Gymnasium):  
  - Observation: window of normalized features, shape = `(window_size, n_features)`  
  - Action space: `Discrete(3)` → **0: flat**, **1: long**, **2: short**  
  - Exits: priority **SL → TP → time‑exit**, optional slippage/fees.
- **Robust data pipeline** (via `yfinance`) with **EMA, RSI, MACD, Bollinger Bands, ATR**, returns, etc.
- **Scaler** fit on **train only**, applied to val/test to prevent leakage.
- **Walk‑forward CV** with configurable train/val/test windows and step size.
- **Algorithms**: `DQN`, `QRDQN` (quantile regression DQN), `RPPO` (LSTM). Easy to switch.
- **Metrics & reports**: trade list in R, daily MTM, Sharpe/Sortino/MaxDD,
  and explicit local CSV/JSON/HTML report and ZIP helpers.

---

## Repo Contents

- `src/forex_rl/` — reusable package: config, data, features, preprocessing,
  environment, evaluation, walk-forward boundaries, and optional model imports.
- `Forex_RL_Agent_Final.ipynb` — a thin experiment interface that imports the
  package. It prepares data and an environment but does not launch training.
- `tests/test_baseline_characterization.py` — compact regression tests for the
  established trading and data-processing semantics.

Install the dependency-light core with `pip install -e .`; use
`pip install -e .[research]` only when data download or SB3 training is needed.
See `docs/STEP1_CURRENT_IMPLEMENTATION_MAP.md` for intentionally preserved
methodological caveats.

---

## Quickstart

### Option A — Google Colab
1. Clone/upload the complete repository, not just the notebook.
2. In a notebook cell, install it with `!pip install -e ".[research]"`.
3. In **Config (edit here)**, set symbol/dates and hyper‑parameters. Defaults are:
   ```python
   SYMBOL = "EURUSD=X"
   START_DATE = "2015-01-01"
   END_DATE   = "2025-08-01"
   ```
4. Run the notebook cells in order through **Data & Features** → **Scaler & Walk‑Forward** → **Env** → **Training**.
5. Set `RUN_FULL_WALKFORWARD = True` only when you intentionally want training.
6. The opt-in runner writes local walk-forward CSV/JSON/HTML outputs; call the
   explicit ZIP helper if an archive is needed.

### Option B — Local (Linux/Mac/Windows)
```bash
# 1) Create env (recommended)
python -m venv .venv && source .venv/bin/activate   # (Windows) .venv\Scripts\activate

# 2) Install this repository and its optional research dependencies
pip install -e ".[research]"

# 3) Launch Jupyter and open the notebook
pip install jupyterlab
jupyter lab
```
The notebook exposes the same workflow as explicit stages: data/features,
leakage-safe vector-environment setup, optional model construction, and an
opt-in full walk-forward runner. It deliberately does not train merely by being
opened or imported.

### Clean install and lightweight verification

For development and CI-safe verification, install only the small development
extra. It does not install Gymnasium/SB3, download market data, or run training.

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

The test suite includes a tiny deterministic in-memory smoke path covering
OHLCV data, feature construction, train-only scaling, the environment, a
prediction-only policy interface, and metric evaluation. It intentionally uses
no meaningful model training. GitHub Actions runs only these lightweight checks;
never add full RL training, market-data downloads, notebook runs, or complete
historical walk-forward evaluation to CI.

For reproducible scripts, build an immutable `ExperimentConfig` or deserialize
one explicitly with `ExperimentConfig.from_mapping(...)`, record
`config.to_dict()` beside results, and call `seed_everything(config.random_seed)`
before constructing supported optional model interfaces. `make_model` continues
to pass the same seed to SB3 models. `configure_logging()` provides concise
package-level progress logs without changing the application's root logger.

---

## Data & Features

- Data source: `yfinance` (e.g., **EURUSD=X**, auto‑adjusted).  
- Indicators (subset used in the notebook): `ema20`, `rsi`, `macd`, `macd_signal`, `macd_hist`,  
  Bollinger (`bb_upper`, `bb_middle`, `bb_lower`), `atr`/`atr_n`, and `ret`.  
- The notebook includes safeguards to flatten `yfinance` MultiIndex columns and clean/align OHLCV.

---

## Environment Details

- **Observation**: last `WINDOW_SIZE` rows of z‑scored features → `Box(low=-inf, high=inf, shape=(WINDOW_SIZE, N_FEATURES))`  
- **Actions**: `0=flat, 1=long, 2=short` (no manual close; exits handled by rules)  
- **Reward**: realized PnL in **R** where `R = price risk / stop size`.  
- **Exits** (intraday priority): **stop‑loss** → **take‑profit** → **time‑based**; configurable ATR multipliers.  
- **Friction**: `slippage` changes entry prices. `transaction_cost` is retained
  for configuration compatibility but is currently inert; this preserved
  limitation does not simulate costs when set above zero.
- **Penalties**: optional invalid‑action penalty, drawdown penalty, turnover penalty.

---

## Walk‑Forward Training

- Build folds with `(train → val → test)` windows and **step‑ahead** increments.  
- **Scaler** is fit on **train** only, then applied to val/test.  
- Models are trained on **train** and selected using **R‑based validation** (risk‑aware metrics).  
- Final backtest is run on **test**, and consolidated in an **HTML report**.

---

## Algorithms

Switch via the `ALGO` config (examples):
```python
ALGO = "DQN"         # vanilla DQN
ALGO = "QRDQN"       # quantile-regression DQN (risk-aware)
ALGO = "RPPO"          # PPO with LSTM policy
```
- Custom `policy_kwargs` are provided for each (MLP and LSTM).  
- Vectorized environments via `DummyVecEnv` / `SubprocVecEnv`.  
- Reproducible seeds are set for NumPy, Python, and SB3.

---

## Key Configuration (edit in the Config cell)

```python
# Asset & range
SYMBOL = "EURUSD=X"
START_DATE = "2015-01-01"
END_DATE   = "2025-08-01"

# Features & window
WINDOW_SIZE = 64            # lookback length
POSITION_RISK_FRAC = 0.01   # position sizing (fractional risk)

# Risk/exit
SL_ATR_MULT = 1.8           # stop-loss ATR multiple
TP_ATR_MULT = 2.2           # take-profit ATR multiple
MAX_BARS_IN_TRADE = 18      # time-based exit

# Friction & penalties
TRANSACTION_COST = 0.0
SLIPPAGE = 0.0
INVALID_ACTION_PENALTY = 0.0
DD_PENALTY = 0.0
TURNOVER_PENALTY = 0.0

# SB3
ALGO = "DQN"                # or "QRDQN", "RPPO"
TOTAL_TIMESTEPS = 400_000
EVAL_FREQ = 120_000
N_ENVS = 24                 # vectorized envs
WF_N_ENVS = N_ENVS

# Walk-forward (example)
MIN_TRAIN_MONTHS = 24
VAL_MONTHS = 9
TEST_MONTHS = 6
STEP_MONTHS = 12
```

> Tip: Use the synthetic test suite for refactor validation. Any training budget,
> including a small experiment, is a deliberate research run rather than a test.

---

## Outputs & Reports

The notebook writes artifacts under a run directory (e.g., `logs_rl/`), including:
- Selected model checkpoints and training logs when training is explicitly run.
- `reports/`: `wf_results.csv`, `wf_aggregates.json`, and `wf_report.html`.
- Use `zip_directory` explicitly to create a local report archive. No Drive-copy
  helper or one-click training/smoke workflow is included.

---

## Reproducibility

- Deterministic seeds are set for Python/NumPy/SB3.  
- The scaler is re‑fit on each fold’s training split only.  
- The environment is stateless across episodes apart from the current position and window buffer.

---

## Roadmap (ideas)

- Transaction cost/slippage calibration and sensitivity analysis  
- Multi‑instrument training (e.g., other FX pairs)  
- Regime features (volatility state, dollar index, macro calendar)  
- Hyper‑parameter sweeps and Optuna integration  
- Ensemble of policies across folds

---

## Disclaimer

This repository is for **research/education**. Markets involve risk. **No financial advice**.

---

## Citation

If you find this helpful, please cite or star the repo. Libraries used include:  
`gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch`, `numpy`, `pandas`, `yfinance`, `finta`, `mplfinance`, `matplotlib`.

---

## License

MIT — feel free to use and adapt. (Change this if your project uses a different license.)
