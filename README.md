# Forex RL — Refined R‑Reward

> End‑to‑end reinforcement learning (RL) pipeline for EUR/USD with **R‑multiple rewards**, ATR‑based risk, and **walk‑forward cross‑validation**.  
> Built around a custom `gymnasium` environment + Stable‑Baselines3 (DQN/QR‑DQN/RecurrentPPO), robust feature engineering, and rich reporting.

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
- **Algorithms**: `DQN`, `QRDQN` (quantile regression DQN), `RecurrentPPO` (LSTM). Easy to switch.
- **Metrics & reports**: trade list in R, daily MTM, Sharpe/Sortino/MaxDD, underwater plots, histograms, monthly heatmap.  
  One‑click **smoke test** + full **HTML report** (with ZIP bundle).

---

## Repo Contents

- `Forex_RL_Agent_Final.ipynb` — the main notebook (Colab/local ready).  
  It contains:
  1) Install & imports  
  2) **Config (edit here)**  
  3) Data & indicators  
  4) Scaler & walk‑forward splits  
  5) **`ForexEnv` with R‑reward** (+ SB3 wrapper)  
  6) VecEnv builder (Dummy/Subproc)  
  7) Metrics & plotting  
  8) Training & evaluation (R‑based model selection)  
  9) **Smoke test**, **Full walk‑forward**, **HTML report & ZIP**, **Copy to Drive**

---

## Quickstart

### Option A — Google Colab
1. Upload/open `Forex_RL_Agent_Final.ipynb` in Colab (enable **GPU**).  
2. Run **Install & Imports**.  
3. In **Config (edit here)**, set symbol/dates and hyper‑parameters. Defaults are:
   ```python
   SYMBOL = "EURUSD=X"
   START_DATE = "2015-01-01"
   END_DATE   = "2025-08-01"
   ```
4. Run the notebook cells in order through **Data & Features** → **Scaler & Walk‑Forward** → **Env** → **Training**.
5. (Optional) Run **Smoke Test** to sanity‑check the full pipeline quickly.
6. Run **Full Walk‑Forward CV**.
7. Run **HTML Report & ZIP** to generate artifacts under `logs_rl/reports`.  
8. (Colab) Run **Save results in drive** to copy `/content/logs_rl` → `MyDrive/results/logs_rl`.

### Option B — Local (Linux/Mac/Windows)
```bash
# 1) Create env (recommended)
python -m venv .venv && source .venv/bin/activate   # (Windows) .venv\Scripts\activate

# 2) Install deps
pip install numpy pandas matplotlib mplfinance yfinance finta gymnasium torch stable-baselines3 sb3-contrib

# 3) Launch Jupyter and open the notebook
pip install jupyterlab
jupyter lab
```
Then run the cells as in the Colab flow.

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
- **Friction**: `transaction_cost`, `slippage` (default 0.0; set >0 to simulate costs).  
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
ALGO = "RecurrentPPO"  # PPO with LSTM policy
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
SL_ATR_MULT = 2.0           # stop-loss ATR multiple
TP_ATR_MULT = 2.5           # take-profit ATR multiple
MAX_BARS_IN_TRADE = 30      # time-based exit

# Friction & penalties
TRANSACTION_COST = 0.0
SLIPPAGE = 0.0
INVALID_ACTION_PENALTY = 0.0
DD_PENALTY = 0.0
TURNOVER_PENALTY = 0.0

# SB3
ALGO = "QRDQN"              # or "DQN", "RecurrentPPO"
TOTAL_TIMESTEPS = 1_200_000
EVAL_FREQ = 140_000
N_ENVS = 8                  # vectorized envs
WF_N_ENVS = N_ENVS

# Walk-forward (example)
MIN_TRAIN_MONTHS = 36
VAL_MONTHS = 6
TEST_MONTHS = 6
STEP_MONTHS = 6
```

> Tip: Start with the **Smoke Test** to validate everything end‑to‑end, then switch to your full budget for WF.

---

## Outputs & Reports

The notebook writes artifacts under a run directory (e.g., `logs_rl/`), including:
- `best_model.zip` per fold, `vecnormalize.pkl` (if used), and training logs.  
- `reports/`:  
  - `wf_results.csv` (fold‑level summary)  
  - Per‑fold PNGs (equity curve, underwater, histograms, monthly heatmap)  
  - `rl_report_smoke.html` (from the smoke test, when enabled)  
  - `wf_report_YYYYMMDD_HHMM.html` + a **ZIP** bundle for sharing

On Colab, a helper cell copies `/content/logs_rl` → `MyDrive/results/logs_rl`.

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
