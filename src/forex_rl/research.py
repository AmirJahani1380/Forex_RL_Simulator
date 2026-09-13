"""Backtesting, baseline comparisons, and opt-in walk-forward orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .environment import ForexEnv
from .evaluation import R_metrics, _daily_sharpe, _max_drawdown, daily_mtm_from_trades
from .preprocessing import ZScoreScaler


def _simulate_trades_from_signals(df, signals, position_frac, sl_atr_mult, tp_atr_mult, max_bars_in_trade):
    """Original baseline simulator: exits first, then opens; no slippage/costs."""
    equity, trades, position, entry = 1.0, [], 0, None
    for t, (_, row) in enumerate(df.iterrows()):
        if position:
            entry["bars"] += 1
            if position == 1:
                exit_price = entry["stop"] if row.low <= entry["stop"] else entry["take"] if row.high >= entry["take"] else row.close if entry["bars"] >= max_bars_in_trade else None
            else:
                exit_price = entry["stop"] if row.high >= entry["stop"] else entry["take"] if row.low <= entry["take"] else row.close if entry["bars"] >= max_bars_in_trade else None
            if exit_price is not None:
                pnl = (exit_price - entry["price"]) * entry["units"] if position == 1 else (entry["price"] - exit_price) * entry["units"]
                risk = abs(entry["price"] - entry["stop"]) * entry["units"]
                realized = pnl / risk if risk > 1e-12 else 0.0
                equity = max(1e-12, equity + realized * position_frac * equity)
                trades.append({"type": "long" if position == 1 else "short", "entry_t": entry["timestamp"], "exit_t": df.index[t], "entry_price": entry["price"], "exit_price": float(exit_price), "stop": entry["stop"], "take": entry["take"], "units": entry["units"], "realized_R": float(realized), "pnl_value": float(pnl), "reason": "sl" if exit_price == entry["stop"] else "tp" if exit_price == entry["take"] else "time", "bars_in_trade": entry["bars"]})
                position, entry = 0, None
        if not position and t < len(df) - 1 and int(signals.iloc[t]) in (1, 2):
            position, price = (1 if int(signals.iloc[t]) == 1 else -1), float(row.close)
            distance, units = float(row.atr) * max(1e-12, sl_atr_mult), position_frac * equity / (float(row.atr) * max(1e-12, sl_atr_mult))
            entry = {"timestamp": df.index[t], "price": price, "stop": price - distance if position == 1 else price + distance, "take": price + float(row.atr) * tp_atr_mult if position == 1 else price - float(row.atr) * tp_atr_mult, "units": units, "bars": 0}
    if position:
        exit_price = float(df.close.iloc[-1]); pnl = (exit_price - entry["price"]) * entry["units"] if position == 1 else (entry["price"] - exit_price) * entry["units"]
        risk = abs(entry["price"] - entry["stop"]) * entry["units"]; realized = pnl / risk if risk > 1e-12 else 0.0
        trades.append({"type": "long" if position == 1 else "short", "entry_t": entry["timestamp"], "exit_t": df.index[-1], "entry_price": entry["price"], "exit_price": exit_price, "stop": entry["stop"], "take": entry["take"], "units": entry["units"], "realized_R": float(realized), "pnl_value": float(pnl), "reason": "eod", "bars_in_trade": entry["bars"]})
    return pd.DataFrame(trades)


def baseline_flat(df):
    return pd.Series(0, index=df.index)


def baseline_random(df, p_enter=.03, seed=123):
    """Step 1 generator, including its one-entry ``in_pos`` behavior."""
    rng, signal, in_pos = np.random.default_rng(seed), np.zeros(len(df), dtype=int), 0
    for index in range(len(df) - 1):
        if in_pos == 0 and rng.uniform() < p_enter:
            signal[index] = 1 if rng.uniform() < .5 else 2
            in_pos = signal[index]
        elif in_pos != 0:
            signal[index] = 0
    return pd.Series(signal, index=df.index)


def baseline_trend(df, fast=50, slow=200):
    """Step 1 EMA crossover generator, including its one-entry behavior."""
    fast_ma = df["close"].ewm(span=fast, adjust=False).mean()
    slow_ma = df["close"].ewm(span=slow, adjust=False).mean()
    crossings, signal, in_pos = (fast_ma > slow_ma).astype(int).diff().fillna(0), np.zeros(len(df), dtype=int), 0
    for index in range(len(df) - 1):
        if in_pos == 0:
            if crossings.iloc[index] == 1:
                signal[index], in_pos = 1, 1
            elif crossings.iloc[index] == -1:
                signal[index], in_pos = 2, 2
        else:
            signal[index] = 0
    return pd.Series(signal, index=df.index)


def run_baselines(df, config):
    outcomes = {}
    for name, signals in {"flat": baseline_flat(df), "random": baseline_random(df), "trend": baseline_trend(df)}.items():
        trades = _simulate_trades_from_signals(df, signals, config.position_risk_frac, config.sl_atr_mult, config.tp_atr_mult, config.max_bars_in_trade)
        mtm = daily_mtm_from_trades(df, trades, config.position_risk_frac)
        outcomes[name] = {"trades": trades, "r": R_metrics(trades), "mtm": mtm,
                          "sharpe": _daily_sharpe(mtm.daily_ret.values) if len(mtm) else np.nan,
                          "mdd_pct": _max_drawdown(mtm.equity_pct.values) if len(mtm) else np.nan}
    return outcomes


def backtest_model(model, scaler, df_slice, feature_cols, price_cols, config):
    """Deterministic policy rollout using preserved environment mechanics."""
    features = scaler.transform(df_slice[feature_cols])
    env = ForexEnv(df_slice[price_cols], features, seed=config.random_seed, **config.environment_kwargs)
    obs, done = env.reset(seed=config.random_seed)[0], False
    state, episode_start = None, np.ones((1,), dtype=bool)
    while not done:
        if config.algo.upper() == "RPPO":
            action, state = model.predict(obs.reshape(1, -1), state=state, episode_start=episode_start, deterministic=True)
            episode_start = np.array([False])
        else: action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, _, done, _, _ = env.step(int(np.asarray(action).reshape(-1)[0]))
    trades = env.get_trade_log(); mtm = daily_mtm_from_trades(df_slice, trades, config.position_risk_frac)
    return {"trades": trades, "r": R_metrics(trades), "mtm": mtm,
            "sharpe": _daily_sharpe(mtm.daily_ret.values) if len(mtm) else np.nan,
            "maxdd_pct": _max_drawdown(mtm.equity_pct.values) if len(mtm) else np.nan}


def run_walkforward(algo, df_all, folds, config, train_fold, feature_cols, price_cols):
    """Opt-in orchestration; caller supplies training, so this function never trains implicitly."""
    rows = []
    for fold in folds:
        train = df_all.loc[fold["train_start"]:fold["train_end"]].copy(); val = df_all.loc[fold["val_start"]:fold["val_end"]].copy(); test = df_all.loc[fold["test_start"]:fold["test_end"]].copy()
        model, scaler, save_dir = train_fold(algo, train, val, config, feature_cols, price_cols)
        result, baselines = backtest_model(model, scaler, test, feature_cols, price_cols, config), run_baselines(test, config)
        rows.append({"fold": fold["fold"], "total_R": result["r"]["total_R"], "PF": result["r"]["profit_factor"], "trades": result["r"]["trades"], "sharpe": result["sharpe"], "maxDD_pct": result["maxdd_pct"], "exit_mix": json.dumps(result["trades"].reason.value_counts().to_dict()) if len(result["trades"]) else "{}", **{f"{name}_R": value["r"]["total_R"] for name, value in baselines.items()}})
    return pd.DataFrame(rows)
