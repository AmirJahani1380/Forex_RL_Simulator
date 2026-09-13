"""Backtesting, baseline comparisons, and opt-in walk-forward orchestration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .environment import ForexEnv
from .evaluation import R_metrics, daily_mtm_from_trades, performance_metrics


def _simulate_trades_from_signals(
    df,
    signals,
    position_frac,
    sl_atr_mult,
    tp_atr_mult,
    max_bars_in_trade,
    transaction_cost=0.0,
    slippage=0.0,
    window_size=0,
):
    """Run baseline signals through the same canonical execution engine as RL.

    An action opens at the current close and is immediately tested against that
    bar's SL, TP, then time-exit conditions. ``window_size`` applies the same
    observation warmup as policy evaluation. The zero default supports isolated
    execution tests; production baseline comparisons pass the experiment's
    configured window size.
    """
    market = df[["open", "high", "low", "close", "atr"]].copy()
    features = pd.DataFrame({"baseline_signal": np.zeros(len(df))}, index=df.index)
    env = ForexEnv(
        market,
        features,
        window_size=window_size,
        position_frac=position_frac,
        sl_atr_mult=sl_atr_mult,
        tp_atr_mult=tp_atr_mult,
        max_bars_in_trade=max_bars_in_trade,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )
    action_series = signals.reindex(df.index).fillna(0).astype(int)
    _, done = env.reset()[0], False
    while not done:
        action = int(action_series.iloc[env.t]) if env.t < len(df) - 1 else 0
        _, _, done, _, _ = env.step(action)
    return env.get_trade_log()


def baseline_flat(df):
    return pd.Series(0, index=df.index)


def baseline_random(df, p_enter=0.03, seed=123):
    """Step 1 generator, including its one-entry ``in_pos`` behavior."""
    rng, signal, in_pos = np.random.default_rng(seed), np.zeros(len(df), dtype=int), 0
    for index in range(len(df) - 1):
        if in_pos == 0 and rng.uniform() < p_enter:
            signal[index] = 1 if rng.uniform() < 0.5 else 2
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
    for name, signals in {
        "flat": baseline_flat(df),
        "random": baseline_random(df),
        "trend": baseline_trend(df),
    }.items():
        trades = _simulate_trades_from_signals(
            df,
            signals,
            config.position_risk_frac,
            config.sl_atr_mult,
            config.tp_atr_mult,
            config.max_bars_in_trade,
            config.transaction_cost,
            config.slippage,
            config.window_size,
        )
        mtm = daily_mtm_from_trades(df, trades, config.position_risk_frac)
        metrics = performance_metrics(trades, mtm)
        outcomes[name] = {
            "trades": trades,
            "r": R_metrics(trades),
            "mtm": mtm,
            "metrics": metrics,
            "sharpe": metrics["sharpe"],
            "mdd_pct": metrics["maxDD_pct"],
        }
    return outcomes


def equity_R_from_trades(trades, position_frac):
    if trades is None or len(trades) == 0:
        return np.array([1.0])
    equity = [1.0]
    for realized_r in trades["realized_R"].values:
        equity.append(equity[-1] * (1.0 + realized_r * position_frac))
    return np.array(equity)


def backtest_model(model, scaler, df_slice, feature_cols, price_cols, config, algo=None):
    """Deterministic policy rollout using preserved environment mechanics."""
    features = scaler.transform(df_slice[feature_cols])
    env = ForexEnv(df_slice[price_cols], features, seed=config.random_seed, **config.environment_kwargs)
    obs, done = env.reset(seed=config.random_seed)[0], False
    state, episode_start = None, np.ones((1,), dtype=bool)
    selected_algo = (algo or config.algo).upper()
    while not done:
        if selected_algo == "RPPO":
            action, state = model.predict(
                obs.reshape(1, -1), state=state, episode_start=episode_start, deterministic=True
            )
            episode_start = np.array([False])
        else:
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, _, done, _, _ = env.step(int(np.asarray(action).reshape(-1)[0]))
    trades = env.get_trade_log()
    mtm = daily_mtm_from_trades(df_slice, trades, config.position_risk_frac)
    metrics = performance_metrics(trades, mtm)
    return {
        "trades": trades,
        "equity_R": equity_R_from_trades(trades, config.position_risk_frac),
        "r": R_metrics(trades),
        "mtm": mtm,
        "metrics": metrics,
        "sharpe": metrics["sharpe"],
        "maxdd_pct": metrics["maxDD_pct"],
    }


def run_walkforward(algo, df_all, folds, config, train_fold, feature_cols, price_cols):
    """Opt-in orchestration; caller supplies training, so this function never trains implicitly."""
    rows = []
    for fold in folds:
        train = df_all.loc[fold["train_start"] : fold["train_end"]].copy()
        val = df_all.loc[fold["val_start"] : fold["val_end"]].copy()
        test = df_all.loc[fold["test_start"] : fold["test_end"]].copy()
        model, scaler, save_dir = train_fold(
            algo, train, val, config, feature_cols, price_cols, n_envs=config.wf_n_envs
        )
        result, baselines = (
            backtest_model(model, scaler, test, feature_cols, price_cols, config, algo=algo),
            run_baselines(test, config),
        )
        agent, mtm = result["r"], result["mtm"]
        rows.append(
            {
                "fold": fold["fold"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "val_start": fold["val_start"],
                "val_end": fold["val_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "total_R": agent["total_R"],
                "avg_R": agent["avg_R"],
                "PF": agent["profit_factor"],
                "win_rate": agent["win_rate"],
                "maxDD_R": agent["maxDD_R"],
                "MAR_R": agent["MAR_R"],
                "trades": agent["trades"],
                "sharpe": result["sharpe"],
                "sortino": result["metrics"]["sortino"],
                "maxDD_pct": result["maxdd_pct"],
                "turnover": result["metrics"]["turnover"],
                "trade_count": result["metrics"]["trade_count"],
                "return_pct": mtm["equity_pct"].iloc[-1] - 1.0 if len(mtm) else 0.0,
                "exit_mix": json.dumps(result["trades"].reason.value_counts().to_dict())
                if len(result["trades"])
                else "{}",
                "flat_R": baselines["flat"]["r"]["total_R"],
                "flat_PF": baselines["flat"]["r"]["profit_factor"],
                "flat_sharpe": baselines["flat"]["sharpe"],
                "rand_R": baselines["random"]["r"]["total_R"],
                "rand_PF": baselines["random"]["r"]["profit_factor"],
                "rand_sharpe": baselines["random"]["sharpe"],
                "trend_R": baselines["trend"]["r"]["total_R"],
                "trend_PF": baselines["trend"]["r"]["profit_factor"],
                "trend_sharpe": baselines["trend"]["sharpe"],
            }
        )
    return pd.DataFrame(rows)
