"""Executable characterization of the pure logic in the research notebook.

This module intentionally preserves the formulas and event order in notebook cells
6, 8, 9, and 11.  It avoids optional Gym/SB3 dependencies so tiny tests can run in
a fresh Python environment; the notebook remains the training entry point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook cell 6 indicator construction, unchanged."""
    out = df.copy()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    delta = out["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / 14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    out["rsi"] = 100 - (100 / (1 + rs))
    ema12, ema26 = out["close"].ewm(span=12, adjust=False).mean(), out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    mavg, mstd = out["close"].rolling(20).mean(), out["close"].rolling(20).std(ddof=0)
    out["bb_middle"], out["bb_upper"], out["bb_lower"] = mavg, mavg + 2 * mstd, mavg - 2 * mstd
    high_low = (out["high"] - out["low"]).abs()
    high_close, low_close = (out["high"] - out["close"].shift()).abs(), (out["low"] - out["close"].shift()).abs()
    out["atr"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).ewm(alpha=1 / 14, adjust=False).mean()
    out["atr_n"] = out["atr"].to_numpy() / out["close"].to_numpy()
    out["ret"] = out["close"].pct_change()
    return out.dropna().copy()


class ZScoreScaler:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, df_feat: pd.DataFrame):
        self.means = df_feat.mean()
        self.stds = df_feat.std().replace(0, 1e-8)

    def transform(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        return (df_feat - self.means) / self.stds

    def fit_transform(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        self.fit(df_feat)
        return self.transform(df_feat)


def build_walkforward_folds(df: pd.DataFrame, min_train_months=24, val_months=6, test_months=6, step_months=6,
                            use_rolling=True, rolling_train_months=36):
    """Cell 8 folds with its configured rolling defaults made explicit."""
    start, end = df.index.min(), df.index.max()
    folds, test_start, k = [], start + pd.DateOffset(months=min_train_months + val_months), 1
    while True:
        train_end, val_end, test_end = test_start - pd.DateOffset(months=val_months), test_start, test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        train_start = max(train_end - pd.DateOffset(months=max(min_train_months, rolling_train_months)), start) if use_rolling else start
        folds.append({"fold": k, "train_start": train_start, "train_end": train_end, "val_start": train_end,
                      "val_end": val_end, "test_start": val_end, "test_end": test_end})
        k, test_start = k + 1, test_start + pd.DateOffset(months=step_months)
    return folds


def make_scaled_frames(train_df: pd.DataFrame, eval_df: pd.DataFrame, feature_cols):
    """Source-tied pure portion of cells 10/13's train/evaluation orchestration."""
    scaler = ZScoreScaler()
    scaler.fit(train_df[feature_cols])
    return scaler.transform(train_df[feature_cols]), scaler.transform(eval_df[feature_cols]), scaler


class ForexEnv:
    """Cell 9 environment mechanics without the optional Gym base/wrapper."""
    def __init__(self, market_df, feat_df, window_size=64, position_frac=0.01, sl_atr_mult=2.0, tp_atr_mult=2.5,
                 max_bars_in_trade=30, transaction_cost=0.0, slippage=0.0, invalid_action_penalty=0.0,
                 dd_penalty=0.0, turnover_penalty=0.0, seed=42):
        assert len(market_df) == len(feat_df), "Market and feature frames must align."
        self.mkt, self.feat = market_df.reset_index(drop=True).copy(), feat_df.reset_index(drop=True).copy()
        self.window_size, self.position_frac = int(window_size), float(position_frac)
        self.sl_atr_mult, self.tp_atr_mult, self.max_bars_in_trade = float(sl_atr_mult), float(tp_atr_mult), int(max_bars_in_trade)
        self.transaction_cost, self.slippage = float(transaction_cost), float(slippage)
        self.invalid_action_penalty, self.dd_penalty, self.turnover_penalty = float(invalid_action_penalty), float(dd_penalty), float(turnover_penalty)
        self.rng, self.n_features, self.action_count = np.random.default_rng(seed), self.feat.shape[1], 3
        self.entries, self.trades = [], []
        self._reset_state()

    def _reset_state(self):
        self.t, self.net_worth, self.peak_net_worth, self.position = self.window_size, 1.0, 1.0, 0
        self.entry_price = self.stop_price = self.take_price = None
        self.units, self.bars_in_trade, self.open_trade_i = 0.0, 0, None

    def reset(self, seed=None, options=None):
        # Notebook delegates seed to Gym but does not recreate self.rng.
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        return self.feat.iloc[self.t - self.window_size:self.t].values.astype(np.float32)

    def _reward_from_exit(self, exit_price, reason):
        if self.position == 0 or self.entry_price is None or self.stop_price is None:
            return 0.0
        risk_value = abs(self.entry_price - self.stop_price) * self.units
        if risk_value <= 1e-12:
            return 0.0
        profit_value = ((exit_price - self.entry_price) if self.position == 1 else (self.entry_price - exit_price)) * self.units
        realized_R = float(profit_value / (risk_value + 1e-12))
        self.net_worth += profit_value
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)
        dd = max(0.0, 1.0 - self.net_worth / self.peak_net_worth) if self.peak_net_worth > 0 else 0.0
        penalty = (self.dd_penalty * dd if self.dd_penalty > 0 else 0.0) + self.turnover_penalty
        self.trades.append({"type": "long" if self.position == 1 else "short", "reason": reason,
            "entry_t": self.entries[self.open_trade_i]["entry_t"] if self.open_trade_i is not None else None, "exit_t": self.t,
            "entry_price": self.entry_price, "exit_price": exit_price, "stop": self.stop_price, "take": self.take_price,
            "units": self.units, "realized_R": realized_R, "pnl_value": profit_value, "net_worth": self.net_worth})
        self.position, self.entry_price, self.stop_price, self.take_price, self.units, self.bars_in_trade, self.open_trade_i = 0, None, None, None, 0.0, 0, None
        return float(realized_R - penalty)

    def step(self, action):
        reward = 0.0
        row = self.mkt.iloc[self.t]
        high, low, close, atr_abs = float(row["high"]), float(row["low"]), float(row["close"]), max(float(row["atr"]), 1e-6)
        if action in (1, 2):
            if self.position == 0:
                self.position = 1 if action == 1 else -1
                self.entry_price = close + self.slippage if action == 1 else close - self.slippage
                stop_dist, take_dist = self.sl_atr_mult * atr_abs, self.tp_atr_mult * atr_abs
                self.stop_price = self.entry_price - stop_dist if action == 1 else self.entry_price + stop_dist
                self.take_price = self.entry_price + take_dist if action == 1 else self.entry_price - take_dist
                self.units = (self.net_worth * self.position_frac) / max(1e-6, abs(self.entry_price - self.stop_price))
                self.entries.append({"type": "long" if action == 1 else "short", "entry_t": self.t, "entry_price": self.entry_price,
                                     "stop": self.stop_price, "take": self.take_price, "units": self.units})
                self.open_trade_i, reward = len(self.entries) - 1, reward - self.turnover_penalty
            else:
                reward -= self.invalid_action_penalty
        if self.position != 0:
            self.bars_in_trade += 1
            if self.position == 1:
                if low <= self.stop_price: reward += self._reward_from_exit(self.stop_price, "sl")
                elif high >= self.take_price: reward += self._reward_from_exit(self.take_price, "tp")
                elif self.bars_in_trade >= self.max_bars_in_trade: reward += self._reward_from_exit(close, "time")
            else:
                if high >= self.stop_price: reward += self._reward_from_exit(self.stop_price, "sl")
                elif low <= self.take_price: reward += self._reward_from_exit(self.take_price, "tp")
                elif self.bars_in_trade >= self.max_bars_in_trade: reward += self._reward_from_exit(close, "time")
        self.t += 1
        done = self.t >= len(self.mkt) - 1
        if done and self.position != 0:
            reward += self._reward_from_exit(float(self.mkt.iloc[-1]["close"]), "eod")
        return self._get_obs(), float(reward), done, False, {}

    def get_trade_log(self): return pd.DataFrame(self.trades)
    def get_entries_log(self): return pd.DataFrame(self.entries)


def R_metrics(trades):
    if trades is None or len(trades) == 0:
        return {"trades": 0, "win_rate": np.nan, "profit_factor": np.nan, "total_R": 0.0, "avg_R": 0.0, "maxDD_R": 0.0, "MAR_R": np.nan}
    wins, losses, total_R = trades.loc[trades.realized_R > 0, "realized_R"].sum(), abs(trades.loc[trades.realized_R < 0, "realized_R"].sum()), trades.realized_R.sum()
    cum = peak = max_dd = 0.0
    for r in trades.realized_R.values:
        cum += r; peak = max(peak, cum); max_dd = min(max_dd, cum - peak)
    return {"trades": int(len(trades)), "win_rate": float((trades.realized_R > 0).mean()), "profit_factor": float(wins / losses) if losses > 0 else np.nan,
            "total_R": float(total_R), "avg_R": float(trades.realized_R.mean()), "maxDD_R": float(max_dd), "MAR_R": float(total_R / abs(max_dd)) if max_dd < 0 else np.nan}


def _max_drawdown(series):
    if series is None or len(series) == 0: return np.nan
    peak, max_dd = -np.inf, 0.0
    for value in series:
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1.0 if peak > 0 else 0.0)
    return float(max_dd)


def _daily_sharpe(daily_returns, risk_free=0.0):
    if daily_returns is None or len(daily_returns) < 2: return np.nan
    mu, sd = np.nanmean(daily_returns) - risk_free / 252.0, np.nanstd(daily_returns, ddof=1)
    return np.nan if sd <= 1e-12 else float(mu / sd * np.sqrt(252))


def daily_mtm_from_trades(df: pd.DataFrame, trades: pd.DataFrame, position_frac=.01) -> pd.DataFrame:
    """Cell 11 MTM reconstruction, including its timestamp-domain comparison."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    dates = df.index
    equity_pct = np.ones(len(df), dtype=float)
    equity_R = np.ones(len(df), dtype=float)
    in_pos = 0
    entry_price = stop = risk_per_unit = None
    trade_iter = list(trades.to_dict("records")) if trades is not None else []
    k = 0
    for i in range(len(df)):
        d = dates[i]
        if k < len(trade_iter) and trade_iter[k]["entry_t"] == d and (i == 0 or in_pos == 0):
            entry_price, stop = trade_iter[k]["entry_price"], trade_iter[k]["stop"]
            risk_per_unit = abs(entry_price - stop)
            in_pos = 1 if trade_iter[k]["type"] == "long" else -1
        if in_pos != 0 and risk_per_unit and risk_per_unit > 1e-12:
            unrealized_R = ((df["close"].iloc[i] - entry_price) if in_pos == 1 else (entry_price - df["close"].iloc[i])) / risk_per_unit
            equity_R[i] = (equity_R[i - 1] if i > 0 else 1.0) * (1.0 + unrealized_R * position_frac)
        elif i > 0:
            equity_R[i] = equity_R[i - 1]
        if k < len(trade_iter) and trade_iter[k]["exit_t"] == d:
            realized_R = trade_iter[k]["realized_R"]
            equity_R[i] = (equity_R[i - 1] if i > 0 else 1.0) * (1.0 + realized_R * position_frac)
            in_pos = 0
            entry_price = stop = risk_per_unit = None
            k += 1
        equity_pct[i] = equity_R[i]
    daily_ret = np.zeros(len(df), dtype=float)
    if len(df) > 1:
        daily_ret[1:] = np.diff(equity_pct) / equity_pct[:-1]
    return pd.DataFrame({"equity_pct": equity_pct, "daily_ret": daily_ret, "equity_R": equity_R}, index=dates)
