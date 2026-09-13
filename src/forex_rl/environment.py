"""Dependency-light trading environment mechanics.

This preserves the notebook's action timing, entry pricing, SL/TP/time ordering,
and accounting.  A Gymnasium adapter can be added by research callers without
changing this core implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ForexEnv:
    def __init__(self, market_df, feat_df, window_size=64, position_frac=0.01, sl_atr_mult=2.0,
                 tp_atr_mult=2.5, max_bars_in_trade=30, transaction_cost=0.0, slippage=0.0,
                 invalid_action_penalty=0.0, dd_penalty=0.0, turnover_penalty=0.0, seed=42):
        assert len(market_df) == len(feat_df), "Market and feature frames must align."
        self.mkt, self.feat = market_df.reset_index(drop=True).copy(), feat_df.reset_index(drop=True).copy()
        self.window_size, self.position_frac = int(window_size), float(position_frac)
        self.sl_atr_mult, self.tp_atr_mult = float(sl_atr_mult), float(tp_atr_mult)
        self.max_bars_in_trade = int(max_bars_in_trade)
        self.transaction_cost, self.slippage = float(transaction_cost), float(slippage)
        self.invalid_action_penalty, self.dd_penalty = float(invalid_action_penalty), float(dd_penalty)
        self.turnover_penalty, self.rng = float(turnover_penalty), np.random.default_rng(seed)
        self.n_features, self.action_count, self.entries, self.trades = self.feat.shape[1], 3, [], []
        self._reset_state()

    def _reset_state(self):
        self.t, self.net_worth, self.peak_net_worth, self.position = self.window_size, 1.0, 1.0, 0
        self.entry_price = self.stop_price = self.take_price = None
        self.units, self.bars_in_trade, self.open_trade_i = 0.0, 0, None

    def reset(self, seed=None, options=None):
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
        realized_r = float(profit_value / (risk_value + 1e-12))
        self.net_worth += profit_value
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)
        drawdown = max(0.0, 1.0 - self.net_worth / self.peak_net_worth) if self.peak_net_worth > 0 else 0.0
        penalty = (self.dd_penalty * drawdown if self.dd_penalty > 0 else 0.0) + self.turnover_penalty
        self.trades.append({"type": "long" if self.position == 1 else "short", "reason": reason,
            "entry_t": self.entries[self.open_trade_i]["entry_t"] if self.open_trade_i is not None else None,
            "exit_t": self.t, "entry_price": self.entry_price, "exit_price": exit_price,
            "stop": self.stop_price, "take": self.take_price, "units": self.units,
            "realized_R": realized_r, "pnl_value": profit_value, "net_worth": self.net_worth})
        self.position, self.entry_price, self.stop_price, self.take_price = 0, None, None, None
        self.units, self.bars_in_trade, self.open_trade_i = 0.0, 0, None
        return float(realized_r - penalty)

    def step(self, action):
        reward, row = 0.0, self.mkt.iloc[self.t]
        high, low, close, atr = float(row.high), float(row.low), float(row.close), max(float(row.atr), 1e-6)
        if action in (1, 2):
            if self.position == 0:
                self.position = 1 if action == 1 else -1
                self.entry_price = close + self.slippage if action == 1 else close - self.slippage
                stop_distance, take_distance = self.sl_atr_mult * atr, self.tp_atr_mult * atr
                self.stop_price = self.entry_price - stop_distance if action == 1 else self.entry_price + stop_distance
                self.take_price = self.entry_price + take_distance if action == 1 else self.entry_price - take_distance
                self.units = self.net_worth * self.position_frac / max(1e-6, abs(self.entry_price - self.stop_price))
                self.entries.append({"type": "long" if action == 1 else "short", "entry_t": self.t,
                    "entry_price": self.entry_price, "stop": self.stop_price, "take": self.take_price, "units": self.units})
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
            reward += self._reward_from_exit(float(self.mkt.iloc[-1].close), "eod")
        return self._get_obs(), float(reward), done, False, {}

    def get_trade_log(self):
        return pd.DataFrame(self.trades)

    def get_entries_log(self):
        return pd.DataFrame(self.entries)
