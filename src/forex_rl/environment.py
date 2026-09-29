"""Dependency-light trading environment mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import ExecutionAssumptions

try:  # Keep normal package imports dependency-light.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised in minimal installations
    gym = None
    spaces = None

_EnvBase = gym.Env if gym is not None else object


class ForexEnv(_EnvBase):
    def __init__(
        self,
        market_df,
        feat_df,
        window_size=64,
        position_frac=0.01,
        sl_atr_mult=2.0,
        tp_atr_mult=2.5,
        max_bars_in_trade=30,
        transaction_cost=0.0,
        slippage=0.0,
        invalid_action_penalty=0.0,
        dd_penalty=0.0,
        turnover_penalty=0.0,
        seed=42,
    ):
        assert len(market_df) == len(feat_df), "Market and feature frames must align."
        self.mkt, self.feat = market_df.reset_index(drop=True).copy(), feat_df.reset_index(drop=True).copy()
        self.window_size, self.position_frac = int(window_size), float(position_frac)
        self.sl_atr_mult, self.tp_atr_mult = float(sl_atr_mult), float(tp_atr_mult)
        self.max_bars_in_trade = int(max_bars_in_trade)
        self.execution = ExecutionAssumptions(float(transaction_cost), float(slippage))
        self.transaction_cost, self.slippage = self.execution.transaction_cost, self.execution.slippage
        self.invalid_action_penalty, self.dd_penalty = float(invalid_action_penalty), float(dd_penalty)
        self.turnover_penalty, self.rng = float(turnover_penalty), np.random.default_rng(seed)
        if gym is not None:
            super().__init__()
        self.n_features, self.action_count, self.entries, self.trades = self.feat.shape[1], 3, [], []
        if spaces is not None:
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.window_size, self.n_features), dtype=np.float32
            )
        self._reset_state()

    def _reset_state(self):
        self.t, self.net_worth, self.peak_net_worth, self.position = self.window_size, 1.0, 1.0, 0
        self.entry_price = self.stop_price = self.take_price = None
        self.units, self.bars_in_trade, self.open_trade_i, self.entry_cost = 0.0, 0, None, 0.0

    def reset(self, seed=None, options=None):
        if gym is not None:
            super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        return self.feat.iloc[self.t - self.window_size : self.t].values.astype(np.float32)

    def _reward_from_exit(self, exit_price, reason):
        if self.position == 0 or self.entry_price is None or self.stop_price is None:
            return 0.0
        risk_value = abs(self.entry_price - self.stop_price) * self.units
        if risk_value <= 1e-12:
            return 0.0
        gross_profit_value = (
            (exit_price - self.entry_price) if self.position == 1 else (self.entry_price - exit_price)
        ) * self.units
        exit_cost = abs(exit_price * self.units) * self.transaction_cost
        profit_value = gross_profit_value - self.entry_cost - exit_cost
        realized_r = float(profit_value / (risk_value + 1e-12))
        self.net_worth += gross_profit_value - exit_cost
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)
        drawdown = max(0.0, 1.0 - self.net_worth / self.peak_net_worth) if self.peak_net_worth > 0 else 0.0
        penalty = (self.dd_penalty * drawdown if self.dd_penalty > 0 else 0.0) + self.turnover_penalty
        self.trades.append(
            {
                "type": "long" if self.position == 1 else "short",
                "reason": reason,
                "entry_t": self.entries[self.open_trade_i]["entry_t"] if self.open_trade_i is not None else None,
                "exit_t": self.t,
                "entry_price": self.entry_price,
                "exit_price": exit_price,
                "stop": self.stop_price,
                "take": self.take_price,
                "units": self.units,
                "bars_in_trade": self.bars_in_trade,
                "realized_R": realized_r,
                "pnl_value": profit_value,
                "entry_cost": self.entry_cost,
                "exit_cost": exit_cost,
                "transaction_cost": self.entry_cost + exit_cost,
                "net_worth": self.net_worth,
            }
        )
        self.position, self.entry_price, self.stop_price, self.take_price = 0, None, None, None
        self.units, self.bars_in_trade, self.open_trade_i, self.entry_cost = 0.0, 0, None, 0.0
        return float(realized_r - penalty)

    def _check_exit(self, high, low, close):
        self.bars_in_trade += 1
        if self.position == 1:
            if low <= self.stop_price:
                return self._reward_from_exit(self.stop_price, "sl")
            if high >= self.take_price:
                return self._reward_from_exit(self.take_price, "tp")
        else:
            if high >= self.stop_price:
                return self._reward_from_exit(self.stop_price, "sl")
            if low <= self.take_price:
                return self._reward_from_exit(self.take_price, "tp")
        if self.bars_in_trade >= self.max_bars_in_trade:
            return self._reward_from_exit(close, "time")
        return 0.0

    def step(self, action):
        reward, row = 0.0, self.mkt.iloc[self.t]
        high, low, close, atr = float(row.high), float(row.low), float(row.close), max(float(row.atr), 1e-6)
        # A close-priced entry cannot encounter this bar's earlier high or low.
        was_open = self.position != 0
        if action in (1, 2):
            if self.position == 0:
                self.position = 1 if action == 1 else -1
                self.entry_price = close + self.slippage if action == 1 else close - self.slippage
                stop_distance, take_distance = self.sl_atr_mult * atr, self.tp_atr_mult * atr
                self.stop_price = self.entry_price - stop_distance if action == 1 else self.entry_price + stop_distance
                self.take_price = self.entry_price + take_distance if action == 1 else self.entry_price - take_distance
                self.units = self.net_worth * self.position_frac / max(1e-6, abs(self.entry_price - self.stop_price))
                self.entry_cost = abs(self.entry_price * self.units) * self.transaction_cost
                self.net_worth -= self.entry_cost
                self.entries.append(
                    {
                        "type": "long" if action == 1 else "short",
                        "entry_t": self.t,
                        "entry_price": self.entry_price,
                        "stop": self.stop_price,
                        "take": self.take_price,
                        "units": self.units,
                    }
                )
                self.open_trade_i, reward = len(self.entries) - 1, reward - self.turnover_penalty
            else:
                reward -= self.invalid_action_penalty
        if was_open:
            reward += self._check_exit(high, low, close)
        self.t += 1
        done = self.t >= len(self.mkt) - 1
        if done and self.position != 0:
            final = self.mkt.iloc[-1]
            reward += self._check_exit(float(final.high), float(final.low), float(final.close))
            if self.position != 0:
                reward += self._reward_from_exit(float(final.close), "eod")
        return self._get_obs(), float(reward), done, False, {}

    def get_trade_log(self):
        return pd.DataFrame(self.trades)

    def get_entries_log(self):
        return pd.DataFrame(self.entries)


if gym is not None:

    class ForexEnvSB3(gym.Wrapper):
        """Original SB3 shim: flatten a window/features observation for MlpPolicy."""

        def __init__(self, env: ForexEnv):
            super().__init__(env)
            window, features = env.observation_space.shape
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window * features,), dtype=np.float32)
            self.action_space = env.action_space

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return obs.reshape(-1).astype(np.float32), info

        def step(self, action):
            obs, reward, done, truncated, info = self.env.step(action)
            return obs.reshape(-1).astype(np.float32), reward, done, truncated, info
else:

    class ForexEnvSB3:  # pragma: no cover - only used when optional Gym is missing
        def __init__(self, *args, **kwargs):
            raise ImportError("Install forex-rl-simulator[research] for the Gymnasium/SB3 adapter.")


def make_env_factory(df_slice, scaler, feature_cols, price_cols, env_kwargs, seed):
    """Create the original vector-environment thunk; scaler is never refit here."""

    def build():
        env = ForexEnv(
            df_slice[price_cols].copy(), scaler.transform(df_slice[feature_cols].copy()), seed=seed, **env_kwargs
        )
        return ForexEnvSB3(env)

    return build


def make_scaled_env(train_df, eval_df, feature_cols, price_cols, env_kwargs, seed=42, n_envs=1):
    """Fit one scaler on training features and build train/evaluation SB3 VecEnvs."""
    if gym is None:
        raise ImportError("Install forex-rl-simulator[research] for vectorized environments.")
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from .preprocessing import ZScoreScaler

    scaler = ZScoreScaler()
    scaler.fit(train_df[feature_cols])
    train_fns = [
        make_env_factory(train_df, scaler, feature_cols, price_cols, env_kwargs, seed + index)
        for index in range(n_envs)
    ]
    train_env = SubprocVecEnv(train_fns) if n_envs > 1 else DummyVecEnv(train_fns)
    eval_env = DummyVecEnv([make_env_factory(eval_df, scaler, feature_cols, price_cols, env_kwargs, seed + 999)])
    return train_env, eval_env, scaler
