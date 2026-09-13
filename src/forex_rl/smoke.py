"""Deliberately tiny deterministic workflow smoke helper; it never trains a model."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .environment import ForexEnv
from .evaluation import R_metrics
from .features import compute_indicators
from .preprocessing import ZScoreScaler
from .runtime import configure_logging, seed_everything


class PredictModel(Protocol):
    """The minimal non-training Stable-Baselines-like prediction interface."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, object]: ...


class FlatPolicy:
    """A deterministic stand-in used only to exercise the policy/evaluation boundary."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del observation, deterministic
        return np.array([0]), None


def run_tiny_smoke(market_data: pd.DataFrame, config: ExperimentConfig, model: PredictModel | None = None) -> dict:
    """Run data → features → train-only scaling → environment → prediction → metrics.

    The caller provides a tiny in-memory OHLCV frame. No download, training, artifact
    writing, walk-forward loop, or optional RL dependency is involved.
    """
    logger = configure_logging()
    seed_everything(config.random_seed)
    prepared = compute_indicators(market_data)
    feature_cols = [
        "ema20",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "atr_n",
        "ret",
    ]
    price_cols = ["open", "high", "low", "close", "atr"]
    if len(prepared) <= config.window_size + 1:
        raise ValueError("Tiny smoke input must leave more rows than the configured observation window.")

    split = max(config.window_size + 1, len(prepared) // 2)
    scaler = ZScoreScaler()
    scaler.fit(prepared.iloc[:split][feature_cols])
    env = ForexEnv(
        prepared[price_cols],
        scaler.transform(prepared[feature_cols]),
        seed=config.random_seed,
        **config.environment_kwargs,
    )
    policy = model or FlatPolicy()
    observation, done = env.reset()[0], False
    while not done:
        action, _ = policy.predict(observation.reshape(1, -1), deterministic=True)
        observation, _, done, _, _ = env.step(int(np.asarray(action).reshape(-1)[0]))
    metrics = R_metrics(env.get_trade_log())
    logger.info("Tiny smoke completed: rows=%d, trades=%d", len(prepared), metrics["trades"])
    return {"prepared_rows": len(prepared), "metrics": metrics, "scaler": scaler}
