"""Feature construction preserved from notebook cell 6."""

from __future__ import annotations

import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    delta = out["close"].diff()
    roll_up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    roll_down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = 100 - (100 / (1 + roll_up / (roll_down + 1e-12)))
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    middle, std = out["close"].rolling(20).mean(), out["close"].rolling(20).std(ddof=0)
    out["bb_middle"], out["bb_upper"], out["bb_lower"] = middle, middle + 2 * std, middle - 2 * std
    ranges = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs(),
        ],
        axis=1,
    )
    out["atr"] = ranges.max(axis=1).ewm(alpha=1 / 14, adjust=False).mean()
    out["atr_n"] = out["atr"].to_numpy() / out["close"].to_numpy()
    out["ret"] = out["close"].pct_change()
    return out.dropna().copy()
