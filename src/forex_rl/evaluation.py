"""Metric helpers preserved from notebook cell 11."""

import numpy as np
import pandas as pd


def R_metrics(trades):
    if trades is None or len(trades) == 0:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "total_R": 0.0,
            "avg_R": 0.0,
            "maxDD_R": 0.0,
            "MAR_R": np.nan,
        }
    wins = trades.loc[trades.realized_R > 0, "realized_R"].sum()
    losses = abs(trades.loc[trades.realized_R < 0, "realized_R"].sum())
    total = trades.realized_R.sum()
    cumulative = peak = max_dd = 0.0
    for value in trades.realized_R.values:
        cumulative += value
        peak = max(peak, cumulative)
        max_dd = min(max_dd, cumulative - peak)
    return {
        "trades": int(len(trades)),
        "win_rate": float((trades.realized_R > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else np.nan,
        "total_R": float(total),
        "avg_R": float(trades.realized_R.mean()),
        "maxDD_R": float(max_dd),
        "MAR_R": float(total / abs(max_dd)) if max_dd < 0 else np.nan,
    }


def _max_drawdown(series):
    if series is None or len(series) == 0:
        return np.nan
    peak, max_dd = -np.inf, 0.0
    for value in series:
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1.0 if peak > 0 else 0.0)
    return float(max_dd)


def _daily_sharpe(daily_returns, risk_free=0.0):
    if daily_returns is None or len(daily_returns) < 2:
        return np.nan
    mean, std = np.nanmean(daily_returns) - risk_free / 252.0, np.nanstd(daily_returns, ddof=1)
    return np.nan if std <= 1e-12 else float(mean / std * np.sqrt(252))


def daily_mtm_from_trades(df: pd.DataFrame, trades: pd.DataFrame, position_frac=0.01) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    dates, equity_pct, equity_r = df.index, np.ones(len(df)), np.ones(len(df))
    in_position, entry_price, stop, risk_per_unit, k = 0, None, None, None, 0
    records = list(trades.to_dict("records")) if trades is not None else []
    for i, date in enumerate(dates):
        if k < len(records) and records[k]["entry_t"] == date and (i == 0 or in_position == 0):
            entry_price, stop = records[k]["entry_price"], records[k]["stop"]
            risk_per_unit, in_position = abs(entry_price - stop), 1 if records[k]["type"] == "long" else -1
        if in_position != 0 and risk_per_unit and risk_per_unit > 1e-12:
            unrealized_r = (
                (df.close.iloc[i] - entry_price) if in_position == 1 else (entry_price - df.close.iloc[i])
            ) / risk_per_unit
            equity_r[i] = (equity_r[i - 1] if i else 1.0) * (1.0 + unrealized_r * position_frac)
        elif i:
            equity_r[i] = equity_r[i - 1]
        if k < len(records) and records[k]["exit_t"] == date:
            equity_r[i] = (equity_r[i - 1] if i else 1.0) * (1.0 + records[k]["realized_R"] * position_frac)
            in_position, entry_price, stop, risk_per_unit, k = 0, None, None, None, k + 1
        equity_pct[i] = equity_r[i]
    daily_return = np.zeros(len(df))
    if len(df) > 1:
        daily_return[1:] = np.diff(equity_pct) / equity_pct[:-1]
    return pd.DataFrame({"equity_pct": equity_pct, "daily_ret": daily_return, "equity_R": equity_r}, index=dates)
