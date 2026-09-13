"""Leakage-free evaluation helpers shared by policy and baseline backtests.

Return statistics use the supplied marked-to-market equity series; they never
inspect future bars. ``transaction_cost`` is a proportional per-side rate on
executed notional and ``slippage`` remains the existing absolute-price entry
adjustment. Both are execution assumptions, not learned parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Explicit assumptions shared by RL and baseline execution.

    ``transaction_cost`` is charged at entry and exit as a fraction of each
    side's notional. ``slippage`` preserves the historical entry-only,
    absolute-price convention (adverse for long and short entries).
    """

    transaction_cost: float = 0.0
    slippage: float = 0.0

    def __post_init__(self):
        if not isfinite(self.transaction_cost) or not isfinite(self.slippage):
            raise ValueError("Execution assumptions must be finite.")
        if self.transaction_cost < 0 or self.slippage < 0:
            raise ValueError("Execution assumptions must be non-negative.")


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
    values = trades["realized_R"].astype(float).to_numpy()
    wins, losses, total = values[values > 0].sum(), abs(values[values < 0].sum()), values.sum()
    cumulative = np.r_[0.0, np.cumsum(values)]
    max_dd = float(np.min(cumulative - np.maximum.accumulate(cumulative)))
    return {
        "trades": int(len(values)),
        "win_rate": float(np.mean(values > 0)),
        "profit_factor": float(wins / losses) if losses > 0 else np.nan,
        "total_R": float(total),
        "avg_R": float(np.mean(values)),
        "maxDD_R": max_dd,
        "MAR_R": float(total / abs(max_dd)) if max_dd < 0 else np.nan,
    }


def _max_drawdown(series):
    if series is None or len(series) == 0:
        return np.nan
    values = np.asarray(series, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        return np.nan
    peaks = np.maximum.accumulate(values)
    return float(np.min(np.where(peaks > 0, values / peaks - 1.0, 0.0)))


def _daily_sharpe(daily_returns, risk_free=0.0, periods_per_year=252):
    values = np.asarray(daily_returns if daily_returns is not None else [], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    excess = values - risk_free / periods_per_year
    std = np.std(excess, ddof=1)
    return np.nan if std <= 1e-12 else float(np.mean(excess) / std * np.sqrt(periods_per_year))


def _daily_sortino(daily_returns, risk_free=0.0, periods_per_year=252):
    values = np.asarray(daily_returns if daily_returns is not None else [], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    excess = values - risk_free / periods_per_year
    downside_deviation = np.sqrt(np.mean(np.square(np.minimum(excess, 0.0))))
    return (
        np.nan
        if downside_deviation <= 1e-12
        else float(np.mean(excess) / downside_deviation * np.sqrt(periods_per_year))
    )


def drawdown_series(equity):
    """Return running peak and fractional drawdown for an equity series."""
    values = pd.Series(equity, copy=True, dtype=float)
    peaks = values.cummax()
    return pd.DataFrame({"equity": values, "peak_equity": peaks, "drawdown": values.divide(peaks).sub(1.0).fillna(0.0)})


def trade_turnover(trades):
    """Executed entry-plus-exit notional; zero for empty or incomplete logs."""
    needed = {"units", "entry_price", "exit_price"}
    if trades is None or len(trades) == 0 or not needed.issubset(trades.columns):
        return 0.0
    units = trades["units"].astype(float).abs()
    return float((units * (trades["entry_price"].astype(float).abs() + trades["exit_price"].astype(float).abs())).sum())


def performance_metrics(trades, mtm_df, risk_free=0.0, periods_per_year=252):
    """Comparable trade/equity metrics for policies and baselines.

    Sharpe and Sortino annualize finite daily MTM returns. They return ``NaN``
    for fewer than two finite returns, constant returns, or (for Sortino) no
    downside deviation. Maximum drawdown is ``NaN`` for non-finite or
    non-positive equity, otherwise the worst fractional decline from a running
    MTM equity peak. Turnover is entry-plus-exit executed notional and trade
    count is closed rows in the supplied trade log. Empty trade logs report a
    zero count/turnover and ``NaN`` risk ratios.
    """
    base = R_metrics(trades)
    returns = np.asarray(mtm_df["daily_ret"], dtype=float) if mtm_df is not None and len(mtm_df) else np.array([])
    equity = np.asarray(mtm_df["equity_pct"], dtype=float) if mtm_df is not None and len(mtm_df) else np.array([])
    finite_returns = returns[np.isfinite(returns)]
    cagr = (
        np.prod(1.0 + finite_returns) ** (periods_per_year / len(finite_returns)) - 1.0
        if len(finite_returns)
        else np.nan
    )
    max_dd = _max_drawdown(equity)
    return {
        **base,
        "sharpe": _daily_sharpe(returns, risk_free, periods_per_year),
        "sortino": _daily_sortino(returns, risk_free, periods_per_year),
        "maxDD_pct": max_dd,
        "CAGR": float(cagr) if np.isfinite(cagr) else np.nan,
        "calmar": float(cagr / abs(max_dd)) if np.isfinite(cagr) and max_dd < 0 else np.nan,
        "turnover": trade_turnover(trades),
        "trade_count": base["trades"],
    }


def _resolve_trade_time(value, index):
    """Resolve preserved positional environment timestamps to report labels."""
    if value in index:
        return value
    if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(index):
        return index[int(value)]
    return None


def daily_mtm_from_trades(df: pd.DataFrame, trades: pd.DataFrame, position_frac=0.01) -> pd.DataFrame:
    """Build an equity/return/drawdown table from chronologically closed trades.

    Positional timestamps emitted by ``ForexEnv`` are mapped to the supplied
    frame index solely for reporting. This fixes RL MTM alignment without
    altering execution timing or the public trade-log schema.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["equity_pct", "daily_ret", "equity_R", "peak_equity", "drawdown"])
    equity = np.ones(len(df), dtype=float)
    records = [] if trades is None else list(trades.to_dict("records"))
    by_entry, by_exit = {}, {}
    for record in records:
        entry_time = _resolve_trade_time(record.get("entry_t"), df.index)
        exit_time = _resolve_trade_time(record.get("exit_t"), df.index)
        if entry_time is not None and exit_time is not None:
            by_entry.setdefault(entry_time, []).append(record)
            by_exit.setdefault(exit_time, []).append(record)
    active, entry_equity = None, 1.0
    for i, date in enumerate(df.index):
        previous = equity[i - 1] if i else 1.0
        if active is None and by_entry.get(date):
            active, entry_equity = by_entry[date][0], previous
        if active is None:
            equity[i] = previous
        elif active in by_exit.get(date, []):
            recorded_equity = active.get("net_worth")
            if recorded_equity is not None and np.isfinite(float(recorded_equity)):
                equity[i] = float(recorded_equity)
            else:
                equity[i] = entry_equity * (1.0 + float(active.get("realized_R", 0.0)) * position_frac)
            active = None
        else:
            entry_price, stop = float(active["entry_price"]), float(active["stop"])
            risk_per_unit = abs(entry_price - stop)
            if risk_per_unit <= 1e-12:
                equity[i] = previous
            else:
                direction = 1.0 if active.get("type") == "long" else -1.0
                unrealized_r = direction * (float(df.close.iloc[i]) - entry_price) / risk_per_unit
                if "net_worth" in active:
                    gross_pnl = direction * (float(df.close.iloc[i]) - entry_price) * float(active["units"])
                    equity[i] = entry_equity - float(active.get("entry_cost", 0.0)) + gross_pnl
                else:
                    equity[i] = entry_equity * (1.0 + unrealized_r * position_frac)
    daily_return = np.zeros(len(df), dtype=float)
    # Equity begins at one before the first reported bar; include bar-zero fees
    # and same-bar closes so compounded returns reconcile to terminal equity.
    daily_return[0] = equity[0] - 1.0
    if len(df) > 1:
        daily_return[1:] = np.divide(np.diff(equity), equity[:-1], out=np.zeros(len(df) - 1), where=equity[:-1] != 0)
    report = drawdown_series(pd.Series(equity, index=df.index))
    report.insert(0, "daily_ret", daily_return)
    report.insert(0, "equity_pct", equity)
    report.insert(2, "equity_R", equity)
    return report
