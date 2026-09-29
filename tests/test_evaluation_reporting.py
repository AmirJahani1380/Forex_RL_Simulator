import numpy as np
import pandas as pd
import pytest

from forex_rl import ExperimentConfig
from forex_rl.environment import ForexEnv
from forex_rl.evaluation import (
    ExecutionAssumptions,
    _daily_sharpe,
    _daily_sortino,
    _max_drawdown,
    daily_mtm_from_trades,
    performance_metrics,
)
from forex_rl.reporting import extended_metrics, generate_trade_report
from forex_rl.research import _simulate_trades_from_signals, run_baselines


def test_performance_metrics_reports_documented_equity_and_trade_measures():
    index = pd.date_range("2024-01-01", periods=4)
    frame = pd.DataFrame({"close": [10.0, 10.0, 11.0, 10.0]}, index=index)
    trades = pd.DataFrame(
        {
            "entry_t": [0, 2],
            "exit_t": [1, 3],
            "units": [2.0, 1.0],
            "entry_price": [10.0, 11.0],
            "exit_price": [11.0, 10.0],
            "stop": [9.0, 12.0],
            "type": ["long", "short"],
            "realized_R": [1.0, -0.5],
        }
    )
    mtm = daily_mtm_from_trades(frame, trades, position_frac=0.1)
    metrics = performance_metrics(trades, mtm)
    assert np.allclose(mtm["equity_pct"], [1.0, 1.1, 1.1, 1.045])
    assert np.isclose(mtm["drawdown"].iloc[-1], -0.05)
    assert metrics["trade_count"] == 2
    assert metrics["turnover"] == 63.0
    assert np.isclose(metrics["maxDD_pct"], -0.05)
    assert np.isfinite(metrics["sharpe"]) and np.isfinite(metrics["sortino"])


def test_baseline_cost_and_slippage_match_configured_execution_assumptions():
    index = pd.date_range("2024-01-01", periods=3)
    frame = pd.DataFrame(
        {"open": [10.0] * 3, "high": [10.0] * 3, "low": [10.0] * 3, "close": [10.0] * 3, "atr": [1.0] * 3},
        index=index,
    )
    signals = pd.Series([1, 0, 0], index=index)
    baseline = _simulate_trades_from_signals(frame, signals, 0.01, 2.0, 2.0, 1)
    frictional = _simulate_trades_from_signals(frame, signals, 0.01, 2.0, 2.0, 1, transaction_cost=0.01, slippage=0.1)
    assert frictional.iloc[0].entry_price == 10.1
    assert frictional.iloc[0].transaction_cost > 0
    assert frictional.iloc[0].realized_R < baseline.iloc[0].realized_R
    assert ExecutionAssumptions(0.01, 0.1).transaction_cost == 0.01


def test_baseline_reuses_environment_next_bar_execution_and_friction():
    index = pd.date_range("2024-01-01", periods=3)
    frame = pd.DataFrame(
        {
            "open": [10.0] * 3,
            "high": [12.0, 12.0, 10.0],
            "low": [8.0, 8.0, 10.0],
            "close": [10.0] * 3,
            "atr": [1.0] * 3,
        },
        index=index,
    )
    signals = pd.Series([1, 0, 0], index=index)
    baseline = _simulate_trades_from_signals(frame, signals, 0.01, 1.0, 1.0, 2, transaction_cost=0.01, slippage=0.1)
    env = ForexEnv(
        frame,
        pd.DataFrame({"signal": [0.0] * len(frame)}, index=index),
        window_size=0,
        position_frac=0.01,
        sl_atr_mult=1.0,
        tp_atr_mult=1.0,
        max_bars_in_trade=2,
        transaction_cost=0.01,
        slippage=0.1,
    )
    env.reset()
    env.step(1)
    env.step(0)
    expected = env.get_trade_log()
    pd.testing.assert_frame_equal(baseline, expected)
    assert baseline.iloc[0].reason == "sl"  # next-bar SL wins over TP in both paths


def test_mtm_charges_entry_fee_on_entry_bar_and_matches_environment_exit_equity():
    index = pd.date_range("2024-01-01", periods=4)
    frame = pd.DataFrame(
        {
            "open": [10.0] * 4,
            "high": [10.0] * 4,
            "low": [10.0] * 4,
            "close": [10.0, 10.5, 10.5, 10.5],
            "atr": [1.0] * 4,
        },
        index=index,
    )
    trades = _simulate_trades_from_signals(frame, pd.Series([1, 0, 0, 0], index=index), 0.01, 5.0, 5.0, 2, 0.01)
    mtm = daily_mtm_from_trades(frame, trades, 0.01)
    trade = trades.iloc[0]
    assert mtm.iloc[0].equity_pct < 1.0
    assert np.isclose(mtm.iloc[int(trade.exit_t)].equity_pct, trade.net_worth)
    assert np.isclose(np.prod(1.0 + mtm.daily_ret), mtm.equity_pct.iloc[-1])


def test_baseline_comparison_starts_at_policy_observation_warmup():
    index = pd.date_range("2024-01-01", periods=5)
    frame = pd.DataFrame(
        {"open": [10.0] * 5, "high": [10.0] * 5, "low": [10.0] * 5, "close": [10.0] * 5, "atr": [1.0] * 5},
        index=index,
    )
    config = ExperimentConfig(window_size=2, max_bars_in_trade=1)
    outcomes = run_baselines(frame, config)
    signals = pd.Series([1, 0, 1, 0, 0], index=index)
    direct = _simulate_trades_from_signals(
        frame,
        signals,
        config.position_risk_frac,
        config.sl_atr_mult,
        config.tp_atr_mult,
        config.max_bars_in_trade,
        window_size=config.window_size,
    )
    assert direct.iloc[0].entry_t == config.window_size
    assert outcomes["flat"]["trades"].empty
    assert outcomes["flat"]["mtm"].index.equals(frame.index)


def test_datetime_trade_report_uses_time_duration_and_writes_equity_artifact(tmp_path):
    index = pd.date_range("2024-01-01", periods=3)
    trades = pd.DataFrame(
        {
            "entry_t": [index[0]],
            "exit_t": [index[2]],
            "units": [1.0],
            "entry_price": [10.0],
            "exit_price": [11.0],
            "realized_R": [1.0],
        }
    )
    mtm = pd.DataFrame({"equity_pct": [1.0, 1.0, 1.1], "daily_ret": [0.0, 0.0, 0.1]}, index=index)
    metrics = extended_metrics(trades, mtm)
    report = generate_trade_report(pd.DataFrame(index=index), trades, mtm, out_prefix=str(tmp_path / "datetime"))
    assert metrics["avg_holding_days"] == 2.0 and np.isnan(metrics["avg_bars_in_trade"])
    assert (tmp_path / "datetime_equity.csv").exists() and report["equity_csv"].endswith("datetime_equity.csv")


def test_metric_edge_cases_are_explicit_and_execution_assumptions_are_finite():
    assert np.isnan(_daily_sharpe([0.0, 0.0])) and np.isnan(_daily_sortino([0.01, 0.02]))
    assert np.isnan(_daily_sharpe([np.nan, np.inf])) and np.isnan(_max_drawdown([1.0, 0.0]))
    empty = performance_metrics(pd.DataFrame(), pd.DataFrame())
    assert empty["trade_count"] == 0 and empty["turnover"] == 0.0 and np.isnan(empty["sharpe"])
    with pytest.raises(ValueError, match="finite"):
        ExecutionAssumptions(np.inf, 0.0)
