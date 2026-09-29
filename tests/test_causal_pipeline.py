import numpy as np
import pandas as pd
import pytest

from forex_rl import ExperimentConfig
from forex_rl.environment import ForexEnv
from forex_rl.preprocessing import make_scaled_frames
from forex_rl.research import _simulate_trades_from_signals, baseline_random, baseline_trend
from forex_rl.walk_forward import build_walkforward_folds, slice_fold


def test_entry_ignores_past_range_then_stop_precedes_target_for_both_sides():
    frame = pd.DataFrame(
        {
            "open": [10.0] * 4,
            "high": [10.0, 12.0, 12.0, 10.0],
            "low": [10.0, 8.0, 8.0, 10.0],
            "close": [10.0] * 4,
            "atr": [1.0] * 4,
        }
    )
    for action, entry_price in [(1, 10.1), (2, 9.9)]:
        env = ForexEnv(
            frame,
            pd.DataFrame({"f": [0.0] * 4}),
            window_size=1,
            sl_atr_mult=1,
            tp_atr_mult=1,
            slippage=0.1,
            transaction_cost=0.01,
            max_bars_in_trade=3,
        )
        env.step(action)
        assert env.position and env.trades == [] and env.bars_in_trade == 0
        assert np.isclose(env.entry_price, entry_price)
        env.step(0)
        trade = env.get_trade_log().iloc[0]
        assert trade.reason == "sl" and trade.exit_t == 2 and trade.bars_in_trade == 1
        assert trade.transaction_cost > 0 and trade.realized_R < -1


def test_next_bar_target_and_time_exit_keep_existing_cost_accounting():
    frame = pd.DataFrame(
        {
            "open": [10.0] * 4,
            "high": [10.0, 12.0, 12.0, 10.0],
            "low": [10.0, 10.0, 10.0, 10.0],
            "close": [10.0] * 4,
            "atr": [1.0] * 4,
        }
    )
    env = ForexEnv(frame, pd.DataFrame({"f": [0.0] * 4}), window_size=1, sl_atr_mult=1, tp_atr_mult=1)
    env.step(1)
    env.step(0)
    assert env.trades[0]["reason"] == "tp" and np.isclose(env.trades[0]["realized_R"], 1)
    env = ForexEnv(
        frame, pd.DataFrame({"f": [0.0] * 4}), window_size=1, sl_atr_mult=9, tp_atr_mult=9, max_bars_in_trade=1
    )
    env.step(1)
    env.step(0)
    assert env.trades[0]["reason"] == "time" and env.trades[0]["exit_t"] == 2


def test_final_candle_checks_stops_before_targets_then_time_or_eod():
    cases = [
        (1, 12, 8, 9, "sl"),
        (2, 12, 8, 9, "sl"),
        (1, 12, 10, 9, "tp"),
        (2, 10, 8, 9, "tp"),
        (1, 10, 10, 1, "time"),
        (2, 10, 10, 9, "eod"),
    ]
    for action, final_high, final_low, max_bars, expected in cases:
        frame = pd.DataFrame(
            {
                "open": [10.0] * 3,
                "high": [10.0, 12.0, final_high],
                "low": [10.0, 8.0, final_low],
                "close": [10.0] * 3,
                "atr": [1.0] * 3,
            }
        )
        env = ForexEnv(
            frame,
            pd.DataFrame({"f": [0.0] * 3}),
            window_size=1,
            sl_atr_mult=1,
            tp_atr_mult=1,
            max_bars_in_trade=max_bars,
        )
        _, _, done, _, _ = env.step(action)
        assert done and len(env.trades) == 1
        assert env.trades[0]["reason"] == expected
        assert env.trades[0]["exit_t"] == 2
        assert env.trades[0]["bars_in_trade"] == 1


def test_shortest_slice_requires_a_bar_after_first_possible_entry():
    frame = pd.DataFrame(
        {"open": [10.0] * 3, "high": [10.0, 12.0, 12.0], "low": [10.0, 8.0, 8.0], "close": [10.0] * 3, "atr": [1.0] * 3}
    )
    with pytest.raises(ValueError, match="At least one bar"):
        ForexEnv(frame.iloc[:2], pd.DataFrame({"f": [0.0] * 2}), window_size=1)
    env = ForexEnv(frame, pd.DataFrame({"f": [0.0] * 3}), window_size=1, sl_atr_mult=1, tp_atr_mult=1)
    _, _, done, _, _ = env.step(1)
    assert done and env.trades[0]["exit_t"] == 2 and env.trades[0]["reason"] == "sl"


def test_half_open_folds_and_train_only_scaler_exclude_endpoint_rows():
    dates = pd.date_range("2020-01-01", "2020-09-01", freq="D")
    frame = pd.DataFrame({"f": np.arange(len(dates), dtype=float)}, index=dates)
    fold = build_walkforward_folds(frame, 3, 2, 2, 2)[0]
    train, val, test = (slice_fold(frame, fold, part) for part in ("train", "val", "test"))
    assert train.index.max() < val.index.min() and val.index.max() < test.index.min()
    assert not train.index.intersection(val.index).size
    assert not val.index.intersection(test.index).size
    assert fold["train_end"] in val.index and fold["val_end"] in test.index
    changed = val.copy()
    changed.iloc[0, 0] = 1_000_000
    _, scaled_val, scaler = make_scaled_frames(train, changed, ["f"])
    assert np.isclose(scaler.means.f, train.f.mean()) and scaled_val.f.iloc[0] > 100


def test_baselines_trade_again_after_close_with_shared_execution():
    dates = pd.date_range("2020-01-01", periods=12)
    close = [10, 9, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    frame = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "atr": [1.0] * 12}, index=dates)
    cfg = ExperimentConfig(
        window_size=1, max_bars_in_trade=1, sl_atr_mult=99, tp_atr_mult=99, transaction_cost=0.001, slippage=0.1
    )
    for signals in (baseline_random(frame, p_enter=1, seed=4), baseline_trend(frame, fast=2, slow=3)):
        trades = _simulate_trades_from_signals(
            frame,
            signals,
            cfg.position_risk_frac,
            cfg.sl_atr_mult,
            cfg.tp_atr_mult,
            cfg.max_bars_in_trade,
            cfg.transaction_cost,
            cfg.slippage,
            cfg.window_size,
        )
        assert len(trades) >= 2
        assert (trades.entry_t.diff().dropna() >= 2).all()
        assert (trades.transaction_cost > 0).all()
