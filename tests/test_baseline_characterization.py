import unittest

import numpy as np
import pandas as pd

from forex_rl_simulator.baseline import (
    ForexEnv,
    R_metrics,
    _daily_sharpe,
    _max_drawdown,
    build_walkforward_folds,
    compute_indicators,
    daily_mtm_from_trades,
    make_scaled_frames,
)


def market(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "atr"])


def features(n):
    return pd.DataFrame({"f1": np.arange(n), "f2": np.arange(n) + 100})


class BaselineCharacterizationTests(unittest.TestCase):
    def test_observation_reset_action_and_timestamp_alignment(self):
        env = ForexEnv(market([(10, 10.5, 9.5, 10, 1)] * 7), features(7), window_size=2, max_bars_in_trade=9)
        obs, info = env.reset(seed=7)
        self.assertEqual(info, {})
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_array_equal(obs, np.array([[0, 100], [1, 101]], dtype=np.float32))
        _, _, _, _, _ = env.step(0)
        np.testing.assert_array_equal(env._get_obs(), np.array([[1, 101], [2, 102]], dtype=np.float32))
        self.assertEqual(env.action_count, 3)

    def test_long_short_pnl_and_sl_before_tp_priority(self):
        # The next bar touches both thresholds; the stop keeps priority.
        both = market([(10, 10.1, 9.9, 10, 1), (10, 12, 8, 10, 1), (10, 12, 8, 10, 1), (10, 10, 10, 10, 1)])
        long_env = ForexEnv(
            both, features(4), window_size=1, position_frac=0.01, sl_atr_mult=1, tp_atr_mult=1, max_bars_in_trade=4
        )
        long_env.step(1)
        self.assertEqual(len(long_env.trades), 0)
        long_env.step(0)
        self.assertEqual(long_env.trades[0]["reason"], "sl")
        self.assertAlmostEqual(long_env.trades[0]["realized_R"], -1.0)
        short_env = ForexEnv(
            both, features(4), window_size=1, position_frac=0.01, sl_atr_mult=1, tp_atr_mult=1, max_bars_in_trade=4
        )
        short_env.step(2)
        short_env.step(0)
        self.assertEqual(short_env.trades[0]["reason"], "sl")
        self.assertAlmostEqual(short_env.trades[0]["pnl_value"], -0.01)

    def test_tp_time_and_end_of_data_exits(self):
        tp = market([(10, 10, 10, 10, 1), (10, 12, 10.1, 10.5, 1), (10, 12, 10, 10.5, 1), (10, 10, 10, 10, 1)])
        env = ForexEnv(tp, features(4), window_size=1, sl_atr_mult=1, tp_atr_mult=1, max_bars_in_trade=4)
        env.step(1)
        env.step(0)
        self.assertEqual(env.trades[0]["reason"], "tp")
        timed = market([(10, 10, 10, 10, 1), (10, 10.2, 9.8, 10.4, 1), (10, 10, 10, 10.8, 1), (10, 10, 10, 11, 1)])
        env = ForexEnv(timed, features(4), window_size=1, sl_atr_mult=2, tp_atr_mult=2, max_bars_in_trade=1)
        env.step(1)
        env.step(0)
        self.assertEqual(env.trades[0]["reason"], "time")
        eod = ForexEnv(timed, features(4), window_size=1, sl_atr_mult=99, tp_atr_mult=99, max_bars_in_trade=99)
        eod.step(1)
        eod.step(0)
        self.assertEqual(eod.trades[0]["reason"], "eod")

    def test_slippage_and_configured_transaction_cost_affect_execution_pnl(self):
        rows = market([(10, 10, 10, 10, 1), (10, 12, 8, 10, 1), (10, 10, 10, 10, 1), (10, 10, 10, 10, 1)])
        slipped = ForexEnv(
            rows,
            features(4),
            window_size=1,
            sl_atr_mult=99,
            tp_atr_mult=99,
            max_bars_in_trade=1,
            slippage=0.25,
            transaction_cost=0.7,
        )
        slipped.step(1)
        slipped.step(0)
        self.assertAlmostEqual(slipped.trades[0]["entry_price"], 10.25)
        self.assertLess(slipped.trades[0]["pnl_value"], 0)
        base = ForexEnv(
            rows, features(4), window_size=1, sl_atr_mult=99, tp_atr_mult=99, max_bars_in_trade=1, transaction_cost=0
        )
        base.step(1)
        base.step(0)
        self.assertNotEqual(slipped.trades[0]["pnl_value"], base.trades[0]["pnl_value"])
        costly = ForexEnv(
            rows, features(4), window_size=1, sl_atr_mult=99, tp_atr_mult=99, max_bars_in_trade=1, transaction_cost=0.7
        )
        costly.step(1)
        costly.step(0)
        self.assertLess(costly.trades[0]["pnl_value"], base.trades[0]["pnl_value"])
        self.assertGreater(costly.trades[0]["transaction_cost"], 0)
        short = ForexEnv(
            rows, features(4), window_size=1, sl_atr_mult=99, tp_atr_mult=99, max_bars_in_trade=1, slippage=0.25
        )
        short.step(2)
        short.step(0)
        self.assertLess(short.trades[0]["pnl_value"], 0)

    def test_train_only_scaler_orchestration_and_exact_walkforward_boundaries(self):
        train, val = pd.DataFrame({"a": [1.0, 3.0], "b": [7.0, 7.0]}), pd.DataFrame({"a": [101.0], "b": [7.0]})
        _, transformed_val, scaler = make_scaled_frames(train, val, ["a", "b"])
        self.assertEqual(float(scaler.means.a), 2.0)
        self.assertGreater(float(transformed_val.iloc[0].a), 50.0)
        dates = pd.date_range("2020-01-01", "2021-01-01", freq="D")
        folds = build_walkforward_folds(pd.DataFrame(index=dates), 3, 2, 2, 2, rolling_train_months=4)
        self.assertEqual(len(folds), 3)
        self.assertEqual(
            folds[0],
            {
                "fold": 1,
                "train_start": pd.Timestamp("2020-01-01"),
                "train_end": pd.Timestamp("2020-04-01"),
                "val_start": pd.Timestamp("2020-04-01"),
                "val_end": pd.Timestamp("2020-06-01"),
                "test_start": pd.Timestamp("2020-06-01"),
                "test_end": pd.Timestamp("2020-08-01"),
            },
        )
        self.assertEqual(folds[1]["train_start"], pd.Timestamp("2020-02-01"))
        self.assertEqual(folds[2]["test_start"], pd.Timestamp("2020-10-01"))
        self.assertEqual(folds[-1]["test_end"], pd.Timestamp("2020-12-01"))

    def test_indicator_alignment_seed_and_metrics(self):
        n = 25
        raw = pd.DataFrame(
            {
                "open": np.arange(n) + 10.0,
                "high": np.arange(n) + 11.0,
                "low": np.arange(n) + 9.0,
                "close": np.arange(n) + 10.0,
                "volume": 1.0,
            },
            index=pd.date_range("2020-01-01", periods=n),
        )
        calculated = compute_indicators(raw)
        self.assertTrue(calculated.index.equals(raw.index[-len(calculated) :]))
        one = ForexEnv(market([(10, 10, 10, 10, 1)] * 5), features(5), window_size=1, seed=42)
        expected = np.random.default_rng(42)
        self.assertEqual(one.rng.random(), expected.random())
        one.reset(seed=7)
        self.assertEqual(one.rng.random(), expected.random())
        trades = pd.DataFrame({"realized_R": [1.0, -0.5, 2.0]})
        metrics = R_metrics(trades)
        self.assertEqual(metrics["trades"], 3)
        self.assertAlmostEqual(metrics["total_R"], 2.5)
        self.assertAlmostEqual(metrics["maxDD_R"], -0.5)
        self.assertAlmostEqual(_max_drawdown(np.array([1.0, 2.0, 1.0, 3.0])), -0.5)
        self.assertTrue(np.isfinite(_daily_sharpe(np.array([0.01, -0.01, 0.02]))))

    def test_agent_integer_trade_timestamps_are_resolved_for_datetime_mtm(self):
        dated = market(
            [
                (10, 10, 10, 10, 1),
                (10, 10.2, 9.8, 10, 1),
                (10, 10.2, 9.8, 11, 1),
                (10, 10, 10, 11, 1),
                (10, 10, 10, 11, 1),
            ]
        )
        dated.index = pd.date_range("2020-01-01", periods=len(dated))
        env = ForexEnv(dated, features(len(dated)), window_size=1, sl_atr_mult=99, tp_atr_mult=99, max_bars_in_trade=2)
        env.step(1)
        env.step(0)
        env.step(0)
        trades = env.get_trade_log()
        self.assertEqual(trades.iloc[0]["entry_t"], 1)
        self.assertGreater(trades.iloc[0]["realized_R"], 0)
        mtm = daily_mtm_from_trades(dated, trades)
        self.assertGreater(mtm["equity_pct"].iloc[-1], 1.0)
        self.assertGreater(mtm["daily_ret"].iloc[2], 0.0)


if __name__ == "__main__":
    unittest.main()
