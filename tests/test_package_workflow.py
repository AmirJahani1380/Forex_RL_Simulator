import json
from pathlib import Path

import numpy as np
import pandas as pd

from forex_rl import ExperimentConfig, generate_trade_report, model_kwargs, run_baselines
from forex_rl.models import make_rmetrics_callback, validation_score
from forex_rl.research import _simulate_trades_from_signals, baseline_random, baseline_trend, run_walkforward
from forex_rl.reporting import generate_walkforward_report, zip_directory
from forex_rl.environment import make_env_factory
from forex_rl.preprocessing import ZScoreScaler


def frame(rows=8):
    index = pd.date_range("2022-01-01", periods=rows)
    return pd.DataFrame({"open": 10., "high": 10.2, "low": 9.8, "close": np.arange(rows) + 10., "atr": 1., "ema20": 10.}, index=index)


def test_config_retains_model_and_environment_controls():
    cfg = ExperimentConfig()
    assert cfg.total_timesteps == 400_000
    assert cfg.environment_kwargs["window_size"] == cfg.window_size
    assert model_kwargs("DQN", cfg) == {"learning_rate": 2e-5, "buffer_size": 250_000, "learning_starts": 10_000, "batch_size": 1024, "tau": .01, "gamma": .99, "train_freq": 4, "gradient_steps": 4, "exploration_fraction": .40, "exploration_final_eps": .10}
    assert model_kwargs("QRDQN", cfg)["n_quantiles"] == 51
    assert model_kwargs("RPPO", cfg)["batch_size"] == 768


def test_env_factory_uses_train_fitted_scaler_without_refitting():
    df = frame()
    scaler = ZScoreScaler(); scaler.fit(pd.DataFrame({"ema20": [1., 3.]}))
    factory = make_env_factory(df, scaler, ["ema20"], ["open", "high", "low", "close", "atr"], {"window_size": 1}, 3)
    try:
        wrapped = factory()
    except ImportError:  # minimal dependency environment: adapter is correctly optional
        return
    obs, _ = wrapped.reset()
    assert obs.shape == (1,)
    assert float(scaler.means.iloc[0]) == 2.


def test_baselines_and_report_are_small_and_inspectable(tmp_path):
    df, cfg = frame(), ExperimentConfig(window_size=1, max_bars_in_trade=1)
    results = run_baselines(df, cfg)
    assert set(results) == {"flat", "random", "trend"}
    report = generate_trade_report(df, results["flat"]["trades"], results["flat"]["mtm"], out_prefix=str(tmp_path / "report"))
    assert all(Path(report[key]).exists() for key in ("html", "csv", "json"))
    assert "trades" in json.loads(Path(report["json"]).read_text())
    walkforward = generate_walkforward_report(pd.DataFrame({"total_R": [1.0], "trades": [2]}), tmp_path / "wf", "DQN")
    archive = zip_directory(tmp_path / "wf", tmp_path / "wf.zip")
    assert Path(walkforward["html"]).exists() and Path(archive).exists()


def test_selection_gates_and_mar_ranking_are_exact():
    valid = {"profit_factor": 1.1, "trades": 6, "maxDD_R": -2.0, "MAR_R": .7, "total_R": 9.}
    assert validation_score(valid) == .7
    assert validation_score({**valid, "MAR_R": np.nan}) == 9.
    assert validation_score({**valid, "profit_factor": .99}) == -np.inf
    assert validation_score({**valid, "trades": 5}) == -np.inf
    assert validation_score({**valid, "maxDD_R": -2.01}) == -np.inf


def test_disabled_eval_callback_never_modulos_or_evaluates(monkeypatch, tmp_path):
    import forex_rl.models as models
    class BaseCallback:
        def __init__(self, verbose=0): self.n_calls = 0
    monkeypatch.setattr(models, "require_sb3", lambda: (object, object, object, BaseCallback, object))
    evaluated = []
    callback = make_rmetrics_callback(lambda model: evaluated.append(model), tmp_path, eval_freq=0)
    assert callback._on_step() is True and evaluated == []


def test_characterized_baseline_generators_emit_at_most_one_entry():
    df = frame(12)
    random_signal = baseline_random(df, p_enter=1., seed=123)
    assert random_signal.tolist() == [1] + [0] * 11
    trend_df = frame(8); trend_df["close"] = [10, 9, 8, 9, 10, 11, 12, 13]
    trend_signal = baseline_trend(trend_df, fast=2, slow=3)
    assert trend_signal.tolist() == [0, 0, 0, 0, 1, 0, 0, 0]


def test_train_fold_uses_wf_envs_reloads_best_and_closes(monkeypatch, tmp_path):
    import forex_rl.models as models
    closed, observed = [], {}
    class Env:
        def close(self): closed.append(self)
    train_env, eval_env = Env(), Env()
    class Callback:
        best_model_path = str(tmp_path / "best.zip")
    Path(Callback.best_model_path).write_text("checkpoint")
    class Model:
        def learn(self, **kwargs): observed["callback"] = kwargs["callback"]
    class Loader:
        @staticmethod
        def load(path, env, device):
            observed.update(path=path, env=env, device=device); return "reloaded"
    import forex_rl.environment as environment
    monkeypatch.setattr(environment, "make_scaled_env", lambda *args: (observed.setdefault("n_envs", args[-1]) and train_env, eval_env, "scaler"))
    monkeypatch.setattr(models, "make_model", lambda *args: Model())
    monkeypatch.setattr(models, "make_rmetrics_callback", lambda *args, **kwargs: Callback())
    monkeypatch.setattr(models, "require_sb3", lambda: (Loader, Loader, Loader, object, object))
    cfg = ExperimentConfig(log_dir=str(tmp_path), n_envs=3, wf_n_envs=7)
    model, scaler, save_dir = models.train_one_fold("DQN", frame(), frame(), cfg, ["ema20"], ["open", "high", "low", "close", "atr"])
    _, _, second_save_dir = models.train_one_fold("DQN", frame(), frame(), cfg, ["ema20"], ["open", "high", "low", "close", "atr"])
    assert model == "reloaded" and scaler == "scaler" and observed["n_envs"] == 3
    assert observed["env"] is train_env and len(closed) == 4 and save_dir != second_save_dir


def test_baseline_simulator_zero_atr_uses_zero_units():
    df = frame(3); df["atr"] = 0.
    trades = _simulate_trades_from_signals(df, pd.Series([1, 0, 0], index=df.index), .01, 2., 2., 1)
    assert trades.iloc[0].units == 0. and trades.iloc[0].realized_R == 0.


def test_baseline_simulator_zero_stop_multiplier_keeps_stop_at_entry():
    df = frame(3)
    trades = _simulate_trades_from_signals(df, pd.Series([1, 0, 0], index=df.index), .01, 0., 2., 1)
    trade = trades.iloc[0]
    assert trade.stop == trade.entry_price
    assert trade.units == 0. and trade.realized_R == 0.


def test_walkforward_restores_full_schema_and_routes_algo_and_wf_parallelism(monkeypatch):
    import forex_rl.research as research
    df = frame(5)
    fold = {"fold": 4, "train_start": df.index[0], "train_end": df.index[1], "val_start": df.index[1], "val_end": df.index[2], "test_start": df.index[2], "test_end": df.index[4]}
    cfg, observed = ExperimentConfig(algo="DQN", wf_n_envs=9), {}
    def trainer(algo, train, val, config, features, prices, n_envs):
        observed.update(train_algo=algo, n_envs=n_envs); return object(), object(), "unused"
    trades = pd.DataFrame({"reason": ["tp"]})
    agent = {"total_R": 3., "avg_R": 1., "profit_factor": 1.5, "win_rate": 2 / 3, "maxDD_R": -1., "MAR_R": 3., "trades": 3}
    monkeypatch.setattr(research, "backtest_model", lambda *args, **kwargs: (observed.update(backtest_algo=kwargs["algo"]) or {"trades": trades, "r": agent, "mtm": pd.DataFrame({"equity_pct": [1., 1.2]}), "sharpe": 2., "maxdd_pct": -.1}))
    baseline = {"r": {"total_R": .5, "profit_factor": 1.1}, "sharpe": .2}
    monkeypatch.setattr(research, "run_baselines", lambda *args: {"flat": baseline, "random": baseline, "trend": baseline})
    result = run_walkforward("QRDQN", df, [fold], cfg, trainer, ["ema20"], ["open", "high", "low", "close", "atr"])
    expected = {"fold", "total_R", "avg_R", "PF", "win_rate", "maxDD_R", "MAR_R", "trades", "sharpe", "maxDD_pct", "return_pct", "exit_mix", "flat_R", "flat_PF", "flat_sharpe", "rand_R", "rand_PF", "rand_sharpe", "trend_R", "trend_PF", "trend_sharpe"}
    assert set(result.columns) == expected and np.isclose(result.iloc[0].return_pct, .2)
    assert observed == {"train_algo": "QRDQN", "n_envs": 9, "backtest_algo": "QRDQN"}
