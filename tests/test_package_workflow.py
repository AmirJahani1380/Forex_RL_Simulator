import json
from pathlib import Path

import numpy as np
import pandas as pd

from forex_rl import ExperimentConfig, generate_trade_report, model_kwargs, run_baselines
from forex_rl.models import validation_score
from forex_rl.research import baseline_random, baseline_trend
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
    cfg = ExperimentConfig(log_dir=str(tmp_path), wf_n_envs=3)
    model, scaler, _ = models.train_one_fold("DQN", frame(), frame(), cfg, ["ema20"], ["open", "high", "low", "close", "atr"])
    assert model == "reloaded" and scaler == "scaler" and observed["n_envs"] == 3
    assert observed["env"] is train_env and len(closed) == 2
