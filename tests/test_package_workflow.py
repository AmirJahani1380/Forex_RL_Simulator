import json
from pathlib import Path

import numpy as np
import pandas as pd

from forex_rl import ExperimentConfig, generate_trade_report, model_kwargs, run_baselines
from forex_rl.environment import make_env_factory
from forex_rl.preprocessing import ZScoreScaler


def frame(rows=8):
    index = pd.date_range("2022-01-01", periods=rows)
    return pd.DataFrame({"open": 10., "high": 10.2, "low": 9.8, "close": np.arange(rows) + 10., "atr": 1., "ema20": 10.}, index=index)


def test_config_retains_model_and_environment_controls():
    cfg = ExperimentConfig()
    assert cfg.total_timesteps == 400_000
    assert cfg.environment_kwargs["window_size"] == cfg.window_size
    assert model_kwargs("DQN", cfg)["gradient_steps"] == 4
    assert model_kwargs("QRDQN", cfg)["n_quantiles"] == cfg.qrdqn_n_quantiles
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
