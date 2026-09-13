import logging
import random

import numpy as np
import pandas as pd
import pytest

from forex_rl import ExperimentConfig, configure_logging, run_tiny_smoke, seed_everything


def synthetic_ohlcv(rows: int = 96) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="D")
    close = 1.1 + np.linspace(0, 0.04, rows) + np.sin(np.arange(rows) / 3) * 0.002
    return pd.DataFrame(
        {
            "open": close - 0.0005,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "volume": np.full(rows, 1000),
        },
        index=index,
    )


def test_config_round_trip_rejects_unknown_keys_and_keeps_defaults():
    config = ExperimentConfig.from_mapping({"window_size": 4, "random_seed": 17})
    assert config.to_dict()["window_size"] == 4
    assert config.to_dict()["transaction_cost"] == 0.0
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        ExperimentConfig.from_mapping({"window_szie": 4})


def test_seed_everything_replays_python_and_numpy_sequences():
    first_generator = seed_everything(12)
    first = (random.random(), np.random.random(), first_generator.random())
    second_generator = seed_everything(12)
    second = (random.random(), np.random.random(), second_generator.random())
    assert first == second


def test_configure_logging_does_not_duplicate_handlers():
    logger = configure_logging(logging.DEBUG)
    handler_count = len(logger.handlers)
    assert configure_logging().handlers == logger.handlers
    assert len(logger.handlers) == handler_count


def test_tiny_smoke_is_deterministic_and_does_not_train():
    config = ExperimentConfig(window_size=4, max_bars_in_trade=2, random_seed=11)
    first = run_tiny_smoke(synthetic_ohlcv(), config)
    second = run_tiny_smoke(synthetic_ohlcv(), config)
    assert first["prepared_rows"] > config.window_size + 1
    assert first["metrics"] == second["metrics"]
    assert first["metrics"]["trades"] == 0
