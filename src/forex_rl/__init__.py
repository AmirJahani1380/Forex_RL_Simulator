"""Importable building blocks for the Forex RL research notebook."""

from .config import ExperimentConfig
from .data import download_eurusd
from .environment import ForexEnv, ForexEnvSB3, make_scaled_env
from .evaluation import (
    ExecutionAssumptions,
    R_metrics,
    daily_mtm_from_trades,
    drawdown_series,
    performance_metrics,
    trade_turnover,
)
from .features import compute_indicators
from .models import make_model, model_kwargs, train_one_fold, validation_score
from .preprocessing import ZScoreScaler, make_scaled_frames
from .reporting import (
    aggregate_walkforward_results,
    extended_metrics,
    generate_trade_report,
    generate_walkforward_report,
    zip_directory,
)
from .research import backtest_model, run_baselines, run_walkforward
from .runtime import configure_logging, seed_everything
from .smoke import FlatPolicy, run_tiny_smoke
from .walk_forward import build_walkforward_folds

__all__ = [
    "ExperimentConfig",
    "ExecutionAssumptions",
    "ForexEnv",
    "ForexEnvSB3",
    "R_metrics",
    "ZScoreScaler",
    "build_walkforward_folds",
    "compute_indicators",
    "daily_mtm_from_trades",
    "drawdown_series",
    "download_eurusd",
    "make_scaled_env",
    "make_model",
    "make_scaled_frames",
    "model_kwargs",
    "train_one_fold",
    "backtest_model",
    "run_baselines",
    "run_walkforward",
    "performance_metrics",
    "trade_turnover",
    "validation_score",
    "aggregate_walkforward_results",
    "extended_metrics",
    "generate_trade_report",
    "generate_walkforward_report",
    "zip_directory",
    "configure_logging",
    "seed_everything",
    "FlatPolicy",
    "run_tiny_smoke",
]
