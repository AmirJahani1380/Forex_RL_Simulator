"""Backward-compatible imports for the Step 1 characterization suite.

New research code belongs in :mod:`forex_rl`; this module remains so existing
notebook extracts and tests keep their established import paths.
"""

from forex_rl.environment import ForexEnv
from forex_rl.evaluation import R_metrics, _daily_sharpe, _max_drawdown, daily_mtm_from_trades
from forex_rl.features import compute_indicators
from forex_rl.preprocessing import ZScoreScaler, make_scaled_frames
from forex_rl.walk_forward import build_walkforward_folds

__all__ = [
    "ForexEnv",
    "R_metrics",
    "ZScoreScaler",
    "_daily_sharpe",
    "_max_drawdown",
    "build_walkforward_folds",
    "compute_indicators",
    "daily_mtm_from_trades",
    "make_scaled_frames",
]
