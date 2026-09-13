"""Explicit, serializable experiment configuration used by the notebook."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    symbol: str = "EURUSD=X"
    start_date: str = "2015-01-01"
    end_date: str = "2025-08-01"
    window_size: int = 64
    position_risk_frac: float = 0.01
    sl_atr_mult: float = 1.8
    tp_atr_mult: float = 2.2
    max_bars_in_trade: int = 18
    transaction_cost: float = 0.0
    slippage: float = 0.0
    invalid_action_penalty: float = 0.0
    dd_penalty: float = 0.0
    turnover_penalty: float = 0.0
    min_train_months: int = 24
    val_months: int = 9
    test_months: int = 6
    step_months: int = 12
    use_rolling: bool = True
    rolling_train_months: int = 36
    random_seed: int = 42
