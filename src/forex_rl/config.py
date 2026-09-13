"""Explicit, serializable experiment configuration used by the notebook."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


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
    algo: str = "DQN"
    total_timesteps: int = 400_000
    eval_freq: int = 120_000
    n_envs: int = 24
    wf_n_envs: int = 24
    rppo_n_steps: int = 128
    qrdqn_n_quantiles: int = 51
    min_trades_val: int = 4
    log_dir: str = "./logs_rl"
    report_dir: str = "./logs_rl/reports"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        """Create a config while rejecting misspelled keys instead of ignoring them."""
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot suitable for recording a research run."""
        return asdict(self)

    @property
    def environment_kwargs(self) -> dict[str, float | int]:
        return {
            "window_size": self.window_size,
            "position_frac": self.position_risk_frac,
            "sl_atr_mult": self.sl_atr_mult,
            "tp_atr_mult": self.tp_atr_mult,
            "max_bars_in_trade": self.max_bars_in_trade,
            "transaction_cost": self.transaction_cost,
            "slippage": self.slippage,
            "invalid_action_penalty": self.invalid_action_penalty,
            "dd_penalty": self.dd_penalty,
            "turnover_penalty": self.turnover_penalty,
        }
