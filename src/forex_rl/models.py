"""Optional Stable-Baselines model construction and R-based validation selection."""

from __future__ import annotations

from pathlib import Path


def require_sb3():
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import BaseCallback
        from sb3_contrib import QRDQN, RecurrentPPO
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install forex-rl-simulator[research] for model training.") from exc
    return DQN, QRDQN, RecurrentPPO, BaseCallback, nn


def model_kwargs(algo, config):
    """Notebook hyperparameters, returned without constructing or training a model."""
    name = algo.upper()
    if name == "DQN":
        return {"learning_rate": 3e-5, "buffer_size": 250_000, "learning_starts": 10_000,
                "batch_size": 1024, "tau": .01, "gamma": .99, "train_freq": 4,
                "gradient_steps": 4, "exploration_fraction": .40, "exploration_final_eps": .10}
    if name == "QRDQN":
        return {"learning_rate": 3e-5, "buffer_size": 250_000, "learning_starts": 10_000,
                "batch_size": 1024, "tau": .01, "gamma": .99, "train_freq": 4,
                "gradient_steps": 2, "n_quantiles": config.qrdqn_n_quantiles}
    if name == "RPPO":
        return {"n_steps": config.rppo_n_steps, "batch_size": 768, "n_epochs": 5,
                "ent_coef": .005, "learning_rate": 3e-5, "clip_range": .2}
    raise ValueError(f"Unknown algo: {algo}")


def make_model(algo, env, config):
    """Build the original DQN/QRDQN/RPPO families; this function never calls learn."""
    DQN, QRDQN, RecurrentPPO, _, nn = require_sb3()
    name, kwargs = algo.upper(), model_kwargs(algo, config)
    common = {"verbose": 1, "seed": config.random_seed, "device": "auto"}
    if name == "DQN":
        return DQN("MlpPolicy", env, policy_kwargs={"activation_fn": nn.ReLU, "net_arch": [256, 256]}, **kwargs, **common)
    if name == "QRDQN":
        n_quantiles = kwargs.pop("n_quantiles")
        return QRDQN("MlpPolicy", env, policy_kwargs={"activation_fn": nn.ReLU, "net_arch": [256, 256], "n_quantiles": n_quantiles}, **kwargs, **common)
    return RecurrentPPO("MlpLstmPolicy", env, policy_kwargs={"activation_fn": nn.ReLU, "net_arch": {"pi": [128, 128], "vf": [128, 128]}}, **kwargs, **common)


def validation_score(metrics, min_trades=6):
    """Preserve the notebook callback's hard-coded six-trade selection gate."""
    if metrics["trades"] < min_trades or metrics["profit_factor"] != metrics["profit_factor"]:
        return float("-inf")
    return metrics["total_R"]


def make_rmetrics_callback(eval_fn, save_dir, eval_freq, min_trades=6):
    """Create a callback around an injected deterministic evaluator.

    Injection keeps callbacks testable and prevents hidden model evaluation during import.
    """
    _, _, _, BaseCallback, _ = require_sb3()
    class RMetricsEvalCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0); self.best_score = float("-inf"); self.best_model_path = None
        def _on_step(self):
            if self.n_calls % eval_freq:
                return True
            metrics = eval_fn(self.model)
            score = validation_score(metrics, min_trades=min_trades)
            if score > self.best_score:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                self.best_score = score; self.best_model_path = str(Path(save_dir) / "best_model")
                self.model.save(self.best_model_path)
            return True
    return RMetricsEvalCallback()


def train_one_fold(algo, train_df, validation_df, config, feature_cols, price_cols):
    """Original opt-in one-fold training flow with deterministic R-metric selection.

    No caller reaches this path accidentally: it is the only function here that
    invokes ``model.learn``.
    """
    from .environment import make_scaled_env
    from .research import backtest_model
    train_env, eval_env, scaler = make_scaled_env(train_df, validation_df, feature_cols, price_cols,
                                                   config.environment_kwargs, config.random_seed, config.n_envs)
    model = make_model(algo, train_env, config)
    save_dir = str(Path(config.log_dir) / f"{algo}_fold_{str(train_df.index.max())[:10]}")
    callback = make_rmetrics_callback(
        lambda candidate: backtest_model(candidate, scaler, validation_df, feature_cols, price_cols, config)["r"],
        save_dir, config.eval_freq, min_trades=6,
    )
    try:
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
    finally:
        eval_env.close()
    return model, scaler, save_dir
