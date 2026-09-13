"""Optional-model dependency boundary.

Training orchestration remains explicitly initiated by the notebook; this module
keeps optional Stable-Baselines dependencies out of normal package imports.
"""


def require_sb3():
    """Return optional SB3 classes or explain how to install them."""
    try:
        from stable_baselines3 import DQN
        from sb3_contrib import QRDQN, RecurrentPPO
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install forex-rl-simulator[research] for model training.") from exc
    return DQN, QRDQN, RecurrentPPO
