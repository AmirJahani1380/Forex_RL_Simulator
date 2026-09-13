"""Small, dependency-light runtime helpers for reproducible research runs."""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the package logger once, without changing the root logger."""
    logger = logging.getLogger("forex_rl")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def seed_everything(seed: int, environment: Any | None = None) -> np.random.Generator:
    """Seed Python, NumPy, optional Torch, and supported environment interfaces.

    This function intentionally does not enable Torch deterministic-algorithm mode:
    callers retain the project's established model execution semantics while getting
    repeatable random-number initialization where the installed dependencies allow it.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # Torch is an optional research dependency.
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    if environment is not None:
        action_space = getattr(environment, "action_space", None)
        if action_space is not None and hasattr(action_space, "seed"):
            action_space.seed(seed)
        if hasattr(environment, "seed"):
            environment.seed(seed)
    return np.random.default_rng(seed)
