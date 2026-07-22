"""
Deterministic seeding utilities.

This module provides utilities for setting random seeds to ensure
reproducible experiments across different random number generators.
"""

import random
from typing import Optional

import numpy as np
import torch


_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: int) -> None:
    """
    Set the random seed for all random number generators.

    Sets seeds for:
        - Python's random module
        - NumPy's random generator
        - PyTorch's random generators (CPU and CUDA)

    Also configures PyTorch for deterministic behavior.

    Args:
        seed: The seed value to use
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configure PyTorch for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seed() -> Optional[int]:
    """
    Get the current global seed.

    Returns:
        The current global seed, or None if not set
    """
    return _GLOBAL_SEED


def seed_worker(worker_id: int) -> None:
    """
    Worker init function for DataLoader to ensure reproducibility.

    Use this with DataLoader's worker_init_fn parameter.

    Args:
        worker_id: The worker ID (provided by DataLoader)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Get a PyTorch generator with the given seed.

    Args:
        seed: The seed to use. If None, uses the global seed.

    Returns:
        A PyTorch Generator instance
    """
    if seed is None:
        seed = _GLOBAL_SEED or 0

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
