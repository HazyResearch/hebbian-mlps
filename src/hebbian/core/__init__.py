"""
Core utilities for the Hebbian MLP library.

This module provides:
    - Registry: Global registry for methods and encoders
    - RuntimeConfig: Typed runtime configuration
    - Seeding utilities for deterministic experiments
    - I/O utilities for jsonl/csv writing
    - Metrics computation (accuracy, margin, param counting)
"""

from hebbian.core.registry import Registry
from hebbian.core.config import RuntimeConfig
from hebbian.core.seed import set_seed, get_seed
from hebbian.core.io import save_results, load_results
from hebbian.core.metrics import compute_accuracy, compute_margin, count_parameters
from hebbian.core.dtype_utils import get_dtype

__all__ = [
    "Registry",
    "RuntimeConfig",
    "set_seed",
    "get_seed",
    "save_results",
    "load_results",
    "compute_accuracy",
    "compute_margin",
    "count_parameters",
    "get_dtype",
]
