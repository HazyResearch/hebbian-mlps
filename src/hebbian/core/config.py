"""
Typed configurations for the Hebbian MLP library.

This module provides pydraclass-based configuration classes for
various components of the library.
"""

from typing import Optional

from hebbian.config import pydraclass


@pydraclass
class RuntimeConfig:
    """
    Base runtime configuration for Hebbian experiments.

    This configuration class holds common parameters shared across
    different experiment types, such as device, dtype, and seed settings.
    """

    # Device and dtype settings
    device: str = "cuda"
    build_dtype: str = "float64"
    final_dtype: Optional[str] = None

    # Random seed for reproducibility
    seed: int = 0

    # Verbosity
    verbose: bool = False
    # Output directory
    base_dir: Optional[str] = None
