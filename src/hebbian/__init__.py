"""Hebbian MLP constructions and paper experiments.

The package provides shared data, model, construction, sweep, transformer, and
fact-editing implementations for comparing GD, NTK, and Hebbian MLPs.
"""

from hebbian.core.config import RuntimeConfig
from hebbian.core.registry import Registry

__version__ = "0.1.0"

__all__ = ["Registry", "RuntimeConfig", "__version__"]
