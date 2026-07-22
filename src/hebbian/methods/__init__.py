"""
MLP construction methods.

This module provides different methods for constructing MLPs:
    - GD MLP: Gradient descent trained MLP
    - NTK: Neural Tangent Kernel construction
    - Hebbian: Hebbian learning-based construction

All methods implement the Method interface for consistent usage.
"""

from hebbian.methods.base import Method

# Import method implementations to trigger registration
from hebbian.methods.gd_mlp import GDMLPMethod
from hebbian.methods.ntk import NTKMethod
from hebbian.methods.hebbian import HebbianMethod

__all__ = [
    "Method",
    "GDMLPMethod",
    "NTKMethod",
    "HebbianMethod",
]
