"""
Neural Tangent Kernel (NTK) MLP construction method.

This module wraps the existing NTKConstructionConfig and get_ntk_mlp from mlps
to provide a consistent Method interface.
"""

from hebbian.methods.ntk.construct import NTKMethod

__all__ = ["NTKMethod"]
