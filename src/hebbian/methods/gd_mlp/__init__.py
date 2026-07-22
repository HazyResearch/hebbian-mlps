"""
Gradient Descent MLP method.

This module wraps the existing GDMLPConfig and get_gd_mlp from mlps
to provide a consistent Method interface.
"""

from hebbian.methods.gd_mlp.model import GDMLPMethod

__all__ = ["GDMLPMethod"]
