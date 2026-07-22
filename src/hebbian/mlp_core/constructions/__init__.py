"""MLP construction methods.

This package provides various methods for constructing MLPs:
- NTK (Neural Tangent Kernel) construction
"""

from hebbian.mlp_core.constructions.ntk import (
    NTKConstructionConfig,
    NTKParameters,
    construct_ntk_parameters,
    build_ntk_mlp_from_params,
    get_ntk_mlp,
)
__all__ = [
    # NTK
    "NTKConstructionConfig",
    "NTKParameters",
    "construct_ntk_parameters",
    "build_ntk_mlp_from_params",
    "get_ntk_mlp",
]
