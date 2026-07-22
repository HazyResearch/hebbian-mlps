"""Minimal building blocks for the paper's gated MLPs."""

from hebbian.mlp_core.blocks.activations import (
    Activation,
    ActivationConfig,
    get_activation_from_string,
    normalized_hermite,
)
from hebbian.mlp_core.blocks.gated_block import (
    GatedLinearBlock,
    GatedLinearBlockWeights,
)
from hebbian.mlp_core.blocks.linear_block import LinearBlock, LinearBlockWeights
from hebbian.mlp_core.blocks.mlps import GatedMLP, MLP

__all__ = [
    "Activation",
    "ActivationConfig",
    "GatedLinearBlock",
    "GatedLinearBlockWeights",
    "GatedMLP",
    "LinearBlock",
    "LinearBlockWeights",
    "MLP",
    "get_activation_from_string",
    "normalized_hermite",
]
