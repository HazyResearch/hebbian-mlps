"""Neural-tangent-kernel baseline used in the paper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from hebbian.config import pydraclass
from hebbian.mlp_core.blocks import (
    GatedLinearBlock,
    GatedLinearBlockWeights,
    GatedMLP,
    LinearBlock,
    LinearBlockWeights,
    normalized_hermite,
)
from hebbian.mlp_core.task import SharedConstructionConfig

if TYPE_CHECKING:
    from hebbian.data.synthetics.factsets import Factset


@pydraclass
class NTKConstructionConfig:
    """Configuration for the finite-width NTK construction."""

    shared: SharedConstructionConfig = field(default_factory=SharedConstructionConfig)
    m: Optional[int] = None
    hermite_degree: int = 1
    loss_fn: str = "mse"


@dataclass
class NTKParameters:
    up_weights: torch.Tensor
    gate_weights: torch.Tensor
    down_weights: torch.Tensor


def construct_ntk_parameters(
    factset: Factset, config: NTKConstructionConfig
) -> NTKParameters:
    """Construct the three matrices in the paper's gated NTK baseline."""

    if config.m is None:
        raise ValueError("NTKConstructionConfig.m must be set")
    m = int(config.m)
    gate_weights = torch.randn(
        m,
        factset.input_embeddings.shape[1],
        dtype=config.shared.build_dtype,
        device=config.shared.device,
    )
    down_weights = torch.randn(
        factset.output_embeddings.shape[1],
        m,
        dtype=config.shared.build_dtype,
        device=config.shared.device,
    )
    down_weights = torch.nn.functional.normalize(down_weights, dim=0)
    projections = factset.input_embeddings @ gate_weights.T
    features = normalized_hermite(config.hermite_degree, projections)
    coefficients = factset.output_embeddings[factset.mapping.outputs] @ down_weights
    up_weights = (features * coefficients).T @ factset.input_embeddings / m
    return NTKParameters(up_weights, gate_weights, down_weights)


def build_ntk_mlp_from_params(
    parameters: NTKParameters, config: NTKConstructionConfig
) -> GatedMLP:
    activation = config.shared.mlp_config.activation.get_activation()
    up = GatedLinearBlock.from_weights(
        GatedLinearBlockWeights(
            main_weight=parameters.up_weights,
            main_bias=torch.zeros_like(parameters.up_weights[:, 0]),
            gate_weight=parameters.gate_weights,
            gate_bias=torch.zeros_like(parameters.gate_weights[:, 0]),
            gate_fn=activation,
        )
    )
    down = LinearBlock.from_weights(
        LinearBlockWeights(
            weight=parameters.down_weights,
            bias=torch.zeros_like(parameters.down_weights[:, 0]),
        )
    )
    return GatedMLP(up, down)


def get_ntk_mlp(
    factset: Factset, config: NTKConstructionConfig
) -> tuple[GatedMLP, dict]:
    """Construct the paper's gated NTK MLP for ``factset``."""

    factset = factset.to(
        device=config.shared.device, dtype=config.shared.build_dtype
    )
    parameters = construct_ntk_parameters(factset, config)
    return build_ntk_mlp_from_params(parameters, config), {}
