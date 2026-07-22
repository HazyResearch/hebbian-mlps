"""Gated hidden block used by the released MLPs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hebbian.mlp_core.blocks.activations import Activation


@dataclass
class GatedLinearBlockWeights:
    main_weight: torch.Tensor
    main_bias: torch.Tensor | None
    gate_weight: torch.Tensor
    gate_bias: torch.Tensor | None
    gate_fn: Activation


class GatedLinearBlock(nn.Module):
    """Compute ``(W_main x + b_main) * gate(W_gate x_gate + b_gate)``."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        gate_fn: Activation,
        *,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dtype = dtype
        self.device = device
        self.main_linear = nn.Linear(
            in_dim, out_dim, bias=bias, dtype=dtype, device=device
        )
        self.gate_linear = nn.Linear(
            in_dim, out_dim, bias=bias, dtype=dtype, device=device
        )
        self.gate_fn = gate_fn

    def forward(
        self,
        x: torch.Tensor,
        x_gate: torch.Tensor | None = None,
        debug: bool = False,
    ) -> torch.Tensor:
        del debug
        if x_gate is None:
            x_gate = x
        if x.dtype != self.main_linear.weight.dtype:
            self.main_linear = self.main_linear.to(dtype=x.dtype)
            self.gate_linear = self.gate_linear.to(dtype=x.dtype)
        return self.main_linear(x) * self.gate_fn(self.gate_linear(x_gate))

    def print_shapes(
        self,
        batch_size: int | None = None,
        indent: int = 0,
        return_str: bool = False,
    ) -> str:
        batch = "B" if batch_size is None else str(batch_size)
        text = (
            f"{' ' * indent}GatedLinearBlock(gate_fn={self.gate_fn.name()}): "
            f"{self.in_dim}->{self.out_dim} [{batch},{self.out_dim}]"
        )
        if not return_str:
            print(text)
        return text

    @classmethod
    def from_weights(cls, weights: GatedLinearBlockWeights) -> "GatedLinearBlock":
        has_bias = weights.main_bias is not None
        block = cls(
            weights.main_weight.shape[1],
            weights.main_weight.shape[0],
            weights.gate_fn,
            bias=has_bias,
            dtype=weights.main_weight.dtype,
            device=weights.main_weight.device,
        )
        with torch.no_grad():
            block.main_linear.weight.copy_(weights.main_weight)
            block.gate_linear.weight.copy_(weights.gate_weight)
            if has_bias:
                block.main_linear.bias.copy_(weights.main_bias)
                block.gate_linear.bias.copy_(weights.gate_bias)
        return block


__all__ = ["GatedLinearBlock", "GatedLinearBlockWeights"]
