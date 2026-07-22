"""Linear output block used by the released gated MLPs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LinearBlockWeights:
    weight: torch.Tensor
    bias: torch.Tensor | None


class LinearBlock(nn.Module):
    """Thin wrapper that preserves the public ``.linear`` edit interface."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
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
        self.linear = nn.Linear(
            in_dim, out_dim, bias=bias, dtype=dtype, device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        x_gate: torch.Tensor | None = None,
        debug: bool = False,
    ) -> torch.Tensor:
        del x_gate, debug
        if x.dtype != self.linear.weight.dtype:
            self.linear = self.linear.to(dtype=x.dtype)
        return self.linear(x)

    def print_shapes(
        self,
        batch_size: int | None = None,
        indent: int = 0,
        return_str: bool = False,
    ) -> str:
        batch = "B" if batch_size is None else str(batch_size)
        text = f"{' ' * indent}LinearBlock: {self.in_dim}->{self.out_dim} [{batch},{self.out_dim}]"
        if not return_str:
            print(text)
        return text

    @classmethod
    def from_weights(cls, weights: LinearBlockWeights) -> "LinearBlock":
        block = cls(
            weights.weight.shape[1],
            weights.weight.shape[0],
            bias=weights.bias is not None,
            dtype=weights.weight.dtype,
            device=weights.weight.device,
        )
        with torch.no_grad():
            block.linear.weight.copy_(weights.weight)
            if weights.bias is not None:
                block.linear.bias.copy_(weights.bias)
        return block


__all__ = ["LinearBlock", "LinearBlockWeights"]
