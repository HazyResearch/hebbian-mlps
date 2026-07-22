"""Paper gated MLP architecture."""

from __future__ import annotations

import torch
from torch import nn

from hebbian.mlp_core.blocks.gated_block import GatedLinearBlock
from hebbian.mlp_core.blocks.linear_block import LinearBlock


class GatedMLP(nn.Module):
    """A gated hidden projection followed by a linear readout."""

    def __init__(self, up: GatedLinearBlock, down: LinearBlock):
        super().__init__()
        if up.out_dim != down.in_dim:
            raise ValueError(
                f"Hidden dimensions differ: {up.out_dim} != {down.in_dim}"
            )
        self.in_dim = up.in_dim
        self.out_dim = down.out_dim
        self.dtype = up.dtype
        self.device = up.device
        self.up = up
        self.down = down

    def forward(
        self,
        x: torch.Tensor,
        x_gate: torch.Tensor | None = None,
        debug: bool = False,
    ) -> torch.Tensor:
        return self.down(self.up(x, x_gate=x_gate, debug=debug), debug=debug)

    def print_shapes(
        self,
        batch_size: int | None = None,
        indent: int = 0,
        return_str: bool = False,
    ) -> str:
        batch = "B" if batch_size is None else str(batch_size)
        text = f"{' ' * indent}GatedMLP: {self.in_dim}->{self.out_dim} [{batch},{self.out_dim}]"
        children = (
            self.up.print_shapes(batch_size, indent + 2, True),
            self.down.print_shapes(batch_size, indent + 2, True),
        )
        result = "\n".join((text, *children))
        if not return_str:
            print(result)
        return result


MLP = GatedMLP

__all__ = ["GatedMLP", "MLP"]
