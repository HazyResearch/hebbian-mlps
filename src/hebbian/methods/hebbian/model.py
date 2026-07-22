"""Bilinear feature map and constructed Hebbian MLP."""

from __future__ import annotations

import torch
from torch import nn


class BilinearFeatureMap(nn.Module):
    """Compute ``phi(x) = (A0 x) * (A1 x)``.

    Random paper features use the conventional ``1 / sqrt(m)`` normalization.
    Fitted data-dependent features retain their native scale. This distinction
    is fixed by the construction and is not exposed as an experiment knob.
    """

    def __init__(
        self,
        A0: torch.Tensor,
        A1: torch.Tensor,
        *,
        normalize: bool,
    ) -> None:
        super().__init__()
        if A0.ndim != 2 or A0.shape != A1.shape:
            raise ValueError(
                f"A0 and A1 must be matching matrices; got {A0.shape} and {A1.shape}"
            )
        self.register_buffer("A0", A0)
        self.register_buffer("A1", A1)
        self._scale = A0.shape[0] ** -0.5 if normalize else 1.0

    @property
    def out_dim(self) -> int:
        return int(self.A0.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = (x @ self.A0.T) * (x @ self.A1.T)
        return features * self._scale


class HebbianMLP(nn.Module):
    """A fixed bilinear feature map followed by a constructed readout."""

    def __init__(self, feature_map: BilinearFeatureMap, W: torch.Tensor) -> None:
        super().__init__()
        if W.ndim != 2 or W.shape[1] != feature_map.out_dim:
            raise ValueError(
                f"W must have shape (d_out, {feature_map.out_dim}); got {W.shape}"
            )
        self.feature_map = feature_map
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.ndim == 3:
            x = x.reshape(-1, original_shape[-1])
        output = self.feature_map(x) @ self.W.T
        if len(original_shape) == 3:
            output = output.reshape(original_shape[0], original_shape[1], -1)
        return output

    def weight_count(self) -> int:
        """Count the two feature matrices and readout matrix."""
        return sum(tensor.numel() for tensor in self.buffers())
