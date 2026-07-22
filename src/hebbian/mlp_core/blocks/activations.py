"""Activation functions used by the paper MLPs."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from hebbian.config import pydraclass


class Activation(nn.Module):
    """Small named wrapper around an elementwise activation."""

    def __init__(self, function: Callable[[torch.Tensor], torch.Tensor], name: str):
        super().__init__()
        self.function = function
        self._name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.function(x)

    def name(self) -> str:
        return self._name


def get_activation_from_string(activation: str) -> Activation:
    """Return one of the gate functions used in the released experiments."""

    normalized = activation.lower()
    if normalized == "relu":
        return Activation(torch.relu, "ReLU")
    if normalized in {"identity", "linear"}:
        return Activation(lambda x: x, "identity")
    if normalized in {"silu", "swish"}:
        return Activation(torch.nn.functional.silu, normalized)
    raise ValueError(
        f"Unknown activation {activation!r}; expected relu, identity, silu, or swish"
    )


@pydraclass
class ActivationConfig:
    """Gate activation for GD and NTK MLPs."""

    activation: str = "relu"

    def get_activation(self) -> Activation:
        return get_activation_from_string(self.activation)


def normalized_hermite(degree: int, x: torch.Tensor) -> torch.Tensor:
    """Evaluate the normalized probabilists' Hermite polynomial ``h_degree``."""

    if degree < 0:
        raise ValueError("degree must be non-negative")
    h0 = torch.ones_like(x)
    if degree == 0:
        return h0
    h1 = x
    if degree == 1:
        return h1
    hm2, hm1 = h0, h1
    for n in range(1, degree):
        hp1 = (x * hm1 - (n**0.5) * hm2) / ((n + 1) ** 0.5)
        hm2, hm1 = hm1, hp1
    return hm1


__all__ = [
    "Activation",
    "ActivationConfig",
    "get_activation_from_string",
    "normalized_hermite",
]
