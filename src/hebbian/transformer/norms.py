"""Normalization layers used by the simplified GPT2 model."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, ndim, bias, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.register_buffer("eps", torch.tensor(eps))

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class FrozenRMSNorm(nn.Module):
    """RMSNorm with a frozen scalar scale set to match embedding RMS norms.

    During construction the scale is uninitialized. Call
    ``get_frozen_rms_scale(embeddings)`` once before the first forward pass
    to lock the scale to the mean RMS norm of the provided embeddings.
    """

    def __init__(self, ndim, eps=0.0):
        super().__init__()
        self.register_buffer("weight", torch.empty(()))
        self.register_buffer("eps", torch.tensor(eps))
        self.ndim = ndim
        self.set = False

    def forward(self, input):
        return F.rms_norm(input, (self.ndim,), eps=self.eps) * self.weight

    @torch.no_grad()
    def get_frozen_rms_scale(self, mlp_embeddings, enforce_small_err=True):
        if self.set:
            return
        self.set = True
        mlp_embeddings = mlp_embeddings.weight.data
        mlp_embeddings_rmsnorm = torch.sqrt(
            torch.mean(mlp_embeddings**2, dim=1) + self.eps
        )
        mean_mlp_embeddings_rmsnorm = mlp_embeddings_rmsnorm.mean()

        if enforce_small_err:
            assert torch.all(
                torch.isclose(
                    mean_mlp_embeddings_rmsnorm
                    * torch.ones_like(mlp_embeddings_rmsnorm),
                    mlp_embeddings_rmsnorm,
                    rtol=0.2,
                )
            ), "MLP embeddings must all have approximately the same RMS norm to use FrozenRMSNorm"

        self.register_buffer("weight", mean_mlp_embeddings_rmsnorm)


class UnitRMSNorm(nn.Module):
    """L2 norm that normalizes outputs to unit L2 norm (scale = 1).
    
    Unlike FrozenRMSNorm which scales to match embedding norms, this always
    outputs vectors with L2 norm of exactly 1. Useful for attention-only
    experiments where we want normalized outputs before the lm_head.
    """

    def __init__(self, ndim, eps=1e-30):
        super().__init__()
        self.ndim = ndim
        self.register_buffer("eps", torch.tensor(eps))

    def forward(self, input):
        l2_norm = torch.norm(input, p=2, dim=-1, keepdim=True) + self.eps
        return input / l2_norm


def get_norm(norm_type: str, ndim: int, bias: bool) -> nn.Module:
    """Factory for normalization layers.

    Args:
        norm_type: One of "layernorm", "rmsnorm", "frozen_rmsnorm", "unit_rmsnorm", "none".
        ndim: Normalization dimension.
        bias: Whether to include a bias term (only applies to layernorm).

    Returns:
        The requested normalization module.
    """
    if norm_type == "layernorm":
        return LayerNorm(ndim, bias, eps=1e-30)
    elif norm_type == "rmsnorm":
        return nn.RMSNorm(ndim, elementwise_affine=bias, eps=1e-30)
    elif norm_type == "frozen_rmsnorm":
        return FrozenRMSNorm(ndim, eps=1e-30)
    elif norm_type == "unit_rmsnorm":
        return UnitRMSNorm(ndim, eps=1e-30)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
