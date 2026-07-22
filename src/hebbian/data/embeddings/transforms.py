"""
Embedding transformations.

This module provides utilities for transforming embeddings:
    - Normalization (L2 norm to unit sphere)
    - Conditioning (modify condition number of embedding matrix)
    - Spiked covariance (rank-1 spike to control coherence)
"""

from typing import Optional

import torch


def normalize_embeddings(
    embeddings: torch.Tensor,
    p: float = 2.0,
    dim: int = -1,
) -> torch.Tensor:
    """
    Normalize embeddings to unit norm.

    Args:
        embeddings: Embedding tensor
        p: Norm order (default: L2)
        dim: Dimension to normalize over

    Returns:
        Normalized embeddings
    """
    return torch.nn.functional.normalize(embeddings, p=p, dim=dim)


def condition_embeddings(
    embeddings: torch.Tensor,
    target_condition_number: float,
    method: str = "svd",
) -> torch.Tensor:
    """
    Modify the condition number of the embedding matrix.

    Adjusts singular values to achieve the target condition number
    while preserving the singular vector structure.

    Args:
        embeddings: Embedding tensor (n, d)
        target_condition_number: Desired condition number (>= 1)
        method: Method for modification ("svd")

    Returns:
        Conditioned embeddings
    """
    if target_condition_number < 1:
        raise ValueError("Condition number must be >= 1")

    if method != "svd":
        raise ValueError(f"Unknown method: {method}. Only 'svd' is supported.")

    # SVD
    U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)

    # Current condition number
    current_cond = (S[0] / S[-1]).item()
    if len(S) < 2:
        # Condition number is fixed at 1 for 1D singular spectrum.
        return embeddings
    if abs(current_cond - target_condition_number) / max(target_condition_number, 1e-12) < 1e-6:
        return embeddings

    # Modify singular values to achieve target condition number
    # Map singular values to range [1/kappa, 1] linearly
    n_singular = len(S)
    target_s = torch.linspace(
        1.0,
        1.0 / target_condition_number,
        n_singular,
        dtype=embeddings.dtype,
        device=embeddings.device,
    )

    # Scale to preserve Frobenius norm
    scale = S.norm() / target_s.norm()
    target_s = target_s * scale

    # Reconstruct
    return U @ torch.diag(target_s) @ Vh


def spike_embeddings(
    embeddings: torch.Tensor,
    beta: float,
    spike_direction: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Apply a rank-1 spiked covariance transform to embeddings.

    Transforms each embedding by amplifying its component along a spike
    direction u:

        e_i' = (I + β u u^T) e_i = e_i + β (e_i · u) u

    Then (optionally) re-projects to the unit sphere.  As β increases,
    embeddings concentrate along u and coherence grows monotonically.

    This is equivalent to applying the square-root of a spiked covariance
    model Σ = I + β' u u^T (where β' = β² + 2β) to isotropic samples.

    Args:
        embeddings: Embedding tensor (n, d), typically on the unit sphere.
        beta: Spike strength (>= 0).  β = 0 is a no-op.
        spike_direction: Unit vector (d,) defining the spike axis.
            If None, a random direction is drawn (seeded by `seed`).
        seed: Random seed used to generate spike_direction when it is None.
        normalize: If True (default), re-project to the unit sphere after
            spiking.

    Returns:
        Spiked embeddings (n, d).  Unit-norm if normalize=True.
    """
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}")

    if beta == 0:
        return embeddings

    d = embeddings.shape[1]

    # Determine spike direction
    if spike_direction is None:
        gen = torch.Generator(device=embeddings.device)
        if seed is not None:
            gen.manual_seed(seed)
        spike_direction = torch.randn(d, dtype=embeddings.dtype, device=embeddings.device, generator=gen)
        spike_direction = spike_direction / spike_direction.norm()
    else:
        if spike_direction.shape != (d,):
            raise ValueError(
                f"spike_direction must have shape ({d},), got {spike_direction.shape}"
            )
        # Ensure unit norm
        spike_direction = spike_direction / spike_direction.norm()

    # Apply (I + β u u^T) to each row:  e_i + β (e_i · u) u
    projections = embeddings @ spike_direction          # (n,)
    spiked = embeddings + beta * projections.unsqueeze(1) * spike_direction.unsqueeze(0)

    if normalize:
        spiked = torch.nn.functional.normalize(spiked, dim=-1)

    return spiked


def compute_condition_number(embeddings: torch.Tensor) -> float:
    """
    Compute the condition number of the embedding matrix.

    Args:
        embeddings: Embedding tensor (n, d)

    Returns:
        Condition number (ratio of largest to smallest singular value)
    """
    S = torch.linalg.svdvals(embeddings)
    return (S[0] / S[-1]).item()


__all__ = [
    "normalize_embeddings",
    "condition_embeddings",
    "spike_embeddings",
    "compute_condition_number",
]
