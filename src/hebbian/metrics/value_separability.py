"""Fit optimal value separators and measure value-set separability."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _build_constraint_batch(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Build normalized pairwise-difference constraints for selected anchors."""

    device = values.device
    dtype = values.dtype
    num_values, dim = values.shape
    batch_size = indices.shape[0]

    differences = values[indices, None, :] - values[None, :, :]
    norms = differences.norm(dim=-1)
    mask = torch.ones(batch_size, num_values, dtype=torch.bool, device=device)
    mask[torch.arange(batch_size, device=device), indices] = False
    norms = norms + (~mask).to(dtype)
    normalized = differences / norms.unsqueeze(-1)
    return normalized[mask].view(batch_size, num_values - 1, dim)


def _fit_separator_batch(
    constraints: torch.Tensor,
    *,
    rho: float,
    num_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a batch of max-min unit-separator problems with ADMM."""

    device = constraints.device
    dtype = constraints.dtype
    batch_size, num_constraints, dim = constraints.shape

    u = torch.zeros(batch_size, dim, 1, device=device, dtype=dtype)
    margin = torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)
    slack = torch.zeros(batch_size, num_constraints, 1, device=device, dtype=dtype)
    projected_u = torch.zeros(batch_size, dim, 1, device=device, dtype=dtype)
    constraint_dual = torch.zeros_like(slack)
    norm_dual = torch.zeros_like(projected_u)

    ones = torch.ones(batch_size, num_constraints, 1, device=device, dtype=dtype)
    identity = torch.eye(dim, device=device, dtype=dtype)
    constraints_t = constraints.transpose(1, 2)
    gram = constraints_t @ constraints

    system = torch.zeros(batch_size, dim + 1, dim + 1, device=device, dtype=dtype)
    system[:, :dim, :dim] = rho * (gram + identity)
    system[:, :dim, dim] = -rho * (constraints_t @ ones).squeeze(-1)
    system[:, dim, :dim] = -rho * (ones.transpose(1, 2) @ constraints).squeeze(1)
    system[:, dim, dim] = rho * float(num_constraints)

    for _ in range(num_iters):
        constraint_residual = slack - constraint_dual
        norm_residual = projected_u - norm_dual
        rhs_u = rho * (
            (constraints_t @ constraint_residual).squeeze(-1)
            + norm_residual.squeeze(-1)
        )
        rhs_margin = torch.ones(batch_size, 1, device=device, dtype=dtype) - rho * (
            ones.transpose(1, 2) @ constraint_residual
        ).squeeze(1)
        solution = torch.linalg.solve(
            system,
            torch.cat([rhs_u, rhs_margin], dim=1).unsqueeze(-1),
        ).squeeze(-1)

        u = solution[:, :dim].unsqueeze(-1)
        margin = solution[:, dim:].unsqueeze(-1)

        violations = constraints @ u - margin * ones
        slack = torch.clamp(violations + constraint_dual, min=0.0)

        shifted_u = u + norm_dual
        norms = torch.norm(shifted_u, dim=1, keepdim=True)
        scale = torch.clamp(1.0 / torch.clamp(norms, min=1e-12), max=1.0)
        projected_u = shifted_u * scale

        constraint_dual = constraint_dual + violations - slack
        norm_dual = norm_dual + u - projected_u

    return u.squeeze(-1), margin.squeeze(-1).squeeze(-1)


def fit_value_separators(
    values: torch.Tensor,
    *,
    batch_size: int = 256,
    rho: float = 1.0,
    num_iters: int = 300,
    verbose: bool = False,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Fit optimal unit separators and return global/per-value separability.

    For each value ``v_i``, this solves

    ``max_{||u_i|| <= 1} min_{j != i} <(v_i-v_j)/||v_i-v_j||, u_i>``.

    Returns the unit-normalized separator witnesses, the minimum separability
    across values, and the per-value separabilities.
    """

    if values.ndim != 2:
        raise ValueError(f"values must be a 2D tensor, got shape {tuple(values.shape)}")
    if values.shape[0] < 2:
        raise ValueError("value separability requires at least two values")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    num_values = values.shape[0]
    separators = torch.zeros_like(values)
    per_value = torch.empty(num_values, device=values.device, dtype=values.dtype)
    all_indices = torch.arange(num_values, device=values.device)

    starts = range(0, num_values, batch_size)
    if verbose:
        starts = tqdm(starts, desc="Fitting value separators")

    for start in starts:
        end = min(start + batch_size, num_values)
        constraints = _build_constraint_batch(values, all_indices[start:end])
        separator_batch, margin_batch = _fit_separator_batch(
            constraints,
            rho=rho,
            num_iters=num_iters,
        )
        separators[start:end] = separator_batch
        per_value[start:end] = margin_batch

    normalized_separators = F.normalize(separators, dim=1)
    return normalized_separators, float(per_value.min().item()), per_value
