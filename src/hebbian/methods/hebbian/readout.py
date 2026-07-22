"""Readouts used by the three paper Hebbian constructions."""

from __future__ import annotations

import torch


_CUDA_SOLVE_ERROR_TOKENS = (
    "CUBLAS_STATUS_EXECUTION_FAILED",
    "cublasDtrsm",
    "CUSOLVER_STATUS_INTERNAL_ERROR",
    "cusolver error",
)


def solve_linear_system(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve ``A X = B``, falling back to CPU for known CUDA solver failures."""
    try:
        return torch.linalg.solve(A, B)
    except RuntimeError as exc:
        if A.device.type != "cuda" or not any(
            token in str(exc) for token in _CUDA_SOLVE_ERROR_TOKENS
        ):
            raise
    solution = torch.linalg.solve(A.detach().cpu(), B.detach().cpu())
    return solution.to(device=A.device, dtype=A.dtype)


def raw_readout(values: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Return the raw Hebbian correlation ``V.T @ Phi / n``."""
    return (values.T @ features) / features.shape[0]


def full_ridge_readout(
    values: torch.Tensor,
    features: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Return the full-whitened Hebbian readout used in the paper."""
    n, m = features.shape
    if m > n:
        gram = features @ features.T
        reg = gram + ridge * torch.eye(n, device=gram.device, dtype=gram.dtype)
        coefficients = solve_linear_system(reg, values)
        return coefficients.T @ features

    covariance = (features.T @ features) / n
    correlation = raw_readout(values, features)
    reg = covariance + ridge * torch.eye(
        m, device=covariance.device, dtype=covariance.dtype
    )
    return solve_linear_system(reg, correlation.T).T
