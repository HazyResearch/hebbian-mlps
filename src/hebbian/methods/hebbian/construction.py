"""Random and data-dependent bilinear Hebbian constructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from hebbian.core.dtype_utils import get_dtype
from hebbian.data.synthetics.factsets import Factset
from hebbian.mlp_core.task import SharedConstructionConfig

from .model import BilinearFeatureMap, HebbianMLP
from .readout import full_ridge_readout, raw_readout


def _factset_tensors(
    factset: Factset,
    shared: SharedConstructionConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.device, torch.dtype]:
    device = torch.device(shared.device) if shared.device is not None else torch.device("cpu")
    dtype = get_dtype(shared.build_dtype)
    keys = factset.input_embeddings.to(device=device, dtype=dtype)
    value_indices = [factset.mapping.get_output(i) for i in range(keys.shape[0])]
    values = factset.output_embeddings[value_indices].to(device=device, dtype=dtype)
    return keys, values, value_indices, device, dtype


def _random_feature_map(
    d: int,
    m: int,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    normalize: bool,
) -> BilinearFeatureMap:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    A0 = torch.randn((m, d), generator=generator, dtype=dtype).to(device)
    A1 = torch.randn((m, d), generator=generator, dtype=dtype).to(device)
    return BilinearFeatureMap(A0, A1, normalize=normalize)


@dataclass
class _FeatureFitSystem:
    partner_features: torch.Tensor
    keys: torch.Tensor
    keys_t: torch.Tensor
    readout: torch.Tensor
    value_gram: torch.Tensor
    rhs: torch.Tensor
    cg_diagonal: torch.Tensor
    ridge: float


def _build_feature_fit_system(
    partner_features: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    readout: torch.Tensor,
    ridge: float,
    solve_dtype: torch.dtype,
) -> _FeatureFitSystem:
    C = partner_features.to(solve_dtype)
    K = keys.to(solve_dtype)
    W = readout.to(solve_dtype)
    V = values.to(solve_dtype)
    value_gram = V.T @ V
    readout_targets = W.T @ V.T
    rhs = (readout_targets * C) @ K
    readout_diag = (W * (value_gram @ W)).sum(dim=0)
    cg_diagonal = readout_diag[:, None] * (C.square() @ K.square()) + ridge
    return _FeatureFitSystem(
        partner_features=C,
        keys=K,
        keys_t=K.T,
        readout=W,
        value_gram=value_gram,
        rhs=rhs,
        cg_diagonal=cg_diagonal,
        ridge=float(ridge),
    )


def _apply_feature_fit_operator(
    candidate: torch.Tensor,
    system: _FeatureFitSystem,
) -> torch.Tensor:
    features = system.partner_features * (candidate @ system.keys_t)
    decoded = system.readout @ features
    coupled = system.readout.T @ (system.value_gram @ decoded)
    return (
        (system.partner_features * coupled) @ system.keys
        + system.ridge * candidate
    )


def _fit_projection(
    partner_features: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    readout: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    m, _ = partner_features.shape
    d = keys.shape[1]
    solve_dtype = torch.float32 if m * d > 8192 else keys.dtype
    system = _build_feature_fit_system(
        partner_features, keys, values, readout, ridge, solve_dtype
    )
    rhs = system.rhs
    solution = torch.zeros_like(rhs)
    rhs_norm = float(torch.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return solution.to(keys.dtype)

    tolerance = (1e-5 if solve_dtype == torch.float32 else 1e-10) * rhs_norm
    max_iterations = min(max(2 * m, 128), 1024)
    inverse_diag = system.cg_diagonal.clamp_min(
        max(ridge, torch.finfo(solve_dtype).eps)
    ).reciprocal()
    residual = rhs.clone()
    scaled_residual = inverse_diag * residual
    direction = scaled_residual.clone()
    residual_dot = torch.sum(residual * scaled_residual)

    for _ in range(max_iterations):
        operator_direction = _apply_feature_fit_operator(direction, system)
        denominator = torch.sum(direction * operator_direction)
        if float(denominator) <= 0.0:
            raise RuntimeError("Data-dependent feature fit lost positive definiteness")
        step = residual_dot / denominator
        solution = solution + step * direction
        residual = residual - step * operator_direction
        if float(torch.linalg.norm(residual)) <= tolerance:
            break
        scaled_residual = inverse_diag * residual
        next_residual_dot = torch.sum(residual * scaled_residual)
        direction = scaled_residual + (next_residual_dot / residual_dot) * direction
        residual_dot = next_residual_dot

    return solution.to(keys.dtype)


def _data_dependent_feature_map(
    keys: torch.Tensor,
    values: torch.Tensor,
    m: int,
    *,
    seed: int,
    ridge: float,
) -> BilinearFeatureMap:
    initial = _random_feature_map(
        keys.shape[1],
        m,
        seed=seed,
        dtype=keys.dtype,
        device=keys.device,
        normalize=False,
    )
    A0 = initial.A0.detach().clone()
    A1 = initial.A1.detach().clone()
    initial_features = (keys @ A0.T) * (keys @ A1.T)
    initial_readout = raw_readout(values, initial_features)

    A1 = _fit_projection(A0 @ keys.T, keys, values, initial_readout, ridge)
    A0 = _fit_projection(A1 @ keys.T, keys, values, initial_readout, ridge)
    return BilinearFeatureMap(A0.detach(), A1.detach(), normalize=False)


def _metrics(
    mlp: HebbianMLP,
    factset: Factset,
    keys: torch.Tensor,
    values: torch.Tensor,
    value_indices: list[int],
) -> dict[str, Any]:
    with torch.no_grad():
        outputs = mlp(keys)
        candidates = factset.output_embeddings.to(device=keys.device, dtype=keys.dtype)
        predictions = (outputs @ candidates.T).argmax(dim=-1)
        targets = torch.tensor(value_indices, device=keys.device)
        accuracy = float((predictions == targets).float().mean())
        mse = float(F.mse_loss(outputs, values))
    return {"accuracy": accuracy, "mse": mse, "feature_dim": mlp.feature_map.out_dim}


def construct_hebbian_mlp(
    factset: Factset,
    config: Any,
    *,
    seed: int,
) -> tuple[HebbianMLP, dict[str, Any]]:
    """Construct one of the three Hebbian MLP variants used in the paper."""
    valid_variants = {"unwhitened", "whitened", "data_dependent"}
    if config.variant not in valid_variants:
        raise ValueError(
            f"variant must be one of {sorted(valid_variants)}; got {config.variant!r}"
        )
    if config.ridge < 0:
        raise ValueError(f"ridge must be non-negative; got {config.ridge}")

    keys, values, value_indices, device, dtype = _factset_tensors(
        factset, config.shared
    )
    m = int(config.m) if config.m is not None else 4 * keys.shape[1]
    if m <= 0:
        raise ValueError(f"m must be positive; got {m}")

    if config.variant == "data_dependent":
        feature_map = _data_dependent_feature_map(
            keys, values, m, seed=seed, ridge=config.ridge
        )
    else:
        feature_map = _random_feature_map(
            keys.shape[1],
            m,
            seed=seed,
            dtype=dtype,
            device=device,
            normalize=True,
        )

    with torch.no_grad():
        features = feature_map(keys)
    if config.variant == "unwhitened":
        W = raw_readout(values, features)
    else:
        W = full_ridge_readout(values, features, config.ridge)

    mlp = HebbianMLP(feature_map, W.detach())
    return mlp, _metrics(mlp, factset, keys, values, value_indices)
