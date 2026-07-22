"""Shared helpers for MLP demo scripts."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from hebbian.data.synthetics import generate_factset


def resolve_device(device: Optional[str]) -> str:
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def parse_dtype(name: Optional[str]) -> torch.dtype:
    if name is None:
        return torch.float64
    normalized = name.strip().lower()
    if normalized in {"float32", "fp32", "f32"}:
        return torch.float32
    if normalized in {"float64", "fp64", "f64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def make_factset(
    d_model: int,
    facts_multiplier: float,
    *,
    device: str,
    dtype: torch.dtype,
    embedding_init: str = "spherical",
    tie_embeddings: bool = True,
):
    vocab_size = int(facts_multiplier * d_model * d_model)
    factset = generate_factset(
        d_model=d_model,
        vocab_size=vocab_size,
        embedding_init=embedding_init,
        tie_embeddings=tie_embeddings,
    )
    return factset.to(device=device, dtype=dtype)


def forward_mlp(mlp, factset):
    return mlp(factset.input_embeddings)


def compute_accuracy(mlp, factset) -> float:
    with torch.no_grad():
        output = forward_mlp(mlp, factset)
        predictions = output @ factset.output_embeddings.T
        predicted_indices = torch.argmax(predictions, dim=-1)
        targets = torch.tensor(
            factset.mapping.outputs, dtype=torch.long, device=output.device
        )
        return (predicted_indices == targets).float().mean().item()


def count_parameters(mlp) -> int:
    if hasattr(mlp, "weight_count"):
        return mlp.weight_count()
    return sum(p.numel() for p in mlp.parameters()) + sum(
        b.numel() for b in mlp.buffers()
    )


def print_header(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_config(items: Iterable[tuple[str, object]]) -> None:
    print("\nConfig:")
    for key, value in items:
        print(f"  {key}: {value}")


def print_metrics(metrics: dict) -> None:
    if not metrics:
        return
    print("\nMetrics:")
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, (int, float, str, bool)):
            print(f"  {key}: {value}")


def print_results(accuracy: float, param_count: int) -> None:
    print("\n" + "-" * 60)
    print("Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Parameters: {param_count:,}")
    print("-" * 60)
