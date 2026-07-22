"""Shared helpers for LLM activation bundles.

The public bundle format is intentionally small:

    <artifact_root>/activations/x.pt
    <artifact_root>/activations/y.pt
    <artifact_root>/activations/metadata.json

Downstream experiment code consumes the activation directory directly, while
paper scripts use the artifact root so related result files can live nearby.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hebbian.data.synthetics.factsets import Factset, create_identity_mapping


FORMAT_VERSION = "hebbian.llm_embeddings.v1"


@dataclass(frozen=True)
class ActivationBundleInfo:
    """Shape and metadata summary for a loaded activation bundle."""

    activation_dir: Path
    x_path: Path
    y_path: Path
    metadata_path: Path | None
    num_pairs: int
    d_model: int
    x_dtype: str
    y_dtype: str
    metadata: dict[str, Any]


def activation_dir_from_root(path: str | Path) -> Path:
    """Return the directory containing x.pt/y.pt from a root or direct path."""

    root = Path(path).expanduser()
    if (root / "x.pt").is_file() and (root / "y.pt").is_file():
        return root
    return root / "activations"


def load_metadata(activation_dir: str | Path) -> dict[str, Any]:
    """Load metadata.json when present; otherwise return an empty dict."""

    metadata_path = Path(activation_dir) / "metadata.json"
    if not metadata_path.is_file():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"metadata.json must contain a JSON object: {metadata_path}")
    return payload


def write_metadata(activation_dir: str | Path, metadata: dict[str, Any]) -> Path:
    """Write metadata.json with stable indentation."""

    out_path = Path(activation_dir) / "metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def load_activation_pair(
    activation_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load x.pt/y.pt from an activation directory."""

    root = Path(activation_dir).expanduser()
    x_path = root / "x.pt"
    y_path = root / "y.pt"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(
            f"activation directory must contain x.pt and y.pt; got {root}"
        )
    x = torch.load(x_path, map_location=map_location)
    y = torch.load(y_path, map_location=map_location)
    return x, y


def validate_activation_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    min_rows: int = 1,
    expected_d_model: int | None = None,
    require_finite: bool = True,
) -> tuple[int, int]:
    """Validate tensor shape/dtype and return ``(num_pairs, d_model)``."""

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x.pt and y.pt must deserialize to torch.Tensor objects")
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError(
            "x.pt and y.pt must be 2D tensors with matching shape; "
            f"got x={tuple(x.shape)}, y={tuple(y.shape)}"
        )
    n_rows, d_model = int(x.shape[0]), int(x.shape[1])
    if n_rows < int(min_rows):
        raise ValueError(f"bundle has {n_rows} rows, expected at least {min_rows}")
    if expected_d_model is not None and d_model != int(expected_d_model):
        raise ValueError(
            f"bundle d_model={d_model}, expected {int(expected_d_model)}"
        )
    if not torch.is_floating_point(x) or not torch.is_floating_point(y):
        raise TypeError(f"x/y must be floating tensors; got {x.dtype} and {y.dtype}")
    if require_finite:
        if not torch.isfinite(x).all():
            raise ValueError("x.pt contains NaN or Inf values")
        if not torch.isfinite(y).all():
            raise ValueError("y.pt contains NaN or Inf values")
    return n_rows, d_model


def inspect_bundle(
    path: str | Path,
    *,
    min_rows: int = 1,
    expected_d_model: int | None = None,
) -> ActivationBundleInfo:
    """Load, validate, and summarize a bundle root or activation directory."""

    activation_dir = activation_dir_from_root(path)
    x, y = load_activation_pair(activation_dir)
    n_rows, d_model = validate_activation_pair(
        x,
        y,
        min_rows=min_rows,
        expected_d_model=expected_d_model,
    )
    metadata_path = activation_dir / "metadata.json"
    metadata = load_metadata(activation_dir)
    return ActivationBundleInfo(
        activation_dir=activation_dir,
        x_path=activation_dir / "x.pt",
        y_path=activation_dir / "y.pt",
        metadata_path=metadata_path if metadata_path.is_file() else None,
        num_pairs=n_rows,
        d_model=d_model,
        x_dtype=str(x.dtype),
        y_dtype=str(y.dtype),
        metadata=metadata,
    )


def make_factset_from_activation_rows(
    path: str | Path,
    *,
    num_facts: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> Factset:
    """Sample paired activation rows and expose them as an identity-map factset.

    ``path`` may be either the artifact root or the direct ``activations/``
    directory. The sampled row indices are deterministic for a given seed.
    """

    activation_dir = activation_dir_from_root(path)
    x, y = load_activation_pair(activation_dir, map_location="cpu")
    validate_activation_pair(x, y, min_rows=int(num_facts))
    if int(num_facts) <= 0:
        raise ValueError(f"num_facts must be positive, got {num_facts}")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    idx = torch.randperm(int(x.shape[0]), generator=gen)[: int(num_facts)]
    x_fact = x[idx].to(device=device, dtype=dtype)
    y_fact = y[idx].to(device=device, dtype=dtype)
    return Factset(
        input_embeddings=x_fact,
        output_embeddings=y_fact,
        mapping=create_identity_mapping(int(num_facts)),
        d_model=int(x.shape[1]),
        vocab_size=int(num_facts),
    )


def save_activation_pair(
    activation_dir: str | Path,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    metadata: dict[str, Any] | None = None,
) -> ActivationBundleInfo:
    """Validate and save x/y tensors plus optional metadata."""

    root = Path(activation_dir)
    root.mkdir(parents=True, exist_ok=True)
    n_rows, d_model = validate_activation_pair(x, y)
    torch.save(x.cpu(), root / "x.pt")
    torch.save(y.cpu(), root / "y.pt")
    if metadata is not None:
        payload = dict(metadata)
        payload.setdefault("format", FORMAT_VERSION)
        payload.setdefault("num_pairs", n_rows)
        payload.setdefault("d_model", d_model)
        write_metadata(root, payload)
    return inspect_bundle(root)
