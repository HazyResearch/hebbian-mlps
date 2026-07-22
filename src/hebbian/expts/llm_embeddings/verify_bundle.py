"""Validate LLM activation bundles and run small MLP/transformer checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from hebbian.expts.llm_embeddings.bundle import (
    activation_dir_from_root,
    inspect_bundle,
    make_factset_from_activation_rows,
)


def run_mlp_check(
    activation_dir: str | Path,
    *,
    num_facts: int = 4,
    hidden_dim: int = 16,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Construct a tiny whitened Hebbian MLP from bundle rows."""

    from hebbian.core.registry import Registry
    from hebbian.methods.hebbian import HebbianConfig  # noqa: F401
    from hebbian.mlp_core.task import SharedConstructionConfig
    import hebbian.methods  # noqa: F401

    factset = make_factset_from_activation_rows(
        activation_dir,
        num_facts=int(num_facts),
        seed=int(seed),
        dtype=torch.float64,
        device=device,
    )
    config = HebbianConfig(
        variant="whitened",
        m=int(hidden_dim),
        ridge=1e-6,
    )
    config.shared = SharedConstructionConfig(
        device=device,
        build_dtype=torch.float64,
        verbose=False,
    )
    method_cls = Registry.get_method("hebbian")
    method = method_cls()
    method.initialize(config=config, seed=int(seed))
    mlp, metrics = method.fit_or_construct(factset)
    with torch.no_grad():
        inputs = factset.input_embeddings.to(device=device, dtype=torch.float64)
        values = factset.output_embeddings.to(device=device, dtype=torch.float64)
        outputs = mlp(inputs)
        scores = outputs @ values.T
        predictions = scores.argmax(dim=-1).cpu()
        targets = torch.arange(int(num_facts))
        accuracy = (predictions == targets).float().mean().item()
    if not torch.isfinite(outputs).all():
        raise ValueError("MLP check produced non-finite outputs")
    return {
        "num_facts": int(num_facts),
        "hidden_dim": int(hidden_dim),
        "device": device,
        "accuracy": float(accuracy),
        "metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in metrics.items()
        },
    }


def run_transformer_check(
    activation_dir: str | Path,
    *,
    output_root: str | Path,
    num_facts: int = 4,
    hidden_dim_max: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a reduced transformer hidden-dim check using this bundle."""

    from hebbian.expts.transformer.config import resolve_hidden_dim_config
    from hebbian.expts.transformer.run_hidden_dim import run as run_hidden_dim

    info = inspect_bundle(activation_dir, min_rows=int(num_facts) + 2)
    junk_vocab_size = 1
    if int(num_facts) + junk_vocab_size + 1 > info.num_pairs:
        raise ValueError(
            "transformer check needs num_facts + junk_vocab_size + Q rows; "
            f"got num_facts={num_facts}, rows={info.num_pairs}"
        )
    config = resolve_hidden_dim_config(
        preset="integration_hidden_dim",
        output_root=str(output_root),
        d_models=[info.d_model],
        num_facts_values=[int(num_facts)],
        mlp_methods=["hebbian_whitened"],
        junk_vocab_size=junk_vocab_size,
        junk_len=1,
        device=device,
        max_gpus=0,
        use_local_runner=True,
        hidden_dim_search_min=1,
        hidden_dim_search_max=int(hidden_dim_max),
        binary_search_precision=1,
        embeddings_dir=str(info.activation_dir),
        best_acc_success_threshold=0.0,
    )
    root = run_hidden_dim(config)
    return {
        "output_root": str(root),
        "capacity_csv": str(Path(root) / "capacity_points.csv"),
        "plot_dir": str(Path(root) / "plots"),
    }


def verify_bundle(args: argparse.Namespace) -> dict[str, Any]:
    activation_dir = (
        Path(args.embeddings_dir).expanduser()
        if args.embeddings_dir is not None
        else activation_dir_from_root(args.artifact_root)
    )
    info = inspect_bundle(
        activation_dir,
        min_rows=args.min_rows,
        expected_d_model=args.expected_d_model,
    )
    metadata_layer = info.metadata.get("layer_index")
    if args.expected_layer_index is not None and metadata_layer is not None:
        if int(metadata_layer) != int(args.expected_layer_index):
            raise ValueError(
                f"metadata layer_index={metadata_layer}, expected {args.expected_layer_index}"
            )
    summary: dict[str, Any] = {
        "activation_dir": str(info.activation_dir),
        "num_pairs": info.num_pairs,
        "d_model": info.d_model,
        "x_dtype": info.x_dtype,
        "y_dtype": info.y_dtype,
        "metadata": info.metadata,
    }
    if args.run_mlp_check:
        summary["mlp_check"] = run_mlp_check(
            info.activation_dir,
            num_facts=args.mlp_num_facts,
            hidden_dim=args.mlp_hidden_dim,
            seed=args.seed,
            device=args.device,
        )
    if args.run_transformer_check:
        transformer_output_root = args.transformer_output_root
        if transformer_output_root is None:
            transformer_output_root = str(info.activation_dir.parent / "verify_transformer_check")
        summary["transformer_check"] = run_transformer_check(
            info.activation_dir,
            output_root=transformer_output_root,
            num_facts=args.transformer_num_facts,
            hidden_dim_max=args.transformer_hidden_dim_max,
            device=args.device,
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Qwen3 activation artifacts.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact-root", default=None)
    source.add_argument("--embeddings-dir", default=None)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--expected-d-model", type=int, default=None)
    parser.add_argument("--expected-layer-index", type=int, default=None)
    parser.add_argument("--run-mlp-check", action="store_true")
    parser.add_argument("--mlp-num-facts", type=int, default=4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=16)
    parser.add_argument("--run-transformer-check", action="store_true")
    parser.add_argument("--transformer-output-root", default=None)
    parser.add_argument("--transformer-num-facts", type=int, default=4)
    parser.add_argument("--transformer-hidden-dim-max", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(verify_bundle(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
