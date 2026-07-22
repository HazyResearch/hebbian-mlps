"""Demo: Hebbian MLP variants memorize a synthetic factset."""

from __future__ import annotations

import argparse
import torch

from hebbian.core.registry import Registry
from hebbian.methods.hebbian import HebbianConfig
from hebbian.mlp_core.task import SharedConstructionConfig
from .demo_utils import (
    compute_accuracy,
    count_parameters,
    make_factset,
    parse_dtype,
    print_config,
    print_header,
    print_metrics,
    print_results,
    resolve_device,
)


def run(
    *,
    d_model: int,
    facts_multiplier: float,
    variant: str,
    m: int | None,
    device: str,
    dtype: torch.dtype,
    seed: int,
    verbose: bool,
):
    device = resolve_device(device)
    torch.manual_seed(seed)

    print_header(f"Hebbian MLP Demo ({variant})")

    factset = make_factset(
        d_model=d_model,
        facts_multiplier=facts_multiplier,
        device=device,
        dtype=dtype,
    )

    shared = SharedConstructionConfig(
        device=device,
        build_dtype=dtype,
        verbose=verbose,
    )
    hebbian_config = HebbianConfig(
        variant=variant,
        m=m,
        shared=shared,
    )

    print_config(
        [
            ("d_model", d_model),
            ("vocab_size", factset.vocab_size),
            ("variant", variant),
            ("m (hidden)", m),
            ("device", device),
            ("dtype", dtype),
        ]
    )

    method_class = Registry.get_method("hebbian")
    method = method_class()
    method.initialize(hebbian_config, seed=seed)

    print("\nConstructing...")
    mlp, metrics = method.fit_or_construct(factset)

    accuracy = compute_accuracy(mlp, factset)
    param_count = int(metrics.get("param_count", count_parameters(mlp)))

    print_metrics(metrics)
    print_results(accuracy, param_count)

    return {"accuracy": accuracy, "param_count": param_count, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hebbian MLP demo")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--facts-multiplier", type=float, default=0.25)
    parser.add_argument(
        "--variant",
        type=str,
        default="unwhitened",
        choices=["unwhitened", "whitened", "data_dependent"],
    )
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")

    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)

    run(
        d_model=args.d_model,
        facts_multiplier=args.facts_multiplier,
        variant=args.variant,
        m=args.m,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
