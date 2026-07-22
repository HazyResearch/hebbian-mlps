"""Demo: NTK-constructed MLP memorizes a synthetic factset."""

from __future__ import annotations

import argparse
import torch

from hebbian.mlp_core.constructions.ntk import NTKConstructionConfig, get_ntk_mlp
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
    m: int,
    hermite_degree: int,
    activation: str,
    device: str,
    dtype: torch.dtype,
    seed: int,
    verbose: bool,
):
    device = resolve_device(device)
    torch.manual_seed(seed)

    print_header("NTK MLP Demo")

    config = NTKConstructionConfig()
    config.m = m
    config.hermite_degree = hermite_degree
    config.shared.device = device
    config.shared.build_dtype = dtype
    config.shared.verbose = verbose
    config.shared.mlp_config.activation.activation = activation

    factset = make_factset(
        d_model=d_model,
        facts_multiplier=facts_multiplier,
        device=device,
        dtype=dtype,
    )

    print_config(
        [
            ("d_model", d_model),
            ("vocab_size", factset.vocab_size),
            ("m (hidden)", m),
            ("hermite_degree", hermite_degree),
            ("activation", activation),
            ("device", device),
            ("dtype", dtype),
        ]
    )

    print("\nConstructing...")
    mlp, metrics = get_ntk_mlp(factset, config)

    accuracy = compute_accuracy(mlp, factset)
    param_count = count_parameters(mlp)

    print_metrics(metrics)
    print_results(accuracy, param_count)

    return {"accuracy": accuracy, "param_count": param_count, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="NTK MLP demo")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--facts-multiplier", type=float, default=0.25)
    parser.add_argument("--m", type=int, default=1000)
    parser.add_argument("--hermite-degree", type=int, default=1)
    parser.add_argument("--activation", type=str, default="swish")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")

    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)

    run(
        d_model=args.d_model,
        facts_multiplier=args.facts_multiplier,
        m=args.m,
        hermite_degree=args.hermite_degree,
        activation=args.activation,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
