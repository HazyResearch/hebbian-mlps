"""Regenerate Qwen3 standalone MLP-capacity pickles from activation rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hebbian.expts.llm_embeddings.bundle import activation_dir_from_root, inspect_bundle
from hebbian.expts.mlp_capacity.run import ExperimentConfig, run as run_capacity_sweep


def _parse_int_list(value: str) -> tuple[int, ...]:
    vals = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(vals) == 0:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    if any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError(f"all values must be positive: {vals}")
    return vals


def _parse_str_list(value: str) -> tuple[str, ...]:
    vals = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(vals) == 0:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return vals


def _parse_float_pair(value: str) -> tuple[float, float]:
    vals = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated floats")
    if vals[0] <= 0 or vals[1] <= vals[0]:
        raise argparse.ArgumentTypeError(f"invalid range: {vals}")
    return vals


def _resolve_embeddings_dir(args: argparse.Namespace) -> Path:
    if args.embeddings_dir is not None:
        return activation_dir_from_root(args.embeddings_dir)
    if args.artifact_root is None:
        raise ValueError("provide --embeddings-dir or --artifact-root")
    return activation_dir_from_root(args.artifact_root)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser()
    if args.artifact_root is not None:
        return Path(args.artifact_root).expanduser() / "results" / "mlp_capacity"
    return Path("results") / "qwen3_mlp_capacity"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact-root", default=None)
    source.add_argument("--embeddings-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-facts", type=_parse_int_list, default=(128, 256, 512))
    parser.add_argument("--methods", type=_parse_str_list, default=("hebbian_whitened",))
    parser.add_argument("--binary-search-range", type=_parse_float_pair, default=(1.0, 65536.0))
    parser.add_argument("--binary-search-precision", type=float, default=0.05)
    parser.add_argument("--success-acc-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--simultaneous-jobs-per-gpu", type=int, default=1)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    embeddings_dir = _resolve_embeddings_dir(args)
    num_facts: Sequence[int] = tuple(args.num_facts)
    info = inspect_bundle(embeddings_dir, min_rows=max(num_facts))
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig()
    config.base_dir = str(output_dir)
    config.embeddings_dir = str(info.activation_dir)
    config.use_embedding_d_model = True
    config.d_models = (int(info.d_model),)
    config.num_facts_values = tuple(int(v) for v in num_facts)
    config.methods = tuple(args.methods)
    config.binary_search_range = tuple(float(v) for v in args.binary_search_range)
    config.binary_search_precision = float(args.binary_search_precision)
    config.success_acc_threshold = float(args.success_acc_threshold)
    config.seed = int(args.seed)
    config.device = args.device
    config.max_gpus = int(args.max_gpus)
    config.simultaneous_jobs_per_gpu = int(args.simultaneous_jobs_per_gpu)
    config.mapping_type = "identity"
    config.tie_embeddings = False

    run_capacity_sweep(config)
    return {
        "activation_dir": str(info.activation_dir),
        "num_pairs": info.num_pairs,
        "d_model": info.d_model,
        "num_facts_values": list(config.num_facts_values),
        "methods": list(config.methods),
        "output_dir": str(output_dir),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
