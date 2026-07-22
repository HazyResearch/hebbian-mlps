"""Regenerate Qwen3 layer-14 margin JSONs from activation rows.

The AK/AV result samples paired rows from ``activations/{x.pt,y.pt}``. The
RK/RV comparison uses a matched isotropic synthetic factset with the same
``d_model`` and sweep grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hebbian.expts.llm_embeddings.bundle import activation_dir_from_root, inspect_bundle
from hebbian.expts.margins.run import MarginSweepRunnerConfig, run as run_margin_sweep


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
        return Path(args.artifact_root).expanduser() / "results" / "margins"
    return Path("results") / "qwen3_margins"


def _make_config(
    *,
    d_model: int,
    output_json: Path,
    base_dir: Path,
    args: argparse.Namespace,
    embeddings_dir: Path | None,
) -> MarginSweepRunnerConfig:
    config = MarginSweepRunnerConfig()
    config.sweep = str(args.sweep)
    config.d = int(d_model)
    if config.sweep == "F":
        config.M = int(args.m)
        config.F_min = int(args.f_min)
        config.F_max = int(args.f_max)
        config.n_F_points = int(args.n_f_points)
    elif config.sweep == "M":
        config.F = int(args.num_facts)
        config.M_min = int(args.m_min)
        config.M_max = int(args.m_max)
        config.n_M_points = int(args.n_m_points)
    else:
        raise ValueError(f"Unsupported Qwen3 margin sweep: {config.sweep!r}")
    config.device = args.device
    config.build_dtype = args.build_dtype
    config.n_seeds = int(args.n_seeds)
    config.seed_offset = int(args.seed_offset)
    config.max_gpus = int(args.max_gpus)
    config.simultaneous_jobs_per_gpu = int(args.simultaneous_jobs_per_gpu)
    config.use_u_star_codes = bool(args.use_u_star_codes)
    config.admm_n_iters = int(args.admm_n_iters)
    config.admm_batch_size = int(args.admm_batch_size)
    config.gamma_min_percentile = args.gamma_min_percentile
    config.base_dir = str(base_dir)
    config.output_json = str(output_json)
    config.embeddings_dir = str(embeddings_dir) if embeddings_dir is not None else None
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact-root", default=None)
    source.add_argument("--embeddings-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--sweep", choices=["F", "M"], default="F")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--f-min", type=int, default=4)
    parser.add_argument("--f-max", type=int, default=128)
    parser.add_argument("--n-f-points", type=int, default=15)
    parser.add_argument("--num-facts", type=int, default=512)
    parser.add_argument("--m-min", type=int, default=64)
    parser.add_argument("--m-max", type=int, default=2048)
    parser.add_argument("--n-m-points", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--build-dtype", default="float64")
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--simultaneous-jobs-per-gpu", type=int, default=1)
    parser.add_argument("--admm-n-iters", type=int, default=300)
    parser.add_argument("--admm-batch-size", type=int, default=256)
    parser.add_argument("--gamma-min-percentile", type=float, default=None)
    parser.add_argument("--use-u-star-codes", dest="use_u_star_codes", action="store_true")
    parser.add_argument("--no-u-star-codes", dest="use_u_star_codes", action="store_false")
    parser.set_defaults(use_u_star_codes=True)
    parser.add_argument("--skip-rkrv", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    embeddings_dir = _resolve_embeddings_dir(args)
    min_rows = int(args.f_max) if args.sweep == "F" else int(args.num_facts)
    info = inspect_bundle(embeddings_dir, min_rows=min_rows)
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"qwen3_06b_layer14_{args.sweep}"

    outputs: dict[str, str] = {}
    akav_json = output_dir / f"{stem}_akav.json"
    akav_config = _make_config(
        d_model=info.d_model,
        output_json=akav_json,
        base_dir=output_dir / "run_akav",
        args=args,
        embeddings_dir=info.activation_dir,
    )
    run_margin_sweep(akav_config)
    outputs["akav_json"] = str(akav_json)

    if not args.skip_rkrv:
        rkrv_json = output_dir / f"{stem}_rkrv.json"
        rkrv_config = _make_config(
            d_model=info.d_model,
            output_json=rkrv_json,
            base_dir=output_dir / "run_rkrv",
            args=args,
            embeddings_dir=None,
        )
        run_margin_sweep(rkrv_config)
        outputs["rkrv_json"] = str(rkrv_json)

    return {
        "activation_dir": str(info.activation_dir),
        "num_pairs": info.num_pairs,
        "d_model": info.d_model,
        "sweep": args.sweep,
        "fixed_m": int(args.m) if args.sweep == "F" else None,
        "num_facts": int(args.num_facts) if args.sweep == "M" else None,
        "outputs": outputs,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
