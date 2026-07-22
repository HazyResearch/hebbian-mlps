"""Run transformer-capacity sweeps by searching over ``num_facts``."""

from __future__ import annotations

import argparse
from pathlib import Path

from hebbian.expts.transformer.sweep import (
    build_num_facts_sweep_configs,
    make_output_root,
    run_binary_searches_portable,
    write_resolved_config,
)
from hebbian.expts.transformer.config import (
    NumFactsConfig,
    parse_bool,
    parse_csv_list,
    parse_model_configs,
    parse_optional_float,
    resolve_num_facts_config,
)
from hebbian.expts.transformer.plot import plot_capacity_csv
from hebbian.expts.transformer.summarize import summarize_results_dir


def run(config: NumFactsConfig) -> Path:
    output_root = make_output_root(
        output_base_dir=config.output_base_dir,
        orientation=config.orientation,
        schedule=config.schedule,
        preset=config.preset,
        timestamp=config.timestamp,
        output_root=config.output_root,
    )
    config.output_root = str(output_root)
    write_resolved_config(output_root, config)
    sweep_configs = build_num_facts_sweep_configs(config)
    run_binary_searches_portable(
        sweep_configs,
        max_gpus=config.max_gpus,
        simultaneous_jobs_per_gpu=config.simultaneous_jobs_per_gpu,
        use_local_runner=config.use_local_runner,
    )
    csv_path = summarize_results_dir(output_root)
    plot_capacity_csv(csv_path, output_root / "plots")
    return output_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run transformer capacity num-facts sweeps.")
    parser.add_argument("--preset", default="full_num_facts_attn_pretrain")
    parser.add_argument("--schedule", default=None)
    parser.add_argument("--model-configs", default=None, help="Comma-separated d:f pairs.")
    parser.add_argument("--hidden-dim-multipliers", default=None)
    parser.add_argument("--hidden-dims", default=None)
    parser.add_argument("--mlp-methods", default=None)
    parser.add_argument("--junk-len", type=int, default=None)
    parser.add_argument("--junk-vocab-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--steps-per-dataset", type=int, default=None)
    parser.add_argument("--disable-early-stopping", default=None)
    parser.add_argument("--attn-residual", default=None)
    parser.add_argument("--freeze-value-dense-identity", default=None)
    parser.add_argument("--use-eval-mlp-for-eval", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--num-facts-low-multiplier", type=float, default=None)
    parser.add_argument("--num-facts-high-multiplier", type=float, default=None)
    parser.add_argument("--min-num-facts", type=int, default=None)
    parser.add_argument("--max-num-facts", type=int, default=None)
    parser.add_argument("--binary-search-precision", type=int, default=None)
    parser.add_argument("--success-metric", default=None)
    parser.add_argument("--best-acc-success-threshold", default=None)
    parser.add_argument("--gamma-success-threshold", default=None)
    parser.add_argument("--seed-success-aggregation", default=None)
    parser.add_argument("--max-gpus", type=int, default=None)
    parser.add_argument("--simultaneous-jobs-per-gpu", type=int, default=None)
    parser.add_argument("--output-base-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--use-local-runner", default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> NumFactsConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return resolve_num_facts_config(
        preset=args.preset,
        schedule=args.schedule,
        model_configs=parse_model_configs(args.model_configs),
        hidden_dim_multipliers=parse_csv_list(args.hidden_dim_multipliers, float),
        hidden_dims_override=parse_csv_list(args.hidden_dims, int),
        mlp_methods=parse_csv_list(args.mlp_methods, str),
        junk_len=args.junk_len,
        junk_vocab_size=args.junk_vocab_size,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        steps_per_dataset=args.steps_per_dataset,
        disable_early_stopping=parse_bool(args.disable_early_stopping),
        attn_residual=parse_bool(args.attn_residual),
        freeze_value_dense_identity=parse_bool(args.freeze_value_dense_identity),
        use_eval_mlp_for_eval=parse_bool(args.use_eval_mlp_for_eval),
        device=args.device,
        dtype=args.dtype,
        n_seeds=args.n_seeds,
        seeds_override=parse_csv_list(args.seeds, int),
        num_facts_low_multiplier=args.num_facts_low_multiplier,
        num_facts_high_multiplier=args.num_facts_high_multiplier,
        min_num_facts=args.min_num_facts,
        max_num_facts=args.max_num_facts,
        binary_search_precision=args.binary_search_precision,
        success_metric=args.success_metric,
        best_acc_success_threshold=parse_optional_float(args.best_acc_success_threshold),
        gamma_success_threshold=parse_optional_float(args.gamma_success_threshold),
        seed_success_aggregation=args.seed_success_aggregation,
        max_gpus=args.max_gpus,
        simultaneous_jobs_per_gpu=args.simultaneous_jobs_per_gpu,
        output_base_dir=args.output_base_dir,
        output_root=args.output_root,
        timestamp=args.timestamp,
        use_local_runner=parse_bool(args.use_local_runner),
    )


def main(argv: list[str] | None = None) -> None:
    output_root = run(parse_args(argv))
    print(output_root)


if __name__ == "__main__":
    main()
