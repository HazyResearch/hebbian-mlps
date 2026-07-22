"""Run the h=512 fact-editing experiment reported in the paper."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NUM_FACTS = 16_384
EDIT_PERCENTAGES = (2, 5, 10)
METHODS = ("memit", "alpha_edit", "rome", "gd_construction")
MLP_VARIANT = "gd_mse_shuffle"
MLP_METHOD_KWARGS = {
    "loss_fn": "MSE",
    "shuffle": True,
    "activation": "swish",
    "build_dtype": "float32",
    "final_dtype": "float32",
}


@dataclass(frozen=True)
class EditJob:
    method: str
    preserve_count: int
    alter_count: int
    label: str
    overrides: tuple[str, ...] = ()


def _override(name: str, value: Any) -> str:
    if isinstance(value, Path):
        value = str(value)
    return f"{name}={value!r}" if isinstance(value, str) else f"{name}={value}"


def _optional_override(name: str, value: Any) -> tuple[str, ...]:
    return () if value is None else (_override(name, value),)


def _method_settings(method: str):
    if method == "memit":
        for steps, lr, lambd, clip_norm in itertools.product(
            (10, 25, 100),
            (0.005, 0.05, 0.5),
            (15000, 1500, 150, 1),
            (0.5, 0.75, 1.0),
        ):
            label = f"memit[s={steps},lr={lr},lambda={lambd},clip={clip_norm}]"
            yield label, (
                _override("num_steps", steps),
                _override("lr", lr),
                _override("lambd", lambd),
                _override("clip_norm", clip_norm),
            )
    elif method == "alpha_edit":
        for steps, lr, clip_norm, tolerance in itertools.product(
            (10, 25, 100),
            (0.005, 0.05, 0.5),
            (0.75, 0.25, None),
            (0.01, 1, 10),
        ):
            label = f"alpha_edit[s={steps},lr={lr},clip={clip_norm},tol={tolerance}]"
            overrides = (
                _override("num_steps", steps),
                _override("lr", lr),
                _override("tol", tolerance),
                *_optional_override("clip_norm", clip_norm),
            )
            yield label, overrides
    elif method == "rome":
        for steps, lr, weight_decay, early_stopping in itertools.product(
            (10, 25, 100),
            (0.005, 0.05, 0.5),
            (0.0015, 0.00015, 0.0),
            (0.05, None),
        ):
            label = (
                f"rome[s={steps},lr={lr},wd={weight_decay},es={early_stopping}]"
            )
            overrides = (
                _override("num_steps", steps),
                _override("lr", lr),
                _override("wd", weight_decay),
                *_optional_override("early_stopping", early_stopping),
            )
            yield label, overrides
    elif method == "gd_construction":
        yield "gd_construction[gd_mse_shuffle]", (
            _override("gd_replacement_variant_label", MLP_VARIANT),
            _override("gd_replacement_method", "gd"),
            _override("gd_replacement_hidden_dim", 512),
            _override("gd_replacement_method_kwargs", MLP_METHOD_KWARGS),
        )
    else:
        raise ValueError(f"Unknown edit method: {method}")


def build_edit_jobs() -> list[EditJob]:
    """Expand the exact edit-fraction and hyperparameter grid from the paper."""

    jobs: list[EditJob] = []
    for percentage in EDIT_PERCENTAGES:
        alter_count = max(1, min(NUM_FACTS, round(NUM_FACTS * percentage / 100)))
        preserve_count = NUM_FACTS - alter_count
        for method in METHODS:
            for method_label, overrides in _method_settings(method):
                jobs.append(
                    EditJob(
                        method=method,
                        preserve_count=preserve_count,
                        alter_count=alter_count,
                        label=f"edit_{percentage}pct:{method_label}",
                        overrides=overrides,
                    )
                )
    return jobs


def _run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    print_only: bool = False,
) -> None:
    print(shlex.join(command), flush=True)
    if not print_only:
        subprocess.run(command, check=True, env=environment)


def _base_train_command(
    python: str,
    base_model_dir: Path,
    authors_csv: Path,
) -> list[str]:
    overrides = {
        "experiment_dir": base_model_dir,
        "mlp_variant_label": MLP_VARIANT,
        "authors_csv_path": authors_csv,
        "tokenizer_name": "EleutherAI/pythia-70m",
        "num_facts": NUM_FACTS,
        "num_rephrases": 16,
        "fact_input_embedding_mode": "normalized_token",
        "overwrite_compound_token_embeddings": False,
        "eval_batch_size": 512,
        "train_num_workers": 4,
        "eval_num_workers": 4,
        "train_config.device": "cuda",
        "train_config.mlp_method": "gd",
        "train_config.mlp_hidden_dim": 512,
        "train_config.mlp_method_kwargs": MLP_METHOD_KWARGS,
        "train_config.embeddings_config.d_model": 256,
        "train_config.embeddings_config.embedding_init": "kaiming_uniform",
        "train_config.embeddings_config.tie_embeddings": True,
        "train_config.transformer_config.n_layers": 1,
        "train_config.transformer_config.n_head": 1,
        "train_config.transformer_config.use_moe": True,
        "train_config.transformer_config.moe_router_num_layers": 2,
        "train_config.transformer_config.moe_router_intermediate_dim": None,
        "train_config.transformer_config.moe_gate": False,
        "train_config.transformer_config.moe_router_use_mlp_input": False,
        "train_config.transformer_config.moe_convex": True,
        "train_config.transformer_config.moe_mlp_type": "lora_linear",
        "train_config.transformer_config.moe_mlp_out_norm": False,
        "train_config.transformer_config.moe_lora_linear_rank": 8,
        "train_config.transformer_config.use_mlp_qk": True,
        "train_config.transformer_config.bias": True,
        "train_config.epochs": 18,
        "train_config.batch_size": 32,
        "train_config.lr": 2e-4,
        "train_config.evaluate_every": 1,
        "train_config.early_stop_accuracy": 0.99,
        "train_config.seed": 10042,
    }
    return [
        python,
        "-m",
        "hebbian.expts.fact_editing.train_base",
        *(_override(name, value) for name, value in overrides.items()),
    ]


def train_base_model(
    *,
    python: str,
    base_model_dir: Path,
    authors_csv: Path,
    gpu_ids: list[str],
    force: bool,
    print_only: bool,
) -> None:
    checkpoint_dir = base_model_dir / "checkpoints"
    if not force and all(
        (checkpoint_dir / name).is_file()
        for name in ("last_model.pt", "embeddings.pt")
    ):
        print(f"Reusing base model in {base_model_dir}")
        return

    command = _base_train_command(python, base_model_dir, authors_csv)
    if len(gpu_ids) > 1:
        command = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(gpu_ids)}",
            "-m",
            "hebbian.expts.fact_editing.train_base",
            *command[3:],
        ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    _run(command, environment=environment, print_only=print_only)


def _run_edit_job(
    job: EditJob,
    *,
    python: str,
    base_model_dir: Path,
    results_dir: Path,
    gpu_id: str,
    print_only: bool,
) -> None:
    command = [
        python,
        "-m",
        "hebbian.expts.fact_editing.run_edit",
        _override("experiment_dir", base_model_dir),
        _override("type", job.method),
        _override("num_preserve_facts", job.preserve_count),
        _override("num_alter_facts", job.alter_count),
        _override("device", "cuda:0"),
        _override("out_dir", results_dir),
        _override("seed", 42),
        _override("compute_non_fact_ppl", True),
        *job.overrides,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    print(f"[{gpu_id}] {job.label}", flush=True)
    _run(command, environment=environment, print_only=print_only)


def run_edit_grid(
    *,
    python: str,
    base_model_dir: Path,
    results_dir: Path,
    gpu_ids: list[str],
    print_only: bool,
) -> None:
    jobs = build_edit_jobs()
    print(f"Running {len(jobs)} edit configurations across {len(gpu_ids)} GPUs")
    if print_only:
        for index, job in enumerate(jobs):
            _run_edit_job(
                job,
                python=python,
                base_model_dir=base_model_dir,
                results_dir=results_dir,
                gpu_id=gpu_ids[index % len(gpu_ids)],
                print_only=True,
            )
        return

    def worker(gpu_id: str, assigned: list[EditJob]) -> None:
        for job in assigned:
            _run_edit_job(
                job,
                python=python,
                base_model_dir=base_model_dir,
                results_dir=results_dir,
                gpu_id=gpu_id,
                print_only=False,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [
            pool.submit(worker, gpu_id, jobs[index:: len(gpu_ids)])
            for index, gpu_id in enumerate(gpu_ids)
        ]
        for future in futures:
            future.result()


def summarize(
    *, python: str, results_dir: Path, print_only: bool
) -> Path:
    output_csv = results_dir / "best_results.csv"
    _run(
        [
            python,
            "-m",
            "hebbian.expts.fact_editing.summarize_results",
            _override("directory", results_dir),
            _override("output_csv", output_csv),
        ],
        print_only=print_only,
    )
    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="artifacts/fact_editing/h512")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--authors-csv",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "language"
            / "book_authors.csv"
        ),
    )
    parser.add_argument("--force-base", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-edits", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = Path(args.base_dir).resolve()
    base_model_dir = base_dir / "base_model"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")

    if not args.skip_pretrain:
        train_base_model(
            python=args.python,
            base_model_dir=base_model_dir,
            authors_csv=Path(args.authors_csv).resolve(),
            gpu_ids=gpu_ids,
            force=args.force_base,
            print_only=args.print_only,
        )
    if not args.skip_edits:
        run_edit_grid(
            python=args.python,
            base_model_dir=base_model_dir,
            results_dir=results_dir,
            gpu_ids=gpu_ids,
            print_only=args.print_only,
        )
    if not args.skip_summary:
        output_csv = summarize(
            python=args.python,
            results_dir=results_dir,
            print_only=args.print_only,
        )
        print(f"Summary: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
