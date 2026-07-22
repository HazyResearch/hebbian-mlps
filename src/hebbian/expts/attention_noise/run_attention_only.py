"""Section 3.1 attention-only sweep: attention noise vs junk length.

This variant removes learned MLP behavior by using identity MLP and trains only
attention on the associative-recall key-copy task:
[junk] K [junk] Q -> predict K

Sweep:
- model scale (d_model, num_facts)
- junk length
- random seeds

Metric:
- attention noise statistics computed from final ln_f output at Q
  against corresponding key embedding.
"""

from __future__ import annotations

import asyncio
import copy
import math
import os
import traceback
from dataclasses import field
from typing import Any, List

import torch
from hebbian.config import main as main_decorator
from hebbian.config import pydraclass

from hebbian.gpu_sweep import GridSearchConfig
from hebbian.gpu_sweep import GPUJobResult, GPUScheduler
from hebbian.gpu_sweep import run_grid_search

from hebbian.transformer.train_attention import train_attention
from hebbian.transformer.train import train_associative_recall


from hebbian.expts.attention_noise.helpers import (  # noqa: E402
    aggregate_seed_metrics,
    build_attention_only_config,
    build_associative_recall_config,
    compute_attention_noise_metrics,
    default_base_dir,
)


@pydraclass
class ExperimentConfig:
    """Top-level sweep config for Section 3.1 attention-only variant."""

    model_configs: list[tuple[int, int]] = field(
        default_factory=lambda: [(64, 512), (90, 1012), (128, 2048)]
    )
    junk_lens: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64, 128])

    base_dir: str | None = None

    # Training
    num_epochs: int = 4000
    batch_size: int = 1280
    learning_rate: float = 2e-4
    steps_per_dataset: int = 1
    disable_early_stopping: bool = False
    junk_vocab_size: int = 16
    couple_jl_jv: bool = False  # if True, junk_vocab_size=junk_len for each sweep point
    attn_residual: bool = False
    freeze_voproj: bool = False

    # "ce_pretrain": train attention on identity associative recall (no MLP)
    # "gd_then_attn": train GD MLP first, insert frozen, then train attention
    setting: str = "ce_pretrain"
    mlp_method: str = "gd"  # only used when setting="gd_then_attn"

    # Sweep seeds
    n_seeds: int = 4

    # Attention-noise evaluation
    noise_eval_batches: int = 64
    noise_eval_batch_size: int | None = 512

    # Scheduler
    max_gpus: int = 4
    simultaneous_jobs_per_gpu: int = 2


@pydraclass
class Section31AttentionOnlyGridSearchConfig(GridSearchConfig):
    """Grid config for one (d_model, num_facts, junk_len) cell over seeds."""

    d_model: int = 64
    num_facts: int = 512
    junk_len: int = 16
    noise_eval_batches: int = 64
    noise_eval_batch_size: int | None = 512
    setting: str = "ce_pretrain"
    mlp_method: str = "gd"

    def get_experiment_config_and_base_dir(self, **kwargs):
        config = copy.deepcopy(self.base_experiment_config)

        seed = int(kwargs.get("seed", 42))
        config.train_config.seed = seed

        base_dir = f"{self.base_dir}/seed_{seed}"
        config.train_config.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        return config, base_dir

    def run_experiment_config(self, config):
        if self.setting == "gd_then_attn":
            full_result = train_associative_recall(config)
        else:
            full_result = train_attention(config)

        gpt_model = full_result["gpt_model"]
        factset = full_result["factset"]

        noise_batch_size = (
            int(self.noise_eval_batch_size)
            if self.noise_eval_batch_size is not None
            else int(config.train_config.batch_size)
        )

        noise_metrics = compute_attention_noise_metrics(
            gpt_model=gpt_model,
            mapping=factset.mapping,
            dataset_config=config.dataset_config,
            train_seed=config.train_config.seed,
            measurement_stage="lnf",
            num_batches=int(self.noise_eval_batches),
            batch_size=noise_batch_size,
        )

        final_acc = float(full_result.get("final_accuracy", float("nan")))
        slim_result = {
            "best_acc": float(full_result.get("best_acc", float("nan"))),
            "final_accuracy": final_acc,
            # Keep keys consistent with existing 3.1 downstream tools.
            "final_eval_accuracy": final_acc,
            "final_train_accuracy": final_acc,
            **noise_metrics,
        }

        gpt_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return slim_result

    def agg_results(self, results: List[GPUJobResult]) -> Any:
        best_result, summary, seed_values, seed_list_values = aggregate_seed_metrics(results)
        if best_result is None:
            return {"error": "No successful results", "all_results": results}

        payload = dict(best_result.result)
        payload["seed_summary"] = summary
        payload["seed_values"] = seed_values
        payload["seed_list_values"] = seed_list_values

        per_key_lists = seed_list_values.get("attn_noise_l2_per_key_max_values", [])
        if isinstance(per_key_lists, list):
            combined_per_key_values = [
                float(v)
                for seed_vals in per_key_lists
                if isinstance(seed_vals, list)
                for v in seed_vals
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            ]
        else:
            combined_per_key_values = []
        payload["attn_noise_l2_per_key_max_values_all_seeds"] = combined_per_key_values

        best_result.result = payload
        return best_result


def get_sweep_configs(config: ExperimentConfig) -> List[Section31AttentionOnlyGridSearchConfig]:
    """Create one grid config per (d_model, num_facts, junk_len)."""
    base_dir_root = config.base_dir if config.base_dir is not None else default_base_dir(config.setting)
    print(f"Base directory: {base_dir_root}")

    seeds = [42 + i * 10000 for i in range(config.n_seeds)]
    configs: List[Section31AttentionOnlyGridSearchConfig] = []

    for d_model, num_facts in config.model_configs:
        for junk_len in config.junk_lens:
            junk_vocab_size = junk_len if config.couple_jl_jv else config.junk_vocab_size
            if config.setting == "gd_then_attn":
                base_experiment_config = build_associative_recall_config(
                    d_model=d_model,
                    num_facts=num_facts,
                    junk_len=junk_len,
                    junk_vocab_size=junk_vocab_size,
                    mlp_method=config.mlp_method,
                    num_epochs=config.num_epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    steps_per_dataset=config.steps_per_dataset,
                    disable_early_stopping=config.disable_early_stopping,
                    attn_residual=config.attn_residual,
                    seed=42,
                    freeze_voproj=config.freeze_voproj,
                )
            else:
                base_experiment_config = build_attention_only_config(
                    d_model=d_model,
                    num_facts=num_facts,
                    junk_len=junk_len,
                    junk_vocab_size=junk_vocab_size,
                    num_epochs=config.num_epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    steps_per_dataset=config.steps_per_dataset,
                    disable_early_stopping=config.disable_early_stopping,
                    attn_residual=config.attn_residual,
                    seed=42,
                    freeze_voproj=config.freeze_voproj,
                )

            run_dir = f"{base_dir_root}/d{d_model}_n{num_facts}/junk_len_{junk_len}"
            os.makedirs(run_dir, exist_ok=True)

            configs.append(
                Section31AttentionOnlyGridSearchConfig(
                    base_dir=run_dir,
                    sweep_props={"seed": seeds},
                    base_experiment_config=base_experiment_config,
                    d_model=d_model,
                    num_facts=num_facts,
                    junk_len=junk_len,
                    noise_eval_batches=config.noise_eval_batches,
                    noise_eval_batch_size=config.noise_eval_batch_size,
                    setting=config.setting,
                    mlp_method=config.mlp_method,
                )
            )

    return configs


async def _run_grid_searches_resilient_async(
    configs: list[Section31AttentionOnlyGridSearchConfig],
    gpu_scheduler: GPUScheduler,
) -> list[Any]:
    tasks = [run_grid_search(cfg, gpu_scheduler) for cfg in configs]
    return await asyncio.gather(*tasks, return_exceptions=True)


def run_grid_searches_resilient(
    configs: list[Section31AttentionOnlyGridSearchConfig],
    max_gpus: int,
    simultaneous_jobs_per_gpu: int,
) -> list[Any]:
    """Run all grid configs and surface per-config exceptions without aborting."""
    gpu_scheduler = GPUScheduler(
        max_gpus=max_gpus,
        simultaneous_jobs_per_gpu=simultaneous_jobs_per_gpu,
    )
    try:
        return asyncio.run(_run_grid_searches_resilient_async(configs, gpu_scheduler))
    finally:
        gpu_scheduler.shutdown()


@main_decorator(ExperimentConfig)
def main(config: ExperimentConfig):
    configs = get_sweep_configs(config)
    print(f"\nRunning {len(configs)} Section 3.1 attention-only sweep configs:")
    for cfg in configs:
        print(f"  {cfg.base_dir}")
        print(f"    d_model={cfg.d_model}, num_facts={cfg.num_facts}, junk_len={cfg.junk_len}")
        print(f"    seeds={cfg.sweep_props['seed']}")
        print(
            "    noise_eval="
            f"{cfg.noise_eval_batches} batches"
            f", batch_size={cfg.noise_eval_batch_size}"
        )

    results = run_grid_searches_resilient(
        configs=configs,
        max_gpus=config.max_gpus,
        simultaneous_jobs_per_gpu=config.simultaneous_jobs_per_gpu,
    )

    failures: list[tuple[Section31AttentionOnlyGridSearchConfig, BaseException]] = []
    for cfg, result in zip(configs, results):
        if isinstance(result, BaseException):
            failures.append((cfg, result))
            print(f"\n[ERROR] Grid config failed: {cfg.base_dir}", flush=True)
            traceback.print_exception(type(result), result, result.__traceback__)

    if failures:
        print(
            f"\nSection 3.1 attention-only sweep finished with {len(failures)} failed grid configs "
            f"(out of {len(configs)}).",
            flush=True,
        )
        raise RuntimeError(
            "Section 3.1 attention-only sweep had grid-level failures. "
            "See traceback(s) above for root cause."
        )


if __name__ == "__main__":
    main()
