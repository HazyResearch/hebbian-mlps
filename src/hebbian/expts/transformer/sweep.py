"""Sweep construction and execution for Transformer capacity experiments."""

from __future__ import annotations

import copy
import datetime as dt
import io
import itertools
import json
import math
import os
import pickle
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from hebbian.gpu_sweep import get_jobs_for_mid, run_binary_searches as run_gpu_binary_searches
from hebbian.gpu_sweep import BinarySearchConfig
from hebbian.gpu_sweep import GPUJobResult
from hebbian.config import pydraclass

from hebbian.expts.transformer.config import DEFAULT_OUTPUT_BASE_DIR
from hebbian.transformer.config import (
    DatasetConfig,
    EmbeddingsConfig,
    AssociativeRecallConfig,
    TrainingConfig,
    TransformerConfig,
    _peek_embeddings_shape,
)
from hebbian.transformer.data import create_associative_recall_batches
from hebbian.transformer.fact_store import build_fact_mlp
from hebbian.transformer.train import train_associative_recall
from hebbian.transformer.train_attention import train_attention
from hebbian.transformer.utils import evaluate, insert_mlp_into_gpt
from hebbian.data.synthetics.factsets import Factset, create_random_permutation_mapping


METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "gd": {
        "mlp_method": "gd",
        "mlp_dtype": None,
        "mlp_method_kwargs": None,
        "capacity_multiplier": 1.0,
    },
    "hebbian": {
        "mlp_method": "hebbian",
        "mlp_dtype": torch.float64,
        "mlp_method_kwargs": {"variant": "unwhitened"},
        "capacity_multiplier": 1.0,
    },
    "hebbian_whitened": {
        "mlp_method": "hebbian",
        "mlp_dtype": torch.float64,
        "mlp_method_kwargs": {"variant": "whitened", "ridge": 1e-6},
        "capacity_multiplier": 1.0,
    },
    "ntk": {
        "mlp_method": "ntk",
        "mlp_dtype": torch.float64,
        "mlp_method_kwargs": {"hermite_degree": 1},
        "capacity_multiplier": 1.0,
    },
    "cf_coord_whitened": {
        "mlp_method": "hebbian",
        "mlp_dtype": torch.float64,
        "mlp_method_kwargs": {"variant": "data_dependent", "ridge": 1e-6},
        "capacity_multiplier": 1.0,
    },
}

METHOD_ORDER = [
    "gd",
    "hebbian",
    "hebbian_whitened",
    "cf_coord_whitened",
    "ntk",
    "unknown",
]
METHOD_DISPLAY = {
    "gd": "GD",
    "hebbian": "Ours (no whitening)",
    "hebbian_whitened": "Ours (whitened)",
    "cf_coord_whitened": "Ours (data-dependent)",
    "ntk": "NTK",
    "unknown": "Unknown",
}


def timestamp_string() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_root(
    *,
    output_base_dir: str = DEFAULT_OUTPUT_BASE_DIR,
    orientation: str,
    schedule: str,
    preset: str,
    timestamp: str | None = None,
    output_root: str | None = None,
) -> Path:
    if output_root is not None:
        return Path(output_root).resolve()
    ts = timestamp or timestamp_string()
    return Path(output_base_dir).resolve() / orientation / schedule / preset / ts


def torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype string {dtype_name!r}.")
    return mapping[dtype_name]


def serialize_jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: serialize_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): serialize_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_jsonable(item) for item in value]
    return value


def write_resolved_config(output_root: Path, config: Any) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "resolved_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_jsonable(config), f, indent=2, sort_keys=True)
    return path


def theoretical_hidden_dim(d_model: int, num_facts: int) -> int:
    return int(np.ceil((4.0 * float(num_facts)) / (3.0 * float(d_model))))


def get_hidden_dims(
    *,
    d_model: int,
    reference_num_facts: int,
    hidden_dim_multipliers: Sequence[float],
    base_hidden_dim_multiplier: float = 1.0,
    hidden_dims_override: Sequence[int] | None = None,
) -> List[int]:
    if hidden_dims_override:
        return sorted({max(1, int(v)) for v in hidden_dims_override})
    m_ref = theoretical_hidden_dim(d_model, reference_num_facts)
    m_ref = max(1, int(round(base_hidden_dim_multiplier * float(m_ref))))
    dims = [max(1, int(round(m_ref * float(mult)))) for mult in hidden_dim_multipliers]
    return sorted(set(dims))


def predicted_num_facts_from_hidden_dim(d_model: int, hidden_dim: int) -> int:
    return max(1, int(round(0.75 * float(d_model) * float(hidden_dim))))


def get_num_facts_search_range(
    *,
    d_model: int,
    hidden_dim: int,
    low_multiplier: float,
    high_multiplier: float,
    min_num_facts: int,
    max_num_facts: int,
) -> tuple[int, int]:
    anchor = predicted_num_facts_from_hidden_dim(d_model, hidden_dim)
    low = int(math.floor(anchor * float(low_multiplier)))
    high = int(math.ceil(anchor * float(high_multiplier)))
    low = max(int(min_num_facts), low)
    high = min(int(max_num_facts), high)
    if high <= low:
        high = min(int(max_num_facts), low + max(1, low // 4))
    if high <= low:
        high = low + 1
    return low, high


def _apply_train_common(
    cfg: AssociativeRecallConfig,
    *,
    device: str,
    dtype_name: str,
) -> AssociativeRecallConfig:
    cfg.train_config.device = device
    cfg.train_config.dtype = torch_dtype_from_name(dtype_name)
    return cfg


def build_insert_then_train_attn_config(
    *,
    d_model: int,
    num_facts: int,
    junk_len: int,
    junk_vocab_size: int,
    mlp_method: str,
    hidden_dim: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    steps_per_dataset: int,
    disable_early_stopping: bool,
    attn_residual: bool,
    freeze_value_dense_identity: bool,
    seed: int,
    device: str,
    dtype_name: str,
    mlp_dtype: torch.dtype | None = None,
    mlp_method_kwargs: dict | None = None,
    embeddings_dir: str | None = None,
    attn_norm_type: str = "rmsnorm",
    mlp_norm_type: str = "unit_rmsnorm",
    lm_head_norm_type: str = "unit_rmsnorm",
    mlp_acc_threshold: float | None = None,
) -> AssociativeRecallConfig:
    # LLM-embedding mode requires untied tables (x.pt and y.pt are independent).
    tie_embeddings = embeddings_dir is None
    cfg = AssociativeRecallConfig(
        dataset_config=DatasetConfig(
            num_facts=int(num_facts),
            junk_vocab_size=int(junk_vocab_size),
            min_seq_length=int(junk_len),
            max_seq_length=int(junk_len),
            use_identity_fact_mapping=False,
        ),
        train_config=TrainingConfig(
            embeddings_config=EmbeddingsConfig(
                d_model=int(d_model),
                tie_embeddings=tie_embeddings,
                embeddings_dir=embeddings_dir,
            ),
            transformer_config=TransformerConfig(
                attn_residual=bool(attn_residual),
                freeze_value_dense_identity=bool(freeze_value_dense_identity),
                use_identity_mlp=True,
                no_positional_encoding=True,
                attn_norm_type=str(attn_norm_type),
                mlp_norm_type=str(mlp_norm_type),
                lm_head_norm_type=str(lm_head_norm_type),
                freeze_input_embeddings=True,
                freeze_output_embeddings=True,
            ),
            mlp_method=mlp_method,
            mlp_hidden_dim=int(hidden_dim),
            mlp_dtype=mlp_dtype,
            mlp_method_kwargs=mlp_method_kwargs,
            mlp_acc_threshold=mlp_acc_threshold,
            batch_size=int(batch_size),
            lr=float(learning_rate),
            steps_per_dataset=int(steps_per_dataset),
            epochs=int(num_epochs),
            early_stop_accuracy=None if disable_early_stopping else 0.99,
            seed=int(seed),
        ),
    )
    return _apply_train_common(cfg, device=device, dtype_name=dtype_name)


def build_attn_pretrain_config(
    *,
    d_model: int,
    num_facts: int,
    junk_len: int,
    junk_vocab_size: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    steps_per_dataset: int,
    disable_early_stopping: bool,
    attn_residual: bool,
    seed: int,
    device: str,
    dtype_name: str,
) -> AssociativeRecallConfig:
    cfg = AssociativeRecallConfig(
        dataset_config=DatasetConfig(
            num_facts=int(num_facts),
            junk_vocab_size=int(junk_vocab_size),
            min_seq_length=int(junk_len),
            max_seq_length=int(junk_len),
            use_identity_fact_mapping=True,
        ),
        train_config=TrainingConfig(
            embeddings_config=EmbeddingsConfig(d_model=int(d_model)),
            transformer_config=TransformerConfig(
                attn_residual=bool(attn_residual),
                freeze_value_dense_identity=False,
                use_identity_mlp=True,
                no_positional_encoding=True,
                lm_head_norm_type="unit_rmsnorm",
                mlp_norm_type="unit_rmsnorm",
                freeze_input_embeddings=True,
                freeze_output_embeddings=True,
            ),
            batch_size=int(batch_size),
            lr=float(learning_rate),
            steps_per_dataset=int(steps_per_dataset),
            epochs=int(num_epochs),
            early_stop_accuracy=None if disable_early_stopping else 0.99,
            seed=int(seed),
            save_at_end=False,
        ),
    )
    return _apply_train_common(cfg, device=device, dtype_name=dtype_name)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(float(value)))
    return False


def _to_float_or_nan(value: Any) -> float:
    return float(value) if _is_finite_number(value) else float("nan")


def extract_scalar_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    mlp_metrics = result.get("mlp_metrics", {}) if isinstance(result, dict) else {}
    gamma_min = result.get("mlp_gamma_min", None)
    if not _is_finite_number(gamma_min):
        gamma_min = mlp_metrics.get("gamma_min", float("nan"))
    return {
        "best_acc": _to_float_or_nan(result.get("best_acc", float("nan"))),
        "best_train_acc": _to_float_or_nan(result.get("best_train_acc", float("nan"))),
        "attn_best_acc": _to_float_or_nan(result.get("attn_best_acc", float("nan"))),
        "final_eval_accuracy": _to_float_or_nan(result.get("final_eval_accuracy", float("nan"))),
        "final_train_accuracy": _to_float_or_nan(result.get("final_train_accuracy", float("nan"))),
        "mlp_gamma_min": _to_float_or_nan(gamma_min),
        "mlp_final_accuracy": _to_float_or_nan(
            mlp_metrics.get("final_accuracy", mlp_metrics.get("accuracy", float("nan")))
        ),
        "mlp_param_count": _to_float_or_nan(mlp_metrics.get("param_count", float("nan"))),
        "mlp_hidden_dim": _to_float_or_nan(mlp_metrics.get("hidden_dim", float("nan"))),
    }


def _metric_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _mean_success(
    *,
    gammas: List[float],
    gamma_threshold: float | None,
    acc_values: List[float],
    acc_threshold: float | None,
) -> bool:
    if gamma_threshold is None:
        gamma_ok = True
    else:
        if not gammas:
            return False
        gamma_ok = float(np.mean(np.asarray(gammas, dtype=np.float64))) > float(gamma_threshold)
    if acc_threshold is None:
        return gamma_ok
    if not acc_values:
        return False
    acc_ok = float(np.mean(np.asarray(acc_values, dtype=np.float64))) >= float(acc_threshold)
    return gamma_ok and acc_ok


def aggregate_seed_results_for_binary_search(
    *,
    results: List[GPUJobResult],
    gamma_threshold: float | None,
    best_acc_threshold: float | None,
    success_aggregation: str,
    success_metric: str = "best_acc",
) -> tuple[GPUJobResult | None, bool, Dict[str, float], Dict[str, List[float]]]:
    successful = [r for r in results if r.success and isinstance(r.result, dict)]
    if not successful:
        return None, False, {"error": "No successful seed runs"}, {}

    gamma_vals: List[float] = []
    best_acc_vals: List[float] = []
    best_train_acc_vals: List[float] = []
    success_metric_vals: List[float] = []
    mlp_acc_vals: List[float] = []
    seed_success_flags: List[bool] = []
    ranking_scores: List[tuple[float, float, float]] = []

    for result in successful:
        gamma = _to_float_or_nan(result.result.get("mlp_gamma_min", float("nan")))
        best_acc = _to_float_or_nan(result.result.get("best_acc", float("nan")))
        best_train_acc = _to_float_or_nan(result.result.get("best_train_acc", float("nan")))
        success_acc = best_acc if success_metric == "best_acc" else best_train_acc
        mlp_acc = _to_float_or_nan(result.result.get("mlp_final_accuracy", float("nan")))

        gamma_ok = True if gamma_threshold is None else (np.isfinite(gamma) and gamma > float(gamma_threshold))
        acc_ok = True if best_acc_threshold is None else (
            np.isfinite(success_acc) and success_acc >= float(best_acc_threshold)
        )
        seed_success_flags.append(bool(gamma_ok and acc_ok))

        if np.isfinite(gamma):
            gamma_vals.append(float(gamma))
        if np.isfinite(best_acc):
            best_acc_vals.append(float(best_acc))
        if np.isfinite(best_train_acc):
            best_train_acc_vals.append(float(best_train_acc))
        if np.isfinite(success_acc):
            success_metric_vals.append(float(success_acc))
        if np.isfinite(mlp_acc):
            mlp_acc_vals.append(float(mlp_acc))

        ranking_scores.append(
            (
                float(success_acc) if np.isfinite(success_acc) else float("-inf"),
                float(gamma) if np.isfinite(gamma) else float("-inf"),
                float(best_acc) if np.isfinite(best_acc) else float("-inf"),
            )
        )

    if success_aggregation == "best":
        success = any(seed_success_flags)
    elif success_aggregation == "all":
        success = all(seed_success_flags)
    else:
        success = _mean_success(
            gammas=gamma_vals,
            gamma_threshold=gamma_threshold,
            acc_values=success_metric_vals,
            acc_threshold=best_acc_threshold,
        )

    best_idx = max(range(len(successful)), key=lambda i: ranking_scores[i])
    best_result = successful[best_idx]

    summary = {
        "n_successful_seeds": float(len(successful)),
        "seed_success_rate": float(np.mean(np.asarray(seed_success_flags, dtype=np.float64))),
    }
    for key, values in (
        ("mlp_gamma_min", gamma_vals),
        ("best_acc", best_acc_vals),
        ("best_train_acc", best_train_acc_vals),
        ("success_metric", success_metric_vals),
        ("mlp_final_accuracy", mlp_acc_vals),
    ):
        stats = _metric_stats(values)
        summary[f"{key}_mean"] = stats["mean"]
        summary[f"{key}_std"] = stats["std"]
        summary[f"{key}_min"] = stats["min"]
        summary[f"{key}_max"] = stats["max"]

    seed_values = {
        "mlp_gamma_min": gamma_vals,
        "best_acc": best_acc_vals,
        "best_train_acc": best_train_acc_vals,
        "success_metric": success_metric_vals,
        "mlp_final_accuracy": mlp_acc_vals,
        "seed_success": [1.0 if flag else 0.0 for flag in seed_success_flags],
    }
    return best_result, bool(success), summary, seed_values


@dataclass
class _AttnMlpExptConfig:
    attention_config: Any
    mlp_method: str
    mlp_hidden_dim: int
    mlp_dtype: Optional[torch.dtype]
    mlp_method_kwargs: Optional[dict]
    seed: int
    save_dir: Optional[str] = None

    def finalize(self) -> None:
        dataset_config = self.attention_config.dataset_config
        train_config = self.attention_config.train_config
        if hasattr(dataset_config, "custom_finalize"):
            dataset_config.custom_finalize()
        if hasattr(train_config, "custom_finalize"):
            train_config.custom_finalize()


@dataclass
class _TrainAssociativeRecallConfig:
    config: Any

    def finalize(self) -> None:
        dataset_config = self.config.dataset_config
        train_config = self.config.train_config
        if hasattr(dataset_config, "custom_finalize"):
            dataset_config.custom_finalize()
        if hasattr(train_config, "custom_finalize"):
            train_config.custom_finalize()


@dataclass
class _MSearchContext:
    d_model: int
    num_facts: int
    junk_len: int
    junk_vocab_size: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    steps_per_dataset: int
    disable_early_stopping: bool
    attn_residual: bool
    freeze_value_dense_identity: bool
    schedule: str
    mlp_method: str
    mlp_dtype: Optional[torch.dtype]
    mlp_method_kwargs: Optional[dict]
    use_eval_mlp_for_eval: bool
    device: str
    dtype_name: str
    embeddings_dir: Optional[str] = None
    attn_norm_type: str = "rmsnorm"
    mlp_norm_type: str = "unit_rmsnorm"
    lm_head_norm_type: str = "unit_rmsnorm"
    mlp_acc_threshold: Optional[float] = None


@torch.no_grad()
def _compute_gamma_min(mlp, factset, device, dtype):
    mlp.eval()
    keys = factset.input_embeddings.to(device=device, dtype=dtype)
    values_all = factset.output_embeddings.to(device=device, dtype=dtype)
    n = keys.shape[0]
    value_idx = torch.tensor(
        [factset.mapping.get_output(i) for i in range(n)],
        device=device,
        dtype=torch.long,
    )
    outputs = mlp(keys)
    outputs = outputs / outputs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    scores = outputs @ values_all.T
    batch = torch.arange(n, device=device)
    correct = scores[batch, value_idx]
    scores_masked = scores.clone()
    scores_masked[batch, value_idx] = float("-inf")
    return (correct - scores_masked.max(dim=1).values).min().item()


def _run_two_phase_experiment(config: _AttnMlpExptConfig) -> dict:
    attn_cfg = config.attention_config
    dataset_config = attn_cfg.dataset_config
    train_config = attn_cfg.train_config
    device = torch.device(
        train_config.device if torch.cuda.is_available() or train_config.device == "cpu" else "cpu"
    )
    dtype = train_config.dtype

    attn_results = train_attention(attn_cfg)
    gpt_model = attn_results["gpt_model"]
    factset = attn_results["factset"]
    attn_best_acc = attn_results["best_acc"]

    eval_mapping = create_random_permutation_mapping(factset.vocab_size, seed=config.seed + 7777)
    mlp_factset = Factset(
        input_embeddings=factset.input_embeddings,
        output_embeddings=factset.output_embeddings,
        mapping=eval_mapping,
        d_model=factset.d_model,
        vocab_size=factset.vocab_size,
    )

    mlp_config = copy.deepcopy(attn_cfg)
    mlp_config.train_config.mlp_method = config.mlp_method
    mlp_config.train_config.mlp_hidden_dim = config.mlp_hidden_dim
    mlp_config.train_config.mlp_dtype = config.mlp_dtype
    mlp_config.train_config.mlp_method_kwargs = config.mlp_method_kwargs
    mlp, mlp_metrics = build_fact_mlp(mlp_config, mlp_factset)

    all_tensors = list(itertools.chain(mlp.parameters(), mlp.buffers()))
    mlp_native_dtype = all_tensors[0].dtype if all_tensors else dtype
    gamma_min = _compute_gamma_min(mlp, mlp_factset, device, mlp_native_dtype)
    mlp_metrics["gamma_min"] = gamma_min
    mlp_metrics["hidden_dim"] = config.mlp_hidden_dim

    mlp = mlp.to(device=device, dtype=dtype)
    gpt_model.eval()
    insert_mlp_into_gpt(
        gpt_model,
        mlp,
        gpt_model.transformer.wte,
        freeze_mlp=True,
        freeze_wte=train_config.transformer_config.freeze_input_embeddings,
        freeze_lm_head=train_config.transformer_config.freeze_output_embeddings,
    )

    _, eval_dataloader = create_associative_recall_batches(
        dataset_config,
        train_config,
        mlp_factset.mapping,
        device=device,
    )
    num_iters = max(
        1,
        int(factset.vocab_size / (len(eval_dataloader) * eval_dataloader.batch_size)),
    ) * 10
    gpt_model.eval()
    eval_result = evaluate(gpt_model, eval_dataloader, device, num_iterations=num_iters)

    gpt_model.to("cpu")
    mlp.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_acc": eval_result["accuracy"],
        "attn_best_acc": attn_best_acc,
        "mlp_gamma_min": gamma_min,
        "mlp_metrics": mlp_metrics,
    }


@pydraclass
class TransformerNumFactsBinarySearchConfig(BinarySearchConfig):
    method_label: str = "gd"
    d_model: int = 64
    reference_num_facts: int = 512
    hidden_dim: int = 32
    junk_len: int = 2
    gamma_success_threshold: float | None = None
    best_acc_success_threshold: float | None = None
    success_metric: str = "best_acc"
    seed_success_aggregation: str = "mean"
    schedule: str = "attn_pretrain_then_insert"
    use_eval_mlp_for_eval: bool = False
    mlp_method: str = "gd"
    mlp_hidden_dim: int = 32
    mlp_dtype: Optional[torch.dtype] = None
    mlp_method_kwargs: Optional[dict] = None

    def get_experiment_config_and_base_dir(self, num_facts: int, **kwargs):
        config = copy.deepcopy(self.base_experiment_config)
        num_facts_int = max(1, int(round(float(num_facts))))
        seed = int(kwargs.get("seed", 42))

        config.dataset_config.num_facts = num_facts_int
        config.train_config.seed = seed
        base_dir = f"{self.base_dir}/num_facts_{num_facts_int}/seed_{seed}"
        os.makedirs(base_dir, exist_ok=True)

        if self.schedule == "insert_then_train_attn":
            config.train_config.mlp_method = self.mlp_method
            config.train_config.mlp_hidden_dim = self.mlp_hidden_dim
            config.train_config.mlp_dtype = self.mlp_dtype
            config.train_config.mlp_method_kwargs = self.mlp_method_kwargs
            if self.use_eval_mlp_for_eval:
                config.train_config.eval_mlp_method = self.mlp_method
                config.train_config.eval_mlp_hidden_dim = self.mlp_hidden_dim
                config.train_config.eval_mlp_dtype = self.mlp_dtype
                config.train_config.eval_mlp_method_kwargs = self.mlp_method_kwargs
            config.train_config.base_dir = base_dir
            config.train_config.save_dir = None
            config.train_config.figs_dir = None
            return _TrainAssociativeRecallConfig(config=config), base_dir

        return _AttnMlpExptConfig(
            attention_config=config,
            mlp_method=self.mlp_method,
            mlp_hidden_dim=self.mlp_hidden_dim,
            mlp_dtype=self.mlp_dtype,
            mlp_method_kwargs=self.mlp_method_kwargs,
            seed=seed,
            save_dir=base_dir,
        ), base_dir

    def run_experiment_config(
        self, config: _AttnMlpExptConfig | _TrainAssociativeRecallConfig
    ):
        if self.schedule == "insert_then_train_attn":
            full_result = train_associative_recall(config.config)
            slim = extract_scalar_metrics(full_result)
            slim["num_facts"] = float(config.config.dataset_config.num_facts)
            slim["d_model"] = float(config.config.train_config.embeddings_config.d_model)
            slim["mlp_hidden_dim"] = float(config.config.train_config.mlp_hidden_dim)
            return slim

        full_result = _run_two_phase_experiment(config)
        slim = extract_scalar_metrics(full_result)
        slim["num_facts"] = float(config.attention_config.dataset_config.num_facts)
        slim["d_model"] = float(config.attention_config.train_config.embeddings_config.d_model)
        slim["mlp_hidden_dim"] = float(config.mlp_hidden_dim)
        return slim

    def agg_results(self, results: List[GPUJobResult]) -> tuple[bool, Any]:
        best_result, success, summary, seed_values = aggregate_seed_results_for_binary_search(
            results=results,
            gamma_threshold=self.gamma_success_threshold,
            best_acc_threshold=self.best_acc_success_threshold,
            success_aggregation=self.seed_success_aggregation,
            success_metric=self.success_metric,
        )
        if best_result is None:
            return False, None
        payload = dict(best_result.result)
        payload["seed_summary"] = summary
        payload["seed_values"] = seed_values
        best_result.result = payload
        return bool(success), best_result


@pydraclass
class TransformerHiddenDimBinarySearchConfig(BinarySearchConfig):
    method_label: str = "ntk"
    d_model: int = 128
    num_facts: int = 1024
    gamma_success_threshold: float | None = None
    best_acc_success_threshold: float | None = 0.98
    success_metric: str = "best_acc"
    seed_success_aggregation: str = "mean"
    schedule: str = "insert_then_train_attn"
    mlp_method: str = "ntk"
    mlp_dtype: Optional[torch.dtype] = None
    mlp_method_kwargs: Optional[dict] = None
    use_eval_mlp_for_eval: bool = False

    def get_experiment_config_and_base_dir(self, m: int, **kwargs):
        m_int = max(1, int(round(float(m))))
        seed = int(kwargs.get("seed", 42))
        ctx: _MSearchContext = copy.deepcopy(self.base_experiment_config)
        base_dir = os.path.join(self.base_dir, f"m_{m_int}", f"seed_{seed}")
        os.makedirs(base_dir, exist_ok=True)

        if self.schedule == "insert_then_train_attn":
            cfg = build_insert_then_train_attn_config(
                d_model=int(ctx.d_model),
                num_facts=int(ctx.num_facts),
                junk_len=int(ctx.junk_len),
                junk_vocab_size=int(ctx.junk_vocab_size),
                mlp_method=str(ctx.mlp_method),
                hidden_dim=m_int,
                num_epochs=int(ctx.num_epochs),
                batch_size=int(ctx.batch_size),
                learning_rate=float(ctx.learning_rate),
                steps_per_dataset=int(ctx.steps_per_dataset),
                disable_early_stopping=bool(ctx.disable_early_stopping),
                attn_residual=bool(ctx.attn_residual),
                freeze_value_dense_identity=bool(ctx.freeze_value_dense_identity),
                seed=seed,
                device=str(ctx.device),
                dtype_name=str(ctx.dtype_name),
                mlp_dtype=ctx.mlp_dtype,
                mlp_method_kwargs=ctx.mlp_method_kwargs,
                embeddings_dir=ctx.embeddings_dir,
                attn_norm_type=str(ctx.attn_norm_type),
                mlp_norm_type=str(ctx.mlp_norm_type),
                lm_head_norm_type=str(ctx.lm_head_norm_type),
                mlp_acc_threshold=ctx.mlp_acc_threshold,
            )
            if ctx.use_eval_mlp_for_eval:
                cfg.train_config.eval_mlp_method = ctx.mlp_method
                cfg.train_config.eval_mlp_hidden_dim = m_int
                cfg.train_config.eval_mlp_dtype = ctx.mlp_dtype
                cfg.train_config.eval_mlp_method_kwargs = ctx.mlp_method_kwargs
            cfg.train_config.base_dir = base_dir
            cfg.train_config.save_dir = None
            cfg.train_config.figs_dir = None
            return _TrainAssociativeRecallConfig(config=cfg), base_dir

        attn_cfg = build_attn_pretrain_config(
            d_model=int(ctx.d_model),
            num_facts=int(ctx.num_facts),
            junk_len=int(ctx.junk_len),
            junk_vocab_size=int(ctx.junk_vocab_size),
            num_epochs=int(ctx.num_epochs),
            batch_size=int(ctx.batch_size),
            learning_rate=float(ctx.learning_rate),
            steps_per_dataset=int(ctx.steps_per_dataset),
            disable_early_stopping=bool(ctx.disable_early_stopping),
            attn_residual=bool(ctx.attn_residual),
            seed=seed,
            device=str(ctx.device),
            dtype_name=str(ctx.dtype_name),
        )
        return _AttnMlpExptConfig(
            attention_config=attn_cfg,
            mlp_method=str(ctx.mlp_method),
            mlp_hidden_dim=m_int,
            mlp_dtype=ctx.mlp_dtype,
            mlp_method_kwargs=ctx.mlp_method_kwargs,
            seed=seed,
            save_dir=base_dir,
        ), base_dir

    def run_experiment_config(
        self, config: _AttnMlpExptConfig | _TrainAssociativeRecallConfig
    ):
        if self.schedule == "insert_then_train_attn":
            full_result = train_associative_recall(config.config)
            slim = extract_scalar_metrics(full_result)
            slim["num_facts"] = float(config.config.dataset_config.num_facts)
            slim["d_model"] = float(config.config.train_config.embeddings_config.d_model)
            slim["mlp_hidden_dim"] = float(config.config.train_config.mlp_hidden_dim)
            slim["m"] = float(config.config.train_config.mlp_hidden_dim)
        else:
            full_result = _run_two_phase_experiment(config)
            slim = extract_scalar_metrics(full_result)
            slim["num_facts"] = float(config.attention_config.dataset_config.num_facts)
            slim["d_model"] = float(config.attention_config.train_config.embeddings_config.d_model)
            slim["mlp_hidden_dim"] = float(config.mlp_hidden_dim)
            slim["m"] = float(config.mlp_hidden_dim)

        d_model = int(round(float(slim["d_model"])))
        m_int = int(round(float(slim["m"])))
        param_count = float(slim.get("mlp_param_count", float("nan")))
        if not np.isfinite(param_count) or param_count <= 0:
            slim["mlp_param_count"] = float(3 * d_model * m_int + 2 * m_int + d_model)
        slim["method"] = self.method_label
        return slim

    def agg_results(self, results: List[GPUJobResult]) -> tuple[bool, Any]:
        best_result, success, summary, seed_values = aggregate_seed_results_for_binary_search(
            results=results,
            gamma_threshold=self.gamma_success_threshold,
            best_acc_threshold=self.best_acc_success_threshold,
            success_aggregation=self.seed_success_aggregation,
            success_metric=self.success_metric,
        )
        if best_result is None:
            return False, None
        payload = dict(best_result.result)
        payload["seed_summary"] = summary
        payload["seed_values"] = seed_values
        best_result.result = payload
        return bool(success), best_result


def seeds_from_config(config: Any) -> list[int]:
    if getattr(config, "seeds_override", None):
        return sorted({int(seed) for seed in config.seeds_override})
    return [42 + i * 10000 for i in range(int(config.n_seeds))]


def build_num_facts_sweep_configs(config: Any) -> list[TransformerNumFactsBinarySearchConfig]:
    base_dir_root = Path(config.output_root) / "raw"
    seeds = seeds_from_config(config)
    sweep_configs: list[TransformerNumFactsBinarySearchConfig] = []

    for method_label in config.mlp_methods:
        spec = METHOD_SPECS[method_label]
        for d_model, reference_num_facts in config.model_configs:
            hidden_dims = get_hidden_dims(
                d_model=int(d_model),
                reference_num_facts=int(reference_num_facts),
                hidden_dim_multipliers=config.hidden_dim_multipliers,
                base_hidden_dim_multiplier=config.base_hidden_dim_multiplier,
                hidden_dims_override=config.hidden_dims_override,
            )
            for hidden_dim in hidden_dims:
                cap_mult = spec.get("capacity_multiplier", 1.0)
                num_facts_range = get_num_facts_search_range(
                    d_model=int(d_model),
                    hidden_dim=int(hidden_dim),
                    low_multiplier=float(config.num_facts_low_multiplier) * cap_mult,
                    high_multiplier=float(config.num_facts_high_multiplier) * cap_mult,
                    min_num_facts=int(config.min_num_facts),
                    max_num_facts=int(config.max_num_facts),
                )

                if config.schedule == "insert_then_train_attn":
                    base_experiment_config = build_insert_then_train_attn_config(
                        d_model=int(d_model),
                        num_facts=int(num_facts_range[0]),
                        junk_len=int(config.junk_len),
                        junk_vocab_size=int(config.junk_vocab_size),
                        mlp_method=spec["mlp_method"],
                        hidden_dim=int(hidden_dim),
                        num_epochs=int(config.num_epochs),
                        batch_size=int(config.batch_size),
                        learning_rate=float(config.learning_rate),
                        steps_per_dataset=int(config.steps_per_dataset),
                        disable_early_stopping=bool(config.disable_early_stopping),
                        attn_residual=bool(config.attn_residual),
                        freeze_value_dense_identity=bool(config.freeze_value_dense_identity),
                        seed=42,
                        device=str(config.device),
                        dtype_name=str(config.dtype),
                        mlp_dtype=spec["mlp_dtype"],
                        mlp_method_kwargs=spec["mlp_method_kwargs"],
                    )
                else:
                    base_experiment_config = build_attn_pretrain_config(
                        d_model=int(d_model),
                        num_facts=int(num_facts_range[0]),
                        junk_len=int(config.junk_len),
                        junk_vocab_size=int(config.junk_vocab_size),
                        num_epochs=int(config.num_epochs),
                        batch_size=int(config.batch_size),
                        learning_rate=float(config.learning_rate),
                        steps_per_dataset=int(config.steps_per_dataset),
                        disable_early_stopping=bool(config.disable_early_stopping),
                        attn_residual=bool(config.attn_residual),
                        seed=42,
                        device=str(config.device),
                        dtype_name=str(config.dtype),
                    )

                run_dir = base_dir_root / method_label / f"d{d_model}_fref{reference_num_facts}" / f"junk_len_{config.junk_len}" / f"m{hidden_dim}"
                run_dir.mkdir(parents=True, exist_ok=True)
                sweep_configs.append(
                    TransformerNumFactsBinarySearchConfig(
                        base_dir=str(run_dir),
                        prop="num_facts",
                        range=num_facts_range,
                        precision=max(1, int(config.binary_search_precision)),
                        success_direction_lower=bool(config.binary_search_success_direction_lower),
                        sweep_props={"seed": seeds},
                        base_experiment_config=base_experiment_config,
                        method_label=method_label,
                        d_model=int(d_model),
                        reference_num_facts=int(reference_num_facts),
                        hidden_dim=int(hidden_dim),
                        junk_len=int(config.junk_len),
                        schedule=str(config.schedule),
                        gamma_success_threshold=config.gamma_success_threshold,
                        best_acc_success_threshold=config.best_acc_success_threshold,
                        success_metric=str(config.success_metric),
                        seed_success_aggregation=str(config.seed_success_aggregation),
                        mlp_method=spec["mlp_method"],
                        mlp_hidden_dim=int(hidden_dim),
                        mlp_dtype=spec["mlp_dtype"],
                        mlp_method_kwargs=spec["mlp_method_kwargs"],
                        use_eval_mlp_for_eval=bool(config.use_eval_mlp_for_eval),
                    )
                )
    return sweep_configs


def _validate_llm_embedding_mode(config: Any) -> None:
    """Sanity-check LLM-embedding mode before building sweep configs.

    - Only ``insert_then_train_attn`` is supported (the ``attn_pretrain_then_insert``
      path uses ``train_attention`` / ``_run_two_phase_experiment``, which have
      not been wired for LLM activations yet).
    - Each ``num_facts + junk_vocab_size + 1`` must fit within the pool ``N``.
    - ``d_models`` entries that disagree with the data's ``d`` get warn-and-override.
    """
    embeddings_dir = getattr(config, "embeddings_dir", None)
    if embeddings_dir is None:
        return
    if config.schedule != "insert_then_train_attn":
        raise ValueError(
            f"embeddings_dir is only supported with schedule="
            f"'insert_then_train_attn'; got {config.schedule!r}."
        )
    n_rows, d_from_data = _peek_embeddings_shape(embeddings_dir)
    for num_facts in config.num_facts_values:
        needed = int(num_facts) + int(config.junk_vocab_size) + 1
        if needed > n_rows:
            raise ValueError(
                f"num_facts={int(num_facts)} + junk_vocab_size="
                f"{int(config.junk_vocab_size)} + 1 = {needed} exceeds available "
                f"rows N={n_rows} in {embeddings_dir!r}"
            )
    cleaned_d_models: list[int] = []
    for d_model in config.d_models:
        if int(d_model) != d_from_data:
            print(
                f"[LLM mode] overriding d_models entry {int(d_model)} -> "
                f"{d_from_data} from {embeddings_dir!r} (N={n_rows})"
            )
        cleaned_d_models.append(d_from_data)
    # Deduplicate while preserving order (all entries collapse to d_from_data).
    config.d_models = list(dict.fromkeys(cleaned_d_models))


def build_hidden_dim_sweep_configs(config: Any) -> list[TransformerHiddenDimBinarySearchConfig]:
    _validate_llm_embedding_mode(config)
    base_dir_root = Path(config.output_root) / "raw"
    seeds = seeds_from_config(config)
    sweep_configs: list[TransformerHiddenDimBinarySearchConfig] = []
    embeddings_dir = getattr(config, "embeddings_dir", None)

    for method_label in config.mlp_methods:
        spec = METHOD_SPECS[method_label]
        for d_model in config.d_models:
            for num_facts in config.num_facts_values:
                run_dir = base_dir_root / method_label / f"d{d_model}_F{int(num_facts)}"
                run_dir.mkdir(parents=True, exist_ok=True)
                ctx = _MSearchContext(
                    d_model=int(d_model),
                    num_facts=int(num_facts),
                    junk_len=int(config.junk_len),
                    junk_vocab_size=int(config.junk_vocab_size),
                    num_epochs=int(config.num_epochs),
                    batch_size=int(config.batch_size),
                    learning_rate=float(config.learning_rate),
                    steps_per_dataset=int(config.steps_per_dataset),
                    disable_early_stopping=bool(config.disable_early_stopping),
                    attn_residual=bool(config.attn_residual),
                    freeze_value_dense_identity=bool(config.freeze_value_dense_identity),
                    schedule=str(config.schedule),
                    mlp_method=str(spec["mlp_method"]),
                    mlp_dtype=spec["mlp_dtype"],
                    mlp_method_kwargs=spec["mlp_method_kwargs"],
                    use_eval_mlp_for_eval=bool(config.use_eval_mlp_for_eval),
                    device=str(config.device),
                    dtype_name=str(config.dtype),
                    embeddings_dir=embeddings_dir,
                    attn_norm_type=str(getattr(config, "attn_norm_type", "rmsnorm")),
                    mlp_norm_type=str(getattr(config, "mlp_norm_type", "unit_rmsnorm")),
                    lm_head_norm_type=str(getattr(config, "lm_head_norm_type", "unit_rmsnorm")),
                    mlp_acc_threshold=(
                        float(config.best_acc_success_threshold)
                        if config.best_acc_success_threshold is not None
                        else None
                    ),
                )
                sweep_configs.append(
                    TransformerHiddenDimBinarySearchConfig(
                        base_dir=str(run_dir),
                        prop="m",
                        range=(int(config.hidden_dim_search_min), int(config.hidden_dim_search_max)),
                        precision=max(1, int(config.binary_search_precision)),
                        success_direction_lower=bool(config.binary_search_success_direction_lower),
                        sweep_props={"seed": seeds},
                        base_experiment_config=ctx,
                        method_label=method_label,
                        d_model=int(d_model),
                        num_facts=int(num_facts),
                        gamma_success_threshold=config.gamma_success_threshold,
                        best_acc_success_threshold=config.best_acc_success_threshold,
                        success_metric=str(config.success_metric),
                        seed_success_aggregation=str(config.seed_success_aggregation),
                        schedule=str(config.schedule),
                        mlp_method=spec["mlp_method"],
                        mlp_dtype=spec["mlp_dtype"],
                        mlp_method_kwargs=spec["mlp_method_kwargs"],
                        use_eval_mlp_for_eval=bool(config.use_eval_mlp_for_eval),
                    )
                )
    return sweep_configs


def _run_job_locally(job) -> GPUJobResult:
    out_file = job.get_out_file()
    try:
        result_value = None
        if out_file is None:
            result_value = job.run()
        else:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                print("Running job locally", file=f, flush=True)
                with redirect_stdout(f), redirect_stderr(f):
                    result_value = job.run()
        return GPUJobResult(
            success=True,
            error=None,
            gpu_id=-1,
            out_file=out_file,
            job=job,
            result=result_value,
        )
    except Exception as exc:
        return GPUJobResult(
            success=False,
            error=str(exc),
            gpu_id=-1,
            out_file=out_file,
            job=job,
            result=None,
        )


def _run_binary_search_local(binary_config: BinarySearchConfig):
    lo, hi = binary_config.range
    precision = binary_config.precision
    achieved_results = None
    failed_results = None
    while (hi - lo) >= precision:
        mid = (lo + hi) / 2
        jobs = get_jobs_for_mid(binary_config, mid)
        job_results = [_run_job_locally(job) for job in jobs]
        success, aggregated_result = binary_config.agg_results(job_results)
        if success:
            achieved_results = (mid, aggregated_result)
            if binary_config.success_direction_lower:
                hi = mid
            else:
                lo = mid
        else:
            failed_results = (mid, aggregated_result)
            if binary_config.success_direction_lower:
                lo = mid
            else:
                hi = mid

    binary_search_results = {
        "search_range": [lo, hi],
        "precision": precision,
        "achieved_results": achieved_results,
        "failed_results": failed_results,
        "timestamp": timestamp_string(),
    }
    results_filename = f"{binary_config.base_dir}/binary_search_results_{timestamp_string()}.pkl"
    pd.to_pickle(binary_search_results, results_filename)
    return achieved_results, failed_results


def run_binary_searches_portable(
    configs: list[BinarySearchConfig],
    *,
    max_gpus: int | None = None,
    simultaneous_jobs_per_gpu: int | None = None,
    use_local_runner: bool | None = None,
):
    if max_gpus == 0:
        return [_run_binary_search_local(config) for config in configs]
    if use_local_runner is True:
        return [_run_binary_search_local(config) for config in configs]
    if use_local_runner is None:
        if max_gpus == 0 or torch.cuda.device_count() == 0:
            return [_run_binary_search_local(config) for config in configs]
    return run_gpu_binary_searches(
        configs,
        max_gpus=max_gpus,
        simultaneous_jobs_per_gpu=simultaneous_jobs_per_gpu,
    )


class _CompatUnpickler(pickle.Unpickler):
    _dummy_cls_cache: dict[tuple[str, str], type] = {}

    @classmethod
    def _dummy_cls(cls, module: str, name: str) -> type:
        def _noop(*_args, **_kwargs):
            return None

        def _fallback_getattr(self, _attr):
            return _noop

        def _setstate(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

        key = (module, name)
        if key not in cls._dummy_cls_cache:
            cls._dummy_cls_cache[key] = type(
                name,
                (),
                {
                    "__module__": module,
                    "__getattr__": _fallback_getattr,
                    "__setstate__": _setstate,
                    "run_experiment_config": _noop,
                    "get_experiment_config_and_base_dir": _noop,
                    "agg_results": _noop,
                },
            )
        return cls._dummy_cls_cache[key]

    def find_class(self, module: str, name: str):
        if module == "__main__":
            return self._dummy_cls(module, name)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_pickle_compat(path: Path) -> Any:
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()
