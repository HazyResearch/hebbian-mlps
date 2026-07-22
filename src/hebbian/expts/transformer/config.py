"""Presets and resolved configurations for Transformer capacity sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


DEFAULT_RELEASE_METHODS = [
    "gd",
    "hebbian",
    "hebbian_whitened",
    "cf_coord_whitened",
    "ntk",
]

ALL_METHODS = list(DEFAULT_RELEASE_METHODS)

VALID_SCHEDULES = ["attn_pretrain_then_insert", "insert_then_train_attn"]
VALID_SUCCESS_METRICS = ["best_acc", "best_train_acc"]
VALID_SUCCESS_AGGREGATIONS = ["mean", "best", "all"]
DEFAULT_OUTPUT_BASE_DIR = "./artifacts/transformer_capacity"


@dataclass
class NumFactsConfig:
    preset: str = "full_num_facts_attn_pretrain"
    orientation: str = "num_facts"
    schedule: str = "attn_pretrain_then_insert"
    model_configs: list[tuple[int, int]] = field(default_factory=lambda: [(8, 8)])
    hidden_dim_multipliers: list[float] = field(default_factory=lambda: [1.0])
    base_hidden_dim_multiplier: float = 1.0
    hidden_dims_override: list[int] = field(default_factory=lambda: [2])
    junk_len: int = 2
    junk_vocab_size: int = 2
    mlp_methods: list[str] = field(default_factory=lambda: list(DEFAULT_RELEASE_METHODS))
    num_epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 2e-4
    steps_per_dataset: int = 1
    disable_early_stopping: bool = True
    attn_residual: bool = True
    freeze_value_dense_identity: bool = False
    use_eval_mlp_for_eval: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    n_seeds: int = 1
    seeds_override: list[int] = field(default_factory=list)
    num_facts_low_multiplier: float = 0.5
    num_facts_high_multiplier: float = 1.5
    min_num_facts: int = 1
    max_num_facts: int = 16
    binary_search_precision: int = 8
    binary_search_success_direction_lower: bool = False
    gamma_success_threshold: float | None = None
    best_acc_success_threshold: float | None = 0.0
    success_metric: str = "best_acc"
    seed_success_aggregation: str = "mean"
    max_gpus: int = 1
    simultaneous_jobs_per_gpu: int = 1
    output_base_dir: str = DEFAULT_OUTPUT_BASE_DIR
    output_root: str | None = None
    timestamp: str | None = None
    use_local_runner: bool | None = True


@dataclass
class HiddenDimConfig:
    preset: str = "paper_trainacc100_hidden_dim"
    orientation: str = "hidden_dim"
    schedule: str = "insert_then_train_attn"
    d_models: list[int] = field(default_factory=lambda: [8])
    num_facts_values: list[int] = field(default_factory=lambda: [8])
    mlp_methods: list[str] = field(default_factory=lambda: list(DEFAULT_RELEASE_METHODS))
    junk_len: int = 2
    junk_vocab_size: int = 2
    num_epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 2e-4
    steps_per_dataset: int = 1
    disable_early_stopping: bool = True
    attn_residual: bool = True
    freeze_value_dense_identity: bool = False
    use_eval_mlp_for_eval: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    hidden_dim_search_min: int = 1
    hidden_dim_search_max: int = 3
    binary_search_precision: int = 1
    binary_search_success_direction_lower: bool = True
    n_seeds: int = 1
    seeds_override: list[int] = field(default_factory=list)
    gamma_success_threshold: float | None = None
    best_acc_success_threshold: float | None = 0.0
    success_metric: str = "best_acc"
    seed_success_aggregation: str = "mean"
    max_gpus: int = 1
    simultaneous_jobs_per_gpu: int = 1
    output_base_dir: str = DEFAULT_OUTPUT_BASE_DIR
    output_root: str | None = None
    timestamp: str | None = None
    use_local_runner: bool | None = True
    # LLM-embedding mode: directory containing x.pt / y.pt activation tables.
    # When set, the sweep trains with embeddings loaded from disk;
    # d_models is overridden from the data and tie_embeddings is forced to
    # False. Only the "insert_then_train_attn" schedule is supported.
    embeddings_dir: str | None = None
    # Override the transformer-block norm types in build_insert_then_train_attn_config.
    # Defaults match the existing hardcoded preset behavior; "none" disables a
    # layer. In LLM mode "frozen_rmsnorm" is rejected at training startup
    # (raw LLM activations break the row-RMS-uniformity assert).
    attn_norm_type: str = "rmsnorm"
    mlp_norm_type: str = "unit_rmsnorm"
    lm_head_norm_type: str = "unit_rmsnorm"


NUM_FACTS_PRESETS: dict[str, dict[str, Any]] = {
    "full_num_facts_attn_pretrain": {
        "schedule": "attn_pretrain_then_insert",
        "model_configs": [(64, 512), (90, 1012), (128, 2048)],
        "hidden_dim_multipliers": [0.5, 1.0, 2.0, 4.0, 8.0],
        "hidden_dims_override": [],
        "junk_len": 9,
        "junk_vocab_size": 9,
        "mlp_methods": list(DEFAULT_RELEASE_METHODS),
        "num_epochs": 10000,
        "batch_size": 1280,
        "learning_rate": 2e-4,
        "steps_per_dataset": 1,
        "disable_early_stopping": True,
        "attn_residual": True,
        "freeze_value_dense_identity": False,
        "use_eval_mlp_for_eval": False,
        "device": "cuda",
        "dtype": "float32",
        "n_seeds": 1,
        "num_facts_low_multiplier": 0.0,
        "num_facts_high_multiplier": 4.0,
        "min_num_facts": 1,
        "max_num_facts": 65536,
        "binary_search_precision": 16,
        "binary_search_success_direction_lower": False,
        "gamma_success_threshold": None,
        "best_acc_success_threshold": 0.98,
        "success_metric": "best_acc",
        "seed_success_aggregation": "mean",
        "max_gpus": 4,
        "simultaneous_jobs_per_gpu": 1,
        "use_local_runner": None,
    },
    "paper_train99_num_facts": {
        "schedule": "insert_then_train_attn",
        "model_configs": [(128, 2048)],
        "hidden_dim_multipliers": [],
        "hidden_dims_override": [44, 88, 176, 352, 704, 1408],
        "junk_len": 9,
        "junk_vocab_size": 9,
        "mlp_methods": list(DEFAULT_RELEASE_METHODS),
        "num_epochs": 4000,
        "batch_size": 1280,
        "learning_rate": 2e-4,
        "steps_per_dataset": 1,
        "disable_early_stopping": True,
        "attn_residual": True,
        "freeze_value_dense_identity": False,
        "use_eval_mlp_for_eval": False,
        "device": "cuda",
        "dtype": "float32",
        "n_seeds": 1,
        "num_facts_low_multiplier": 0.0,
        "num_facts_high_multiplier": 1.0,
        "min_num_facts": 1,
        "max_num_facts": 65536,
        "binary_search_precision": 16,
        "binary_search_success_direction_lower": False,
        "gamma_success_threshold": None,
        "best_acc_success_threshold": 0.99,
        "success_metric": "best_train_acc",
        "seed_success_aggregation": "mean",
        "max_gpus": 4,
        "simultaneous_jobs_per_gpu": 1,
        "use_local_runner": None,
    },
}


HIDDEN_DIM_PRESETS: dict[str, dict[str, Any]] = {
    "integration_hidden_dim": {
        "schedule": "insert_then_train_attn",
        "d_models": [8],
        "num_facts_values": [8],
        "mlp_methods": ["hebbian_whitened"],
        "junk_len": 2,
        "junk_vocab_size": 2,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "steps_per_dataset": 1,
        "disable_early_stopping": True,
        "attn_residual": True,
        "freeze_value_dense_identity": False,
        "use_eval_mlp_for_eval": False,
        "device": "cpu",
        "dtype": "float32",
        "hidden_dim_search_min": 1,
        "hidden_dim_search_max": 3,
        "binary_search_precision": 1,
        "binary_search_success_direction_lower": True,
        "n_seeds": 1,
        "gamma_success_threshold": None,
        "best_acc_success_threshold": 0.0,
        "success_metric": "best_acc",
        "seed_success_aggregation": "mean",
        "max_gpus": 1,
        "simultaneous_jobs_per_gpu": 1,
        "use_local_runner": True,
    },
    "paper_evalacc100_hidden_dim": {
        "schedule": "insert_then_train_attn",
        "d_models": [128],
        "num_facts_values": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
        "mlp_methods": list(DEFAULT_RELEASE_METHODS),
        "junk_len": 9,
        "junk_vocab_size": 9,
        "num_epochs": 4000,
        "batch_size": 1280,
        "learning_rate": 2e-4,
        "steps_per_dataset": 1,
        "disable_early_stopping": True,
        "attn_residual": False,
        "freeze_value_dense_identity": True,
        "use_eval_mlp_for_eval": True,
        "device": "cuda",
        "dtype": "float32",
        "hidden_dim_search_min": 1,
        "hidden_dim_search_max": 65536,
        "binary_search_precision": 16,
        "binary_search_success_direction_lower": True,
        "n_seeds": 1,
        "gamma_success_threshold": None,
        "best_acc_success_threshold": 1.0,
        "success_metric": "best_acc",
        "seed_success_aggregation": "mean",
        "max_gpus": 4,
        "simultaneous_jobs_per_gpu": 1,
        "use_local_runner": None,
    },
    "paper_trainacc100_hidden_dim": {
        "schedule": "insert_then_train_attn",
        "d_models": [128],
        "num_facts_values": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
        "mlp_methods": list(DEFAULT_RELEASE_METHODS),
        "junk_len": 9,
        "junk_vocab_size": 9,
        "num_epochs": 4000,
        "batch_size": 1280,
        "learning_rate": 2e-4,
        "steps_per_dataset": 1,
        "disable_early_stopping": True,
        "attn_residual": True,
        "freeze_value_dense_identity": False,
        "use_eval_mlp_for_eval": False,
        "device": "cuda",
        "dtype": "float32",
        "hidden_dim_search_min": 1,
        "hidden_dim_search_max": 65536,
        "binary_search_precision": 16,
        "binary_search_success_direction_lower": True,
        "n_seeds": 1,
        "gamma_success_threshold": None,
        "best_acc_success_threshold": 1.0,
        "success_metric": "best_train_acc",
        "seed_success_aggregation": "mean",
        "max_gpus": 4,
        "simultaneous_jobs_per_gpu": 1,
        "use_local_runner": None,
    },
}


def _normalize_list(value: Iterable[Any] | None) -> list[Any]:
    return [] if value is None else list(value)


def _merge_preset(
    defaults: dict[str, Any],
    preset_values: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(preset_values)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _validate_common(
    *,
    schedule: str,
    methods: Iterable[str],
    success_metric: str,
    seed_success_aggregation: str,
) -> None:
    if schedule not in VALID_SCHEDULES:
        raise ValueError(f"Unknown schedule {schedule!r}. Valid: {VALID_SCHEDULES}")
    unknown_methods = sorted(set(methods) - set(ALL_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown mlp_methods: {unknown_methods}. Valid: {ALL_METHODS}")
    if success_metric not in VALID_SUCCESS_METRICS:
        raise ValueError(
            f"Unknown success_metric {success_metric!r}. Valid: {VALID_SUCCESS_METRICS}"
        )
    if seed_success_aggregation not in VALID_SUCCESS_AGGREGATIONS:
        raise ValueError(
            "Unknown seed_success_aggregation "
            f"{seed_success_aggregation!r}. Valid: {VALID_SUCCESS_AGGREGATIONS}"
        )


def resolve_num_facts_config(
    preset: str = "full_num_facts_attn_pretrain", **overrides: Any
) -> NumFactsConfig:
    if preset not in NUM_FACTS_PRESETS:
        raise ValueError(f"Unknown num-facts preset {preset!r}. Valid: {sorted(NUM_FACTS_PRESETS)}")
    merged = _merge_preset(NumFactsConfig().__dict__, NUM_FACTS_PRESETS[preset], overrides)
    merged["preset"] = preset
    merged["model_configs"] = [tuple(pair) for pair in merged["model_configs"]]
    merged["hidden_dim_multipliers"] = list(merged["hidden_dim_multipliers"])
    merged["hidden_dims_override"] = _normalize_list(merged["hidden_dims_override"])
    merged["mlp_methods"] = _normalize_list(merged["mlp_methods"])
    merged["seeds_override"] = _normalize_list(merged["seeds_override"])
    config = NumFactsConfig(**merged)
    _validate_common(
        schedule=config.schedule,
        methods=config.mlp_methods,
        success_metric=config.success_metric,
        seed_success_aggregation=config.seed_success_aggregation,
    )
    if config.binary_search_precision <= 0:
        raise ValueError("binary_search_precision must be positive.")
    return config


def resolve_hidden_dim_config(
    preset: str = "paper_trainacc100_hidden_dim", **overrides: Any
) -> HiddenDimConfig:
    if preset not in HIDDEN_DIM_PRESETS:
        raise ValueError(
            f"Unknown hidden-dim preset {preset!r}. Valid: {sorted(HIDDEN_DIM_PRESETS)}"
        )
    merged = _merge_preset(HiddenDimConfig().__dict__, HIDDEN_DIM_PRESETS[preset], overrides)
    merged["preset"] = preset
    merged["d_models"] = _normalize_list(merged["d_models"])
    merged["num_facts_values"] = _normalize_list(merged["num_facts_values"])
    merged["mlp_methods"] = _normalize_list(merged["mlp_methods"])
    merged["seeds_override"] = _normalize_list(merged["seeds_override"])
    config = HiddenDimConfig(**merged)
    _validate_common(
        schedule=config.schedule,
        methods=config.mlp_methods,
        success_metric=config.success_metric,
        seed_success_aggregation=config.seed_success_aggregation,
    )
    if config.hidden_dim_search_max <= config.hidden_dim_search_min:
        raise ValueError("hidden_dim_search_max must be greater than hidden_dim_search_min.")
    return config


def parse_csv_list(value: str | None, cast: type = str) -> list[Any] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return []
    return [cast(item) for item in items]


def parse_model_configs(value: str | None) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    out: list[tuple[int, int]] = []
    for chunk in [part.strip() for part in value.split(",") if part.strip()]:
        if ":" not in chunk:
            raise ValueError(f"Invalid model config {chunk!r}; expected d:f format.")
        d_str, f_str = chunk.split(":", maxsplit=1)
        out.append((int(d_str), int(f_str)))
    return out


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"none", "nan", ""}:
        return None
    return float(value)


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value {value!r}.")
