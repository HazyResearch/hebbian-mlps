"""Helpers for Section 3.1 transformer attention-noise experiments.

This module intentionally keeps only the pieces needed for the Section 3.1 sweep:
1. Build the full associative-recall (attention + GD MLP) training config.
2. Measure query-position signal error against the corresponding key.
3. Aggregate scalar metrics across seeds.
"""

from __future__ import annotations

import datetime
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from hebbian.gpu_sweep import GPUJobResult
from hebbian.transformer.config import (
    DatasetConfig,
    EmbeddingsConfig,
    AssociativeRecallConfig,
    TrainingConfig,
    TransformerConfig,
)
from hebbian.transformer.data import AssociativeRecallBatchGenerator
from hebbian.transformer.model import GPT


def get_theoretical_hidden_dim(d_model: int, num_facts: int) -> int:
    """Return ceil((4 * num_facts) / (3 * d_model)) for gated MLP sizing."""
    return int(np.ceil((num_facts * 4) / (d_model * 3)))


def build_associative_recall_config(
    *,
    d_model: int,
    num_facts: int,
    junk_len: int,
    junk_vocab_size: int,
    mlp_method: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    steps_per_dataset: int,
    disable_early_stopping: bool,
    attn_residual: bool,
    seed: int,
    freeze_voproj: bool = False,
) -> AssociativeRecallConfig:
    """Construct the full associative-recall config used in Section 3.1."""
    hidden_dim = 2 * get_theoretical_hidden_dim(d_model, num_facts)
    return AssociativeRecallConfig(
        dataset_config=DatasetConfig(
            num_facts=num_facts,
            junk_vocab_size=junk_vocab_size,
            min_seq_length=junk_len,
            max_seq_length=junk_len,
            use_identity_fact_mapping=False,
        ),
        train_config=TrainingConfig(
            embeddings_config=EmbeddingsConfig(d_model=d_model),
            transformer_config=TransformerConfig(
                attn_residual=attn_residual,
                freeze_value_dense_identity=freeze_voproj,
            ),
            mlp_method=mlp_method,
            mlp_hidden_dim=hidden_dim,
            batch_size=batch_size,
            lr=learning_rate,
            steps_per_dataset=steps_per_dataset,
            epochs=num_epochs,
            early_stop_accuracy=None if disable_early_stopping else 0.99,
            seed=seed,
        ),
    )


def build_attention_only_config(
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
    freeze_voproj: bool = False,
) -> AssociativeRecallConfig:
    """Construct the attention-only associative-recall config for Section 3.1.

    This uses an identity fact mapping and identity MLP so only attention is
    trained.
    """
    return AssociativeRecallConfig(
        dataset_config=DatasetConfig(
            num_facts=num_facts,
            junk_vocab_size=junk_vocab_size,
            min_seq_length=junk_len,
            max_seq_length=junk_len,
            use_identity_fact_mapping=True,
        ),
        train_config=TrainingConfig(
            embeddings_config=EmbeddingsConfig(d_model=d_model),
            transformer_config=TransformerConfig(
                freeze_value_dense_identity=freeze_voproj,
                use_identity_mlp=True,
                no_positional_encoding=True,
                lm_head_norm_type="unit_rmsnorm",
                mlp_norm_type="unit_rmsnorm",
                attn_residual=attn_residual,
                freeze_input_embeddings=True,
                freeze_output_embeddings=True,
            ),
            batch_size=batch_size,
            lr=learning_rate,
            steps_per_dataset=steps_per_dataset,
            epochs=num_epochs,
            early_stop_accuracy=None if disable_early_stopping else 0.99,
            seed=seed,
        ),
    )


def default_base_dir(mlp_method: str) -> str:
    """Return timestamped output root for Section 3.1 sweeps."""
    ts = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
    return f"./artifacts/attention_noise/{mlp_method}_{ts}"


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(float(value)))
    return False


def _distribution_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan"), "p95": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _angular_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Angular distance (radians) between two vectors."""
    # Avoid torch trig/div kernels here; compute via Python scalars to sidestep SIGFPE.
    x_vals = x.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
    y_vals = y.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
    if len(x_vals) != len(y_vals) or len(x_vals) == 0:
        return float("nan")

    dot = 0.0
    sq_x = 0.0
    sq_y = 0.0
    for a, b in zip(x_vals, y_vals):
        if not (math.isfinite(a) and math.isfinite(b)):
            return float("nan")
        dot += a * b
        sq_x += a * a
        sq_y += b * b

    if not (math.isfinite(dot) and math.isfinite(sq_x) and math.isfinite(sq_y)):
        return float("nan")
    if sq_x <= 0.0 or sq_y <= 0.0:
        return float("nan")

    denom = math.sqrt(sq_x) * math.sqrt(sq_y)
    if not math.isfinite(denom) or denom <= 0.0:
        return float("nan")

    cos_theta = dot / denom
    if not math.isfinite(cos_theta):
        return float("nan")
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0

    return float(math.acos(cos_theta))


@torch.no_grad()
def compute_attention_noise_metrics(
    *,
    gpt_model: GPT,
    mapping: Any,
    dataset_config: DatasetConfig,
    train_seed: int,
    measurement_stage: str = "ln2",
    num_batches: int = 64,
    batch_size: int = 512,
) -> Dict[str, float]:
    """Estimate Section 3.1 attention noise statistics.

    Metric definition used here:
    - query vector: model representation at query position `Q`, measured at one of:
      - ``ln2``: block-0 ``ln_2`` output (pre-MLP)
      - ``lnf``: final ``ln_f`` output (pre-lm_head)
    - ideal key vector: token embedding of the corresponding key in the sequence.

    We report sample-level statistics and per-key maximum-L2 summaries.
    """
    if len(gpt_model.transformer.h) < 1:
        raise ValueError("Model has no transformer blocks.")

    device = next(gpt_model.parameters()).device
    gpt_model.eval()

    # Make the sampling path deterministic per run.
    torch.manual_seed(int(train_seed) + 123456)
    np.random.seed(int(train_seed) + 123456)

    batch_gen = AssociativeRecallBatchGenerator(
        mapping=mapping,
        num_facts=dataset_config.num_facts,
        junk_vocab_size=dataset_config.junk_vocab_size,
        min_seq_length=dataset_config.min_seq_length,
        max_seq_length=dataset_config.max_seq_length,
        batch_size=batch_size,
        num_batches=num_batches,
        device=device,
    )

    num_facts = int(dataset_config.num_facts)
    query_token = num_facts + int(dataset_config.junk_vocab_size)
    wte = gpt_model.transformer.wte.weight.detach()

    block0 = gpt_model.transformer.h[0]
    stage = measurement_stage.strip().lower()
    if stage == "ln2":
        metric_module = block0.ln_2
    elif stage == "lnf":
        metric_module = gpt_model.transformer.ln_f
    else:
        raise ValueError(f"Unsupported measurement_stage='{measurement_stage}'. Use 'ln2' or 'lnf'.")

    cache: Dict[str, torch.Tensor | None] = {"ln1": None, "metric": None}

    def _ln1_hook(_module, _inputs, output):
        cache["ln1"] = output.detach()

    def _metric_hook(_module, _inputs, output):
        cache["metric"] = output.detach()

    handle_ln1 = block0.ln_1.register_forward_hook(_ln1_hook)
    handle_metric = metric_module.register_forward_hook(_metric_hook)

    l2_to_embed: List[float] = []
    l2_to_ln1: List[float] = []
    angle_to_embed: List[float] = []
    per_key_max_embed: Dict[int, float] = {}
    per_key_max_ln1: Dict[int, float] = {}
    nonfinite_sample_count = 0

    try:
        for _ in range(num_batches):
            inputs, targets = batch_gen._generate_batch()
            cache["ln1"] = None
            cache["metric"] = None

            _ = gpt_model(inputs)

            ln1_out = cache["ln1"]
            metric_out = cache["metric"]
            if ln1_out is None or metric_out is None:
                continue

            batch_n = inputs.shape[0]
            for b in range(batch_n):
                pred_positions = torch.where(targets[b] != -100)[0]
                if pred_positions.numel() == 0:
                    continue
                q_pos = int(pred_positions[0].item())
                if int(inputs[b, q_pos].item()) != query_token:
                    continue

                key_positions = torch.where(inputs[b, :q_pos] < num_facts)[0]
                if key_positions.numel() == 0:
                    continue
                key_pos = int(key_positions[0].item())
                key_token = int(inputs[b, key_pos].item())

                query_vec = metric_out[b, q_pos]
                key_embed = wte[key_token]
                key_ln1 = ln1_out[b, key_pos]

                if not (
                    bool(torch.isfinite(query_vec).all())
                    and bool(torch.isfinite(key_embed).all())
                    and bool(torch.isfinite(key_ln1).all())
                ):
                    nonfinite_sample_count += 1
                    continue

                l2_e = torch.norm(query_vec - key_embed, p=2).item()
                l2_k = torch.norm(query_vec - key_ln1, p=2).item()
                ang_e = _angular_distance(query_vec, key_embed)

                if not (
                    np.isfinite(l2_e)
                    and np.isfinite(l2_k)
                    and np.isfinite(ang_e)
                ):
                    nonfinite_sample_count += 1
                    continue

                l2_to_embed.append(l2_e)
                l2_to_ln1.append(l2_k)
                angle_to_embed.append(ang_e)

                prev_e = per_key_max_embed.get(key_token, float("-inf"))
                per_key_max_embed[key_token] = max(prev_e, l2_e)
                prev_k = per_key_max_ln1.get(key_token, float("-inf"))
                per_key_max_ln1[key_token] = max(prev_k, l2_k)
    finally:
        handle_ln1.remove()
        handle_metric.remove()

    embed_stats = _distribution_stats(l2_to_embed)
    ln1_stats = _distribution_stats(l2_to_ln1)
    angle_stats = _distribution_stats(angle_to_embed)

    per_key_embed_items = sorted(per_key_max_embed.items())
    per_key_embed_keys = [int(k) for k, _ in per_key_embed_items]
    per_key_embed_vals = np.asarray([float(v) for _, v in per_key_embed_items], dtype=np.float64)
    per_key_ln1_vals = np.asarray(list(per_key_max_ln1.values()), dtype=np.float64)

    return {
        "attn_noise_l2_mean": embed_stats["mean"],
        "attn_noise_l2_std": embed_stats["std"],
        "attn_noise_l2_max": embed_stats["max"],
        "attn_noise_l2_p95": embed_stats["p95"],
        "attn_noise_l2_floor": float(np.max(per_key_embed_vals)) if per_key_embed_vals.size else float("nan"),
        "attn_noise_l2_per_key_max_mean": (
            float(np.mean(per_key_embed_vals)) if per_key_embed_vals.size else float("nan")
        ),
        "attn_noise_l2_per_key_max_key_ids": per_key_embed_keys,
        "attn_noise_l2_per_key_max_values": per_key_embed_vals.tolist(),
        "attn_noise_l2_to_ln1_mean": ln1_stats["mean"],
        "attn_noise_l2_to_ln1_floor": (
            float(np.max(per_key_ln1_vals)) if per_key_ln1_vals.size else float("nan")
        ),
        "attn_noise_angle_mean": angle_stats["mean"],
        "attn_noise_angle_max": angle_stats["max"],
        "attn_noise_samples": float(len(l2_to_embed)),
        "attn_noise_unique_keys": float(len(per_key_max_embed)),
        "attn_noise_nonfinite_samples_skipped": float(nonfinite_sample_count),
        "attn_noise_measurement_stage": stage,
    }


_SEED_METRICS = [
    "best_acc",
    "final_eval_accuracy",
    "final_train_accuracy",
    "mlp_gamma_min",
    "attn_noise_l2_mean",
    "attn_noise_l2_max",
    "attn_noise_l2_floor",
    "attn_noise_l2_to_ln1_mean",
    "attn_noise_l2_to_ln1_floor",
    "attn_noise_angle_mean",
    "attn_noise_angle_max",
    "attn_noise_nonfinite_samples_skipped",
]


def aggregate_seed_metrics(
    results: List[GPUJobResult],
) -> Tuple[
    GPUJobResult | None,
    Dict[str, float],
    Dict[str, List[float]],
    Dict[str, List[List[float]]],
]:
    """Aggregate scalar metrics across seeds and return the best-seed result."""
    successful = [r for r in results if r.success and isinstance(r.result, dict)]
    if not successful:
        return None, {"error": "No successful results"}, {}, {}

    seed_values: Dict[str, List[float]] = {}
    seed_list_values: Dict[str, List[List[float]]] = {}
    summary: Dict[str, float] = {}

    for metric in _SEED_METRICS:
        vals = []
        for r in successful:
            value = r.result.get(metric, float("nan"))
            if _is_finite_number(value):
                vals.append(float(value))
        if not vals:
            continue

        arr = np.asarray(vals, dtype=np.float64)
        seed_values[metric] = vals
        summary[f"{metric}_mean"] = float(np.mean(arr))
        summary[f"{metric}_std"] = float(np.std(arr))
        summary[f"{metric}_min"] = float(np.min(arr))
        summary[f"{metric}_max"] = float(np.max(arr))

    per_key_lists: List[List[float]] = []
    for r in successful:
        raw_vals = r.result.get("attn_noise_l2_per_key_max_values", [])
        if not isinstance(raw_vals, list):
            continue
        cleaned = [float(v) for v in raw_vals if _is_finite_number(v)]
        if cleaned:
            per_key_lists.append(cleaned)
    if per_key_lists:
        seed_list_values["attn_noise_l2_per_key_max_values"] = per_key_lists

    best_scores = []
    for r in successful:
        score = r.result.get("best_acc", float("-inf"))
        best_scores.append(float(score) if _is_finite_number(score) else float("-inf"))

    best_idx = int(np.argmax(best_scores))
    best_result = successful[best_idx]
    summary["n_successful_seeds"] = float(len(successful))

    return best_result, summary, seed_values, seed_list_values
