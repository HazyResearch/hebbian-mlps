"""Run a fact-editing experiment against a saved native base model."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import torch
from hebbian.config import main as main_decorator

from hebbian.expts.fact_editing.common import load_base_experiment, parse_torch_dtype
from hebbian.expts.fact_editing.config import EditExperimentConfig
from hebbian.expts.fact_editing.methods import (
    EditMetrics,
    edit_factset_alpha_edit,
    edit_factset_gd_construction,
    edit_factset_memit,
    edit_factset_rome,
)


def _metrics_to_dict(metrics: EditMetrics) -> Dict[str, float]:
    return {
        "efficacy": metrics.efficacy,
        "paraphrase": metrics.paraphrase,
        "specificity": metrics.specificity,
        "specificity_paraphrase": metrics.specificity_paraphrase,
        "non_fact_pre_nll": metrics.non_fact_pre_nll,
        "non_fact_post_nll": metrics.non_fact_post_nll,
        "non_fact_pre_ppl": metrics.non_fact_pre_ppl,
        "non_fact_post_ppl": metrics.non_fact_post_ppl,
        "non_fact_ppl_ratio": metrics.non_fact_ppl_ratio,
        "non_fact_num_tokens": metrics.non_fact_num_tokens,
    }


def _prepare_runtime_device(device: str) -> str:
    if not device.startswith("cuda"):
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device} but CUDA is not available.")
    resolved = torch.device(device)
    if resolved.index is None:
        resolved = torch.device("cuda:0")
    torch.cuda.set_device(resolved)
    torch.cuda.init()
    # Force early context creation on the selected device instead of failing later
    # inside the first model forward.
    torch.empty(1, device=resolved)
    return str(resolved)


def _save_results(
    config: EditExperimentConfig,
    metrics: EditMetrics,
    extra_config: Dict[str, Any] | None = None,
) -> str:
    output_dir = os.path.join(config.out_dir, config.type)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{time.time_ns()}.json")
    config_payload = {
        "experiment_dir": config.experiment_dir,
        "type": config.type,
        "num_preserve_facts": config.num_preserve_facts,
        "num_alter_facts": config.num_alter_facts,
        "device": config.device,
        "out_dir": config.out_dir,
        "num_steps": config.num_steps,
        "lr": config.lr,
        "lambd": config.lambd,
        "early_stopping": config.early_stopping,
        "clip_norm": config.clip_norm,
        "wd": config.wd,
        "tol": config.tol,
        "seed": config.seed,
        "gd_replacement_variant_label": config.gd_replacement_variant_label,
        "gd_replacement_method": config.gd_replacement_method,
        "gd_replacement_hidden_dim": config.gd_replacement_hidden_dim,
        "gd_replacement_method_kwargs": config.gd_replacement_method_kwargs,
        "gd_replacement_dtype": (
            None if config.gd_replacement_dtype is None else str(config.gd_replacement_dtype).replace("torch.", "")
        ),
        "compute_non_fact_ppl": config.compute_non_fact_ppl,
    }
    if extra_config:
        config_payload.update(extra_config)
    payload = {
        **_metrics_to_dict(metrics),
        "config": config_payload,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def run(config: EditExperimentConfig) -> Dict[str, Any]:
    config.finalize()
    requested_device = "cuda:0" if config.device == "cuda" else config.device
    device = _prepare_runtime_device(requested_device)
    loaded = load_base_experiment(config.experiment_dir, device)
    if loaded.metadata.get("n_layers") != 1:
        raise ValueError(
            "The native fact-editing port currently expects a single transformer layer "
            "so that the inserted MLP is not ambiguously shared across blocks."
        )
    if config.num_preserve_facts + config.num_alter_facts > len(loaded.facts):
        raise ValueError(
            f"Requested {config.num_preserve_facts + config.num_alter_facts} facts but artifact only has {len(loaded.facts)}."
        )

    if config.type == "memit":
        metrics = edit_factset_memit(
            loaded.gpt_model,
            loaded.facts,
            loaded.fact_groups,
            loaded.factset,
            num_preserve_facts=config.num_preserve_facts,
            num_alter_facts=config.num_alter_facts,
            num_steps=config.num_steps,
            lr=config.lr,
            lambd=config.lambd,
            device=device,
            clip_norm=config.clip_norm,
            early_stopping=config.early_stopping,
            seed=config.seed,
            compute_non_fact_ppl=config.compute_non_fact_ppl,
        )
    elif config.type == "alpha_edit":
        metrics = edit_factset_alpha_edit(
            loaded.gpt_model,
            loaded.facts,
            loaded.fact_groups,
            loaded.factset,
            num_preserve_facts=config.num_preserve_facts,
            num_alter_facts=config.num_alter_facts,
            num_steps=config.num_steps,
            lr=config.lr,
            device=device,
            clip_norm=config.clip_norm,
            early_stopping=config.early_stopping,
            tol=config.tol,
            seed=config.seed,
            compute_non_fact_ppl=config.compute_non_fact_ppl,
        )
    elif config.type == "rome":
        metrics = edit_factset_rome(
            loaded.gpt_model,
            loaded.facts,
            loaded.fact_groups,
            loaded.factset,
            num_preserve_facts=config.num_preserve_facts,
            num_alter_facts=config.num_alter_facts,
            num_steps=config.num_steps,
            lr=config.lr,
            device=device,
            early_stopping=config.early_stopping,
            wd=config.wd,
            seed=config.seed,
            compute_non_fact_ppl=config.compute_non_fact_ppl,
        )
    elif config.type == "gd_construction":
        has_replacement_override = any(
            value is not None
            for value in (
                config.gd_replacement_variant_label,
                config.gd_replacement_method,
                config.gd_replacement_hidden_dim,
                config.gd_replacement_method_kwargs,
                config.gd_replacement_dtype,
            )
        )
        replacement_method = config.gd_replacement_method or loaded.metadata["mlp_method"]
        replacement_variant = (
            config.gd_replacement_variant_label
            or loaded.metadata.get("mlp_variant_label")
            or replacement_method
        )
        replacement_hidden_dim = (
            config.gd_replacement_hidden_dim
            if config.gd_replacement_hidden_dim is not None
            else loaded.metadata["mlp_hidden_dim"]
        )
        if has_replacement_override:
            replacement_method_kwargs = config.gd_replacement_method_kwargs
        else:
            replacement_method_kwargs = loaded.metadata.get("mlp_method_kwargs")
        replacement_dtype = (
            config.gd_replacement_dtype
            if config.gd_replacement_dtype is not None
            else parse_torch_dtype(loaded.metadata.get("mlp_dtype"))
        )
        metrics = edit_factset_gd_construction(
            loaded.gpt_model,
            loaded.facts,
            loaded.fact_groups,
            loaded.factset,
            loaded.full_input_embeddings,
            num_preserve_facts=config.num_preserve_facts,
            num_alter_facts=config.num_alter_facts,
            method_name=replacement_method,
            hidden_dim=replacement_hidden_dim,
            seed=config.seed,
            device=device,
            method_kwargs=replacement_method_kwargs,
            factset_dtype=replacement_dtype,
            compute_non_fact_ppl=config.compute_non_fact_ppl,
        )
    else:
        raise NotImplementedError(f"Unsupported edit type {config.type!r}")

    extra_config = {
        "base_mlp_variant": loaded.metadata.get("mlp_variant_label"),
        "base_mlp_method": loaded.metadata.get("mlp_method"),
        "base_mlp_method_kwargs": loaded.metadata.get("mlp_method_kwargs"),
        "base_num_facts": len(loaded.facts),
        "base_num_rephrases": loaded.metadata.get("num_rephrases"),
    }
    if config.type == "gd_construction":
        extra_config.update(
            {
                "gd_replacement_variant": replacement_variant,
                "gd_replacement_method": replacement_method,
                "gd_replacement_hidden_dim": replacement_hidden_dim,
                "gd_replacement_method_kwargs": replacement_method_kwargs,
                "gd_replacement_dtype": (
                    None if replacement_dtype is None else str(replacement_dtype).replace("torch.", "")
                ),
            }
        )

    output_path = _save_results(
        config,
        metrics,
        extra_config=extra_config,
    )
    result = _metrics_to_dict(metrics)
    result["output_path"] = output_path
    return result


@main_decorator(EditExperimentConfig)
def main(config: EditExperimentConfig):
    run(config)


if __name__ == "__main__":
    main()
