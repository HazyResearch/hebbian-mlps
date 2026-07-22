"""Shared helpers for native fact-editing experiments."""

from __future__ import annotations

import json
import os
import random
import copy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from hebbian.core.registry import Registry
from hebbian.data.language.authors import (
    AuthorFact,
    author_facts_from_dicts,
    build_author_example_groups,
)
from hebbian.data.synthetics.factsets import BijectiveMapping, Factset
from hebbian.methods.hebbian.model import HebbianMLP
from hebbian.transformer.model import BinaryMoE, GPT, GPTConfig
from hebbian.transformer.utils import insert_mlp_into_gpt

import hebbian.methods  # noqa: F401


DTYPE_NAME_TO_TORCH = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


@dataclass
class LoadedExperiment:
    experiment_dir: str
    tokenizer: Any
    facts: list[AuthorFact]
    fact_groups: list[list[dict[str, Any]]]
    gpt_model: GPT
    gpt_config: GPTConfig
    full_input_embeddings: nn.Embedding
    full_output_embeddings: nn.Embedding
    factset: Factset
    checkpoint: Dict[str, Any]
    metadata: Dict[str, Any]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_dtype_to_name(dtype: torch.dtype | None) -> str | None:
    if dtype is None:
        return None
    return str(dtype).replace("torch.", "")


def parse_torch_dtype(value: str | None) -> torch.dtype | None:
    if value is None:
        return None
    if value not in DTYPE_NAME_TO_TORCH:
        raise ValueError(f"Unsupported torch dtype name: {value}")
    return DTYPE_NAME_TO_TORCH[value]


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, torch.dtype):
        return torch_dtype_to_name(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def make_embedding_module(weight: torch.Tensor) -> nn.Embedding:
    embedding = nn.Embedding(weight.shape[0], weight.shape[1], _freeze=False)
    embedding.weight.data.copy_(weight)
    return embedding


def _restore_loaded_gpt_embeddings(
    gpt_model: GPT,
    *,
    full_input_embeddings: nn.Embedding,
    full_output_embeddings: nn.Embedding,
) -> None:
    input_weight = full_input_embeddings.weight.detach().clone().contiguous()
    input_param = nn.Parameter(input_weight, requires_grad=False)
    gpt_model.transformer.wte.weight = input_param

    if gpt_model.config.tie_embeddings:
        gpt_model.lm_head.weight = input_param
    else:
        output_weight = full_output_embeddings.weight.detach().clone().contiguous()
        output_param = nn.Parameter(output_weight, requires_grad=False)
        gpt_model.lm_head.weight = output_param


def _make_method_config(hidden_dim: int | None) -> Dict[str, Any] | None:
    if hidden_dim is None:
        return None
    return {"m": hidden_dim}


def construct_mlp(
    *,
    factset: Factset,
    method_name: str,
    hidden_dim: int | None,
    seed: int,
    device: str,
    method_kwargs: Dict[str, Any] | None = None,
) -> tuple[nn.Module, Dict[str, Any]]:
    factset = factset.to(device=device)
    method_cls = Registry.get_method(method_name)
    method = method_cls()
    method_config = _make_method_config(hidden_dim)
    if method_kwargs:
        method_config = {**(method_config or {}), **method_kwargs}
    method.initialize(config=method_config, seed=seed)
    for attr in vars(method).values():
        if hasattr(attr, "shared") and hasattr(attr.shared, "device"):
            attr.shared.device = device
    mlp, metrics = method.fit_or_construct(factset)
    return mlp, metrics


def unwrap_edit_mlp(mlp: nn.Module) -> nn.Module:
    return mlp


def get_edit_mlp(gpt_model: nn.Module) -> nn.Module:
    block_mlp = gpt_model.transformer.h[0].mlp
    if isinstance(block_mlp, BinaryMoE):
        return block_mlp.fact_expert
    return block_mlp


def set_edit_mlp(gpt_model: nn.Module, mlp: nn.Module) -> None:
    block = gpt_model.transformer.h[0]
    if isinstance(block.mlp, BinaryMoE):
        block.mlp.fact_expert = mlp
    else:
        block.mlp = mlp


def get_mlp_down_module(mlp: nn.Module) -> nn.Module:
    core_mlp = unwrap_edit_mlp(mlp)
    if not hasattr(core_mlp, "down"):
        raise ValueError(f"Expected MLP with `.down`, got {type(core_mlp)}")
    return core_mlp.down


def get_mlp_down_weight(mlp: nn.Module) -> torch.Tensor:
    down = get_mlp_down_module(mlp)
    if not hasattr(down, "linear"):
        raise ValueError(f"Expected `.down.linear`, got {type(down)}")
    return down.linear.weight


def _hebbian_feature_keys(mlp: HebbianMLP, x: torch.Tensor) -> torch.Tensor:
    """Return the local-edit key space used by a constructed HebbianMLP."""
    if x.dim() == 3:
        batch, seq_len, dim = x.shape
        x = x.reshape(batch * seq_len, dim)
        features = mlp.feature_map(x)
        return features.reshape(batch, seq_len, -1)
    return mlp.feature_map(x)


def get_mlp_edit_module(mlp: nn.Module) -> nn.Module:
    core_mlp = unwrap_edit_mlp(mlp)
    if hasattr(core_mlp, "down"):
        return get_mlp_down_module(core_mlp)
    if isinstance(core_mlp, HebbianMLP):
        return core_mlp
    raise ValueError(
        "Expected editable MLP with `.down` or HebbianMLP with `W`, "
        f"got {type(core_mlp)}"
    )


def get_mlp_edit_weight(mlp: nn.Module) -> torch.Tensor:
    core_mlp = unwrap_edit_mlp(mlp)
    if hasattr(core_mlp, "down"):
        return get_mlp_down_weight(core_mlp)
    if isinstance(core_mlp, HebbianMLP):
        return core_mlp.W
    raise ValueError(
        "Expected editable MLP with `.down.linear.weight` or HebbianMLP `W`, "
        f"got {type(core_mlp)}"
    )


def get_mlp_edit_key_from_hook_input(mlp: nn.Module, hook_input: torch.Tensor) -> torch.Tensor:
    core_mlp = unwrap_edit_mlp(mlp)
    if hasattr(core_mlp, "down"):
        return hook_input
    if isinstance(core_mlp, HebbianMLP):
        return _hebbian_feature_keys(core_mlp, hook_input)
    raise ValueError(
        "Expected editable MLP with `.down` or HebbianMLP with `feature_map`, "
        f"got {type(core_mlp)}"
    )


def project_mlp_edit_residuals(
    mlp: nn.Module,
    residuals: list[torch.Tensor],
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Map final-space residuals into the editable matrix output space."""
    return [residual.to(device=device, dtype=dtype) for residual in residuals]


def save_base_artifacts(
    *,
    experiment_dir: str,
    checkpoint_payload: Dict[str, Any],
    embeddings_payload: Dict[str, Any],
    metadata_payload: Dict[str, Any],
) -> None:
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(checkpoint_payload, os.path.join(checkpoint_dir, "last_model.pt"))
    torch.save(embeddings_payload, os.path.join(checkpoint_dir, "embeddings.pt"))
    save_json(os.path.join(experiment_dir, "fact_editing_metadata.json"), metadata_payload)


def load_base_experiment(experiment_dir: str, device: str) -> LoadedExperiment:
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, "last_model.pt"),
        map_location="cpu",
        weights_only=False,
    )
    embeddings = torch.load(
        os.path.join(checkpoint_dir, "embeddings.pt"),
        map_location="cpu",
        weights_only=False,
    )
    with open(
        os.path.join(experiment_dir, "fact_editing_metadata.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer_name"])
    tokenizer.add_tokens(metadata["added_tokens"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    facts = author_facts_from_dicts(metadata["facts"])
    fact_groups = build_author_example_groups(
        tokenizer,
        facts,
        num_rephrases=metadata["num_rephrases"],
        train_last_token=metadata["train_last_token"],
    )

    full_input_embeddings = make_embedding_module(embeddings["full_input_embeddings"])
    full_output_embeddings = make_embedding_module(embeddings["full_output_embeddings"])
    mapping_outputs = embeddings.get("mapping_outputs")
    factset = Factset(
        input_embeddings=embeddings["fact_input_embeddings"].to(device=device),
        output_embeddings=embeddings["fact_output_embeddings"].to(device=device),
        mapping=BijectiveMapping(
            inputs=list(range(len(mapping_outputs))),
            outputs=list(mapping_outputs),
        ),
        d_model=embeddings["fact_input_embeddings"].shape[1],
        vocab_size=embeddings["fact_input_embeddings"].shape[0],
    )
    fact_input_embeddings_for_norm = make_embedding_module(embeddings["fact_input_embeddings"])
    gpt_config = GPTConfig(**checkpoint["gpt_config"])
    gpt_model = GPT(gpt_config)
    if "mlp_module" in checkpoint:
        gpt_model.eval()
        insert_mlp_into_gpt(
            gpt_model,
            copy.deepcopy(checkpoint["mlp_module"]),
            fact_input_embeddings_for_norm,
            freeze_mlp=True,
            freeze_wte=True,
            freeze_lm_head=True,
        )
    gpt_model.load_state_dict(checkpoint["gpt_state_dict"])
    model_dtype = parse_torch_dtype(checkpoint["train_config"]["dtype"])
    gpt_model.to(device=device, dtype=model_dtype)
    full_input_embeddings.to(device=device, dtype=model_dtype)
    full_output_embeddings.to(device=device, dtype=model_dtype)
    _restore_loaded_gpt_embeddings(
        gpt_model,
        full_input_embeddings=full_input_embeddings,
        full_output_embeddings=full_output_embeddings,
    )

    if len(tokenizer) != gpt_config.vocab_size:
        raise ValueError(
            f"Reloaded tokenizer length {len(tokenizer)} does not match saved GPT vocab size {gpt_config.vocab_size}."
        )

    return LoadedExperiment(
        experiment_dir=experiment_dir,
        tokenizer=tokenizer,
        facts=facts,
        fact_groups=fact_groups,
        gpt_model=gpt_model,
        gpt_config=gpt_config,
        full_input_embeddings=full_input_embeddings,
        full_output_embeddings=full_output_embeddings,
        factset=factset,
        checkpoint=checkpoint,
        metadata=metadata,
    )
