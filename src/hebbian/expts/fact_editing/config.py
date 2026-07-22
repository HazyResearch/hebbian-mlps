"""Configs for native fact-editing experiments."""

from __future__ import annotations

import os
from dataclasses import field

import torch
from hebbian.config import pydraclass

from hebbian.data.language import default_author_facts_csv_path
from hebbian.transformer.config import TrainingConfig


def default_fact_editing_train_config() -> TrainingConfig:
    config = TrainingConfig()
    config.transformer_config.use_moe = True
    config.transformer_config.use_mlp_qk = True
    config.transformer_config.bias = True
    return config


@pydraclass
class BaseTrainConfig:
    authors_csv_path: str = field(default_factory=default_author_facts_csv_path)
    tokenizer_name: str = "EleutherAI/pythia-70m"
    experiment_dir: str = "./artifacts/fact_editing/base_model"
    mlp_variant_label: str | None = None
    num_facts: int = 256
    num_rephrases: int = 4
    train_last_token: bool = False
    # "normalized_token" uses the normalized atomic book-token embedding as
    # the fact MLP key while leaving the transformer token row unchanged.
    fact_input_embedding_mode: str = "token"  # "token" | "normalized_token" | "average_compound_normalized"
    overwrite_compound_token_embeddings: bool = True
    eval_batch_size: int | None = None
    train_num_workers: int = 0
    eval_num_workers: int | None = None
    train_config: TrainingConfig = field(default_factory=default_fact_editing_train_config)

    def custom_finalize(self):
        if self.num_facts <= 0:
            raise ValueError("num_facts must be positive.")
        if self.num_rephrases <= 0:
            raise ValueError("num_rephrases must be positive.")
        valid_fact_input_modes = {"token", "normalized_token", "average_compound_normalized"}
        if self.fact_input_embedding_mode not in valid_fact_input_modes:
            raise ValueError(
                "Unknown fact_input_embedding_mode "
                f"{self.fact_input_embedding_mode!r}. Expected one of {sorted(valid_fact_input_modes)}."
            )
        if self.eval_batch_size is None:
            self.eval_batch_size = self.train_config.batch_size
        if self.train_num_workers < 0:
            raise ValueError("train_num_workers must be non-negative.")
        if self.eval_num_workers is None:
            self.eval_num_workers = self.train_num_workers
        if self.eval_num_workers < 0:
            raise ValueError("eval_num_workers must be non-negative.")
        if self.train_config.base_dir is None:
            self.train_config.base_dir = self.experiment_dir
        self.train_config.custom_finalize()


@pydraclass
class EditExperimentConfig:
    experiment_dir: str = "./artifacts/fact_editing/base_model"
    type: str = "memit"  # "memit" | "alpha_edit" | "rome" | "gd_construction"
    num_preserve_facts: int = 128
    num_alter_facts: int = 128
    device: str = "cuda"
    out_dir: str = "./artifacts/fact_editing/results"

    num_steps: int = 100
    lr: float = 0.01
    lambd: float = 1.5e4
    early_stopping: float | None = None
    clip_norm: float | None = None
    wd: float = 0.0
    tol: float = 1e-2
    seed: int = 42
    gd_replacement_variant_label: str | None = None
    gd_replacement_method: str | None = None
    gd_replacement_hidden_dim: int | None = None
    gd_replacement_method_kwargs: dict | None = None
    gd_replacement_dtype: torch.dtype | None = None
    compute_non_fact_ppl: bool = True

    def custom_finalize(self):
        valid_types = {"memit", "alpha_edit", "rome", "gd_construction"}
        if self.type not in valid_types:
            raise ValueError(f"Unknown edit type {self.type!r}. Expected one of {sorted(valid_types)}.")
        if self.num_preserve_facts < 0 or self.num_alter_facts <= 0:
            raise ValueError("num_preserve_facts must be non-negative and num_alter_facts must be positive.")


@pydraclass
class SweepConfig:
    experiment_dir: str = "./artifacts/fact_editing/base_model"
    out_dir: str = "./artifacts/fact_editing/results"
    devices: str = "cuda"
    methods: str = "memit,alpha_edit,rome,gd_construction"
    preserve_counts: str = "128"
    alter_counts: str = "128"
    memit_steps: str = "10,25,100"
    memit_lrs: str = "0.5,0.05,0.005"
    memit_lambdas: str = "15000,1500,150,1"
    memit_clip_norms: str = "1,0.75,0.5"
    alpha_steps: str = "10,25,100"
    alpha_lrs: str = "0.5,0.05,0.005"
    alpha_clip_norms: str = "0.75,0.25,None"
    alpha_tols: str = "0.01,1,10"
    rome_steps: str = "10,25,100"
    rome_lrs: str = "0.5,0.05,0.005"
    rome_wds: str = "0.0015,0.00015,0"
    rome_early_stopping: str = "0.05,None"
    print_only: bool = False


@pydraclass
class SummaryConfig:
    directory: str = "./artifacts/fact_editing/results"
    output_csv: str | None = None

    def custom_finalize(self):
        if self.output_csv is None:
            self.output_csv = os.path.join(self.directory, "best_results.csv")


def dtype_to_name(dtype: torch.dtype | None) -> str | None:
    if dtype is None:
        return None
    return str(dtype).replace("torch.", "")
