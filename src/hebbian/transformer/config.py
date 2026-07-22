"""
Configuration hierarchy for associative-recall experiments.

Mirrors the mlps repo structure but heavily cleaned up:
- Removed unused knobs (StateTracker, compound keys, etc.)
- Defaults match VanillaModsSynthTrainingConfig from mlps
- Uses @pydraclass for CLI override support
"""

from __future__ import annotations

import os
from dataclasses import field

import torch
from hebbian.config import pydraclass


@pydraclass
class DatasetConfig:
    """Dataset configuration for associative-recall experiments.

    Cleaned-up version of mlps SynthDatasetConfig. Single-token keys/values only.
    """

    num_facts: int = 64
    junk_vocab_size: int = 16
    min_seq_length: int = 8
    max_seq_length: int = 16
    use_identity_fact_mapping: bool = False

    # Auto-computed in custom_finalize. The extra token is the query Q.
    vocab_size: int | None = None

    def custom_finalize(self):
        if self.vocab_size is None:
            self.vocab_size = self.num_facts + self.junk_vocab_size + 1


@pydraclass
class EmbeddingsConfig:
    """Embedding configuration (simplified).

    Removed: prefix_size, ModifierConfigs, input/output_embedding_config,
    mlp_input/output_embedding_modifier.
    """

    d_model: int = 64
    tie_embeddings: bool = True
    embedding_init: str = "spherical"  # "spherical" | "gaussian" | "uniform"

    # LLM-embedding mode: load (x, y) tables from <embeddings_dir>/{x.pt,y.pt}
    # instead of generating synthetic embeddings. When set:
    #   - d_model is overridden from the data,
    #   - tie_embeddings is forced to False (x and y are independent tables),
    #   - embedding_init is ignored for the fact slots,
    #   - junk + Q slots are filled from leftover rows of x.pt (option 3 from
    #     the design notes), so junk shares the LLM activation distribution.
    # Synthetic mode is unchanged when embeddings_dir is None.
    embeddings_dir: str | None = None

    def custom_finalize(self):
        if self.embeddings_dir is None:
            return
        n_rows, d_from_data = _peek_embeddings_shape(self.embeddings_dir)
        if self.tie_embeddings:
            raise ValueError(
                "embeddings_dir requires tie_embeddings=False "
                "(x and y are independent LLM tables)"
            )
        if self.d_model is not None and self.d_model != d_from_data:
            raise ValueError(
                f"d_model={self.d_model} != d_from_data={d_from_data} "
                f"from {self.embeddings_dir!r} (N={n_rows})"
            )
        self.d_model = d_from_data


def _peek_embeddings_shape(embeddings_dir: str) -> tuple[int, int]:
    """Load x.pt to read (N, d). Raises if x.pt/y.pt disagree or aren't 2D."""
    x_path = os.path.join(embeddings_dir, "x.pt")
    y_path = os.path.join(embeddings_dir, "y.pt")
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(
            f"embeddings_dir must contain x.pt and y.pt; got {embeddings_dir!r}"
        )
    x = torch.load(x_path, map_location="cpu")
    y = torch.load(y_path, map_location="cpu")
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError(
            f"x.pt and y.pt must be 2D with matching shape; got "
            f"x={tuple(x.shape)}, y={tuple(y.shape)}"
        )
    return int(x.shape[0]), int(x.shape[1])


@pydraclass
class TransformerConfig:
    """Transformer architecture configuration (cleaned up).

    Defaults match VanillaModsSynthTrainingConfig from mlps.

    Removed: mlp_alpha, conditional_alpha, use_lm_head_alpha, mlp_output_norm,
    input_noise, no_sequence_mixer, normalize_similarity, use_key_mask,
    input_embedding_noise, gpt_uses_mlp_embeddings.
    """

    n_layers: int = 1
    n_head: int = 1

    # Residual connections
    mlp_residual: bool = False
    attn_residual: bool = False

    # Normalization
    mlp_norm_type: str = "frozen_rmsnorm"
    attn_norm_type: str = "rmsnorm"
    lm_head_norm_type: str = "rmsnorm"

    # RoPE and positional encoding
    use_rope: bool = True
    rope_base: float = 10000.0
    rope_scale_method: str = "match_high_freq"
    # If True, disables ALL positional encoding (both RoPE and absolute wpe).
    # Useful for attention-only experiments with position-agnostic attention.
    no_positional_encoding: bool = False

    # Regularization
    dropout: float = 0.0
    bias: bool = False

    # Special init & freezing
    freeze_value_dense_identity: bool = True
    freeze_input_embeddings: bool = True
    freeze_output_embeddings: bool = True
    use_mlp_qk: bool = False

    # Use identity MLP (passes input through unchanged).
    # When True, the MLP block effectively does nothing, isolating attention.
    use_identity_mlp: bool = False

    # Optional old-style BinaryMoE/fact_expert path for fact-editing experiments.
    use_moe: bool = False
    moe_router_num_layers: int = 2
    moe_router_intermediate_dim: int | None = None
    moe_gate: bool = False
    moe_router_use_mlp_input: bool = False
    moe_convex: bool = True
    moe_mlp_type: str = "lora_linear"
    moe_mlp_out_norm: bool = False
    moe_lora_linear_rank: int = 8


@pydraclass
class TrainingConfig:
    """Training configuration (cleaned up).

    Removed: moe_config, test_mlp_config, train_mlp_config (use Method interface),
    use_state_tracker, cleanup_memory, compute_embedding_properties,
    save_mlp_embeddings, reorder_plot_embeddings, verbose,
    construction_encoding/decoding overrides, set_eval_to_train,
    fail_on_failed_mlps, summary_csv_path, loss_scale, plot_embeddings.
    """

    embeddings_config: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)

    # MLP method
    mlp_method: str = "gd"  # "gd" | "ntk" | "hebbian"
    mlp_hidden_dim: int | None = None
    # Override dtype used when generating the factset for MLP construction.
    # Useful for Hebbian/NTK which benefit from float64 precision.
    # None means use the same `dtype` as the transformer.
    # The MLP is always cast back to `dtype` before insertion into the transformer.
    mlp_dtype: torch.dtype | None = None
    # Method-specific config kwargs merged into the method config at construction time.
    # Set programmatically (not via CLI). Examples:
    #   Hebbian: {"variant": "whitened", "ridge": 1e-6}
    #   NTK:     {"hermite_degree": 1}
    mlp_method_kwargs: dict | None = None

    # Eval MLP (optional separate MLP used during evaluation)
    eval_mlp_method: str | None = None
    eval_mlp_hidden_dim: int | None = None
    eval_mlp_dtype: torch.dtype | None = None
    eval_mlp_method_kwargs: dict | None = None

    # Training hyperparameters
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    batch_size: int = 1280
    lr: float = 2e-4
    weight_decay: float = 0.1
    epochs: int = 16000
    steps_per_dataset: int = 1
    seed: int = 42

    # torch.compile
    compile_model: bool = False

    # Attention loss type: "ce" (cross-entropy) or "l2" (MSE on hidden states)
    attention_loss_type: str = "ce"

    # Early stopping
    early_stop_accuracy: float | None = 0.99

    # If set, training short-circuits when the constructed/trained MLP's
    # standalone factset accuracy is below this threshold, returning a
    # failed-result dict (best_acc=best_train_acc=0.0) without running the
    # transformer training loop. Lets the sweep's binary search reject m
    # values at which the MLP itself can't store the facts, preventing
    # attention from finding non-MLP shortcuts in the end-to-end signal.
    mlp_acc_threshold: float | None = None

    # I/O
    base_dir: str | None = None
    save_dir: str | None = None
    figs_dir: str | None = None
    evaluate_every: int = 100
    plot_every: int = 100
    save_at_end: bool = True

    def custom_finalize(self):
        if self.base_dir is not None:
            if self.save_dir is None:
                self.save_dir = os.path.join(self.base_dir, "checkpoints")
            if self.figs_dir is None:
                self.figs_dir = os.path.join(self.base_dir, "figs")


@pydraclass
class AssociativeRecallConfig:
    """Top-level configuration for the associative-recall experiment.

    Mirrors TrainingSynthConfig from mlps but simplified.
    """

    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    train_config: TrainingConfig = field(default_factory=TrainingConfig)
