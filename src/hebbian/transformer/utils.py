"""GPT2 utilities for associative-recall experiments."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hebbian.transformer.model import BinaryMoE, GPT, GPTConfig
from hebbian.transformer.norms import FrozenRMSNorm


def compute_weight_norms(
    model: nn.Module,
    trainable_only: bool = True,
) -> Dict[str, float]:
    """Compute the L2 norm of each weight tensor in the model.
    
    Args:
        model: The model to compute weight norms for.
        trainable_only: If True, only include trainable parameters.
    
    Returns:
        Dict mapping parameter names to their L2 norms.
        Also includes 'total_norm' (sum of all norms) and 
        'total_norm_squared' (sum of squared norms).
    
    Example:
        >>> norms = compute_weight_norms(model)
        >>> print(norms['transformer.h.0.attn.c_q.weight'])  # individual
        >>> print(norms['total_norm'])  # total
    """
    norms = {}
    total_norm_sq = 0.0
    
    for name, p in model.named_parameters():
        if trainable_only and not p.requires_grad:
            continue
        norm = p.data.norm(2).item()
        norms[name] = norm
        total_norm_sq += norm ** 2
    
    norms["total_norm"] = total_norm_sq ** 0.5
    norms["total_norm_squared"] = total_norm_sq
    return norms


def insert_mlp_into_gpt(
    model: GPT,
    mlp: nn.Module,
    mlp_embeddings: nn.Embedding,
    freeze_mlp: bool = True,
    freeze_wte: bool = True,
    freeze_lm_head: bool = True,
):
    """Replace the MLP in all transformer blocks with a custom MLP.

    Args:
        model: GPT model (must be in eval mode).
        mlp: Custom MLP module to insert.
        mlp_embeddings: Embeddings used by the MLP (for FrozenRMSNorm scaling).
        freeze_mlp: Whether to freeze the MLP weights.
        freeze_wte: Whether to freeze token embeddings.
        freeze_lm_head: Whether to freeze the LM head.
    """
    assert not model.training, "Model must be in eval mode before MLP insertion"

    for block in model.transformer.h:
        if isinstance(block.mlp, BinaryMoE):
            block.mlp.fact_expert = mlp
        else:
            block.mlp = mlp
        for param in mlp.parameters():
            param.requires_grad = not freeze_mlp

    for param in model.transformer.wte.parameters():
        param.requires_grad = not freeze_wte
    for param in model.lm_head.parameters():
        param.requires_grad = not freeze_lm_head

    model.get_frozen_rms_scale(mlp_embeddings)


def copy_embeddings_to_gpt(
    gpt_model: GPT,
    input_embeddings: nn.Embedding,
    output_embeddings: nn.Embedding | None = None,
    freeze_wte: bool = True,
    freeze_lm_head: bool = True,
    tie_embeddings: bool = True,
):
    """Copy embeddings to the GPT model's token embedding and LM head.

    Args:
        gpt_model: GPT model.
        input_embeddings: Input embeddings to copy to wte.
        output_embeddings: Output embeddings to copy to lm_head (only if not tied).
        freeze_wte: Whether to freeze token embeddings after copying.
        freeze_lm_head: Whether to freeze LM head after copying.
        tie_embeddings: Whether input and output embeddings are tied.
    """
    assert gpt_model.transformer.wte.weight.data.shape == input_embeddings.weight.data.shape, (
        f"Shape mismatch: GPT wte {gpt_model.transformer.wte.weight.data.shape} "
        f"vs input {input_embeddings.weight.data.shape}"
    )
    gpt_model.transformer.wte.weight.data.copy_(input_embeddings.weight.data)

    if tie_embeddings:
        # wte and lm_head share weights, so copying wte is enough
        pass
    else:
        assert output_embeddings is not None
        assert gpt_model.lm_head.weight.data.shape == output_embeddings.weight.data.shape
        gpt_model.lm_head.weight.data.copy_(output_embeddings.weight.data)

    if freeze_wte:
        gpt_model.transformer.wte.weight.requires_grad = False
    if freeze_lm_head:
        gpt_model.lm_head.weight.requires_grad = False


def create_gpt_config(train_config, dataset_config) -> GPTConfig:
    """Create GPTConfig from training and dataset configs.

    Handles rope_base scaling via match_high_freq method.
    """
    tc = train_config.transformer_config
    ec = train_config.embeddings_config
    d_model = ec.d_model

    # Fact editing supplies its tokenized context length directly. Paper
    # Transformer experiments use the associative-recall sequence layout.
    block_size = getattr(dataset_config, "block_size", None)
    if block_size is None:
        block_size = 2 * dataset_config.max_seq_length + 3

    # RoPE base scaling
    rope_base = tc.rope_base
    if tc.rope_scale_method == "match_high_freq":
        rope_base = tc.rope_base ** (d_model / 1024)

    return GPTConfig(
        block_size=block_size,
        vocab_size=dataset_config.vocab_size,
        n_layer=tc.n_layers,
        n_head=tc.n_head,
        n_embd=d_model,
        dropout=tc.dropout,
        bias=tc.bias,
        mlp_residual=tc.mlp_residual,
        attn_residual=tc.attn_residual,
        use_rope=tc.use_rope,
        rope_base=rope_base,
        no_positional_encoding=tc.no_positional_encoding,
        mlp_norm_type=tc.mlp_norm_type,
        attn_norm_type=tc.attn_norm_type,
        lm_head_norm_type=tc.lm_head_norm_type,
        tie_embeddings=ec.tie_embeddings,
        freeze_value_dense_identity=tc.freeze_value_dense_identity,
        use_mlp_qk=tc.use_mlp_qk,
        use_identity_mlp=tc.use_identity_mlp,
        use_moe=tc.use_moe,
        moe_router_num_layers=tc.moe_router_num_layers,
        moe_router_intermediate_dim=tc.moe_router_intermediate_dim,
        moe_gate=tc.moe_gate,
        moe_router_use_mlp_input=tc.moe_router_use_mlp_input,
        moe_convex=tc.moe_convex,
        moe_mlp_type=tc.moe_mlp_type,
        moe_mlp_out_norm=tc.moe_mlp_out_norm,
        moe_lora_linear_rank=tc.moe_lora_linear_rank,
    )


def evaluate(
    gpt_model: GPT,
    dataloader: DataLoader,
    device: torch.device,
    num_iterations: int = 1,
) -> Dict[str, float]:
    """Evaluate model accuracy and loss on a dataloader.

    Args:
        gpt_model: GPT model.
        dataloader: Evaluation dataloader.
        device: Device.
        num_iterations: Number of times to iterate over the dataloader.

    Returns:
        Dict with 'loss' and 'accuracy' keys.
    """
    gpt_model.eval()
    total_loss = 0.0
    total_correct = 0
    total_targets = 0
    num_batches = 0

    with torch.no_grad():
        for _ in range(num_iterations):
            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                logits, loss = gpt_model(inputs, targets=targets)
                total_loss += loss.item()

                # Accuracy: only on non-masked positions
                mask = targets != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    total_correct += (preds[mask] == targets[mask]).sum().item()
                    total_targets += mask.sum().item()

                num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_targets, 1)
    return {"loss": avg_loss, "accuracy": accuracy}
