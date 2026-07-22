"""
Attention-only training pipeline.

This module provides a standalone training loop for attention-only experiments,
without any MLP creation or swapping logic.

**Required config settings for attention-only training:**

1. `transformer_config.freeze_value_dense_identity = False`
   - Unfreezes V and c_proj matrices so attention can be trained

2. `transformer_config.use_identity_mlp = True`
   - Uses identity MLP (passes input unchanged), isolating attention

3. `transformer_config.no_positional_encoding = True`
   - Disables both RoPE and absolute position embeddings

4. `transformer_config.lm_head_norm_type = "unit_rmsnorm"`
   - Normalizes outputs to RMS=1 before lm_head projection

5. `transformer_config.mlp_norm_type` — norm applied between attention and MLP (ln_2).
   Use "none" to disable, "unit_rmsnorm" to normalize, etc.

6. `dataset_config.use_identity_fact_mapping = True` (optional but recommended)
   - Makes associative recall a pure key-copy task

**Task Interpretation with Identity Mapping:**

With identity mapping (f(i) = i), associative recall becomes a key-copy task:
the attention mechanism must retrieve K across intervening junk tokens and
copy its value at the query position Q.

Usage:
    from hebbian.transformer.train_attention import train_attention
    from hebbian.transformer.config import AssociativeRecallConfig

    # train_attention automatically applies attention-only defaults
    results = train_attention(AssociativeRecallConfig())
"""

from __future__ import annotations

import os
from random import triangular
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hebbian.transformer.config import AssociativeRecallConfig
from hebbian.transformer.model import GPT
from hebbian.transformer.utils import (
    create_gpt_config,
    copy_embeddings_to_gpt,
    compute_weight_norms,
    evaluate,
)
from hebbian.transformer.data import create_associative_recall_batches
from hebbian.transformer.fact_store import (
    build_factset,
    build_token_embeddings,
)
from hebbian.transformer.plotting import plot_training_progress


def validate_attention_config(config: AssociativeRecallConfig) -> List[str]:
    """Validate that config is properly set up for attention-only training.
    
    Args:
        config: AssociativeRecallConfig to validate.
        
    Returns:
        List of error messages. Empty if config is valid.
    """
    errors = []
    tc = config.train_config.transformer_config
    train_config = config.train_config
    
    # Must unfreeze attention (V and c_proj)
    if tc.freeze_value_dense_identity:
        errors.append(
            "transformer_config.freeze_value_dense_identity must be False "
            "(got True). Attention matrices must be trainable."
        )
    
    # Must use identity MLP
    if not tc.use_identity_mlp:
        errors.append(
            "transformer_config.use_identity_mlp must be True "
            "(got False). MLP must be identity for attention-only training."
        )
    
    # Must disable positional encoding
    if not tc.no_positional_encoding:
        errors.append(
            "transformer_config.no_positional_encoding must be True "
            "(got False). Positional encoding must be disabled."
        )
    
    # Must use unit_rmsnorm for lm_head
    if tc.lm_head_norm_type != "unit_rmsnorm":
        errors.append(
            f"transformer_config.lm_head_norm_type must be 'unit_rmsnorm' "
            f"(got '{tc.lm_head_norm_type}'). Outputs must be normalized to 1."
        )
    
    # Must not have eval_mlp_method (no MLP in attention-only training)
    if train_config.eval_mlp_method is not None:
        errors.append(
            f"train_config.eval_mlp_method must be None "
            f"(got '{train_config.eval_mlp_method}'). No MLP is used in attention-only training."
        )
    
    # ln_2 can be "none" (Identity) or any norm type

    if errors:
        raise ValueError(
            "Invalid config for attention-only training:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    else:
        print("Config is valid for attention-only training.")


def train_attention(config: AssociativeRecallConfig) -> Dict[str, Any]:
    """Train a transformer with attention-only (MLP disabled).
    
    This is a standalone training loop for attention-only experiments.
    Unlike full associative-recall training, this does not create or swap an MLP.
    uses an identity MLP built into the architecture.
    
    Automatically applies attention-only config defaults and validates.
    
    Args:
        config: AssociativeRecallConfig instance.
    
    Returns:
        Dict with training results including metrics and final accuracy.
        
    Raises:
        ValueError: If config validation fails after applying defaults.
    """
    # Validate config
    validate_attention_config(config)
    dc = config.dataset_config
    tc = config.train_config

    # Finalize configs (auto-compute vocab_size, dirs, etc.)
    if hasattr(dc, "custom_finalize"):
        dc.custom_finalize()
    if hasattr(tc, "custom_finalize"):
        tc.custom_finalize()

    # Set seed
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    # Device
    device = torch.device(tc.device if torch.cuda.is_available() or tc.device == "cpu" else "cpu")

    # Create output directories
    if tc.save_dir is not None:
        os.makedirs(tc.save_dir, exist_ok=True)
    if tc.figs_dir is not None:
        os.makedirs(tc.figs_dir, exist_ok=True)

    # Log the attention-only setup
    print("Attention-only training mode:")
    print("  - Trainable attention matrices (V, c_proj unfrozen)")
    print("  - Identity MLP (disabled)")
    print("  - No positional encoding")
    print("  - No layer norm between attention and MLP (ln_2 disabled)")
    print("  - Unit RMSNorm before lm_head (scale=1)")
    if dc.use_identity_fact_mapping:
        print("  - Identity mapping enabled (associative-recall key-copy task):")
        print("    [junk] K [junk] Q V -> predict V=K at Q")

    # 1. Generate factset (embeddings + identity mapping)
    print("Generating factset...")
    factset = build_factset(config, seed=tc.seed)

    # 2. Create full embeddings (facts + junk + Q)
    print("Creating embeddings...")
    full_embeddings = build_token_embeddings(
        factset=factset,
        junk_vocab_size=dc.junk_vocab_size,
        embedding_init=tc.embeddings_config.embedding_init,
        dtype=tc.dtype,
        seed=tc.seed,
    )

    # 3. Create dataset and dataloader (train only, no separate eval)
    print("Creating dataset...")
    train_dataloader, _ = create_associative_recall_batches(
        dc, tc, factset.mapping, device=device,
    )

    # 4. Create GPT model (with identity MLP built-in via config)
    print("Creating GPT model...")
    gpt_config = create_gpt_config(tc, dc)
    gpt_model = GPT(gpt_config)
    gpt_model.to(device=device, dtype=tc.dtype)

    # 5. Copy embeddings to GPT
    copy_embeddings_to_gpt(
        gpt_model,
        full_embeddings,
        freeze_wte=tc.transformer_config.freeze_input_embeddings,
        freeze_lm_head=tc.transformer_config.freeze_output_embeddings,
        tie_embeddings=tc.embeddings_config.tie_embeddings,
    )

    # 6. Create optimizer
    optimizer = gpt_model.configure_optimizers(
        weight_decay=tc.weight_decay,
        learning_rate=tc.lr,
        betas=(0.9, 0.999),
        device_type=device.type,
    )

    # Log model info
    n_params = sum(p.numel() for p in gpt_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in gpt_model.parameters())
    print(f"  Trainable params: {n_params:,} / {n_total:,} total")

    # Optional torch.compile
    if tc.compile_model:
        print("Compiling model with torch.compile...")
        gpt_model = torch.compile(gpt_model)

    # 7. Training loop
    print(f"Training for {tc.epochs} epochs...")
    train_metrics = []
    train_eval_metrics = []
    best_acc = 0.0

    for epoch in tqdm(range(tc.epochs), desc="Training"):
        # --- Train ---
        gpt_model.train()
        dataloader_iter = iter(train_dataloader)

        for step in range(tc.steps_per_dataset):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_dataloader)
                batch = next(dataloader_iter)

            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            if tc.attention_loss_type == "l2":
                # L2/MSE loss on hidden states vs target embeddings
                h = gpt_model.forward_hidden(inputs)  # (B, T, d)
                target_mask = targets != -100
                h_pred = h[target_mask]
                target_embs = gpt_model.transformer.wte.weight.detach()[targets[target_mask]]
                loss = F.mse_loss(h_pred, target_embs)

                # Accuracy: use logits from lm_head(h)
                with torch.no_grad():
                    logits_all = gpt_model.lm_head(h)
                    preds = logits_all.argmax(dim=-1)
                    if target_mask.any():
                        batch_acc = (preds[target_mask] == targets[target_mask]).float().mean().item()
                    else:
                        batch_acc = 0.0
                logits = logits_all
            else:
                # Default CE loss
                logits, loss = gpt_model(inputs, targets=targets)

                # Batch accuracy
                mask = targets != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    batch_acc = (preds[mask] == targets[mask]).float().mean().item()
                else:
                    batch_acc = 0.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metrics.append({
                "epoch": epoch,
                "step": step,
                "loss": loss.item(),
                "accuracy": batch_acc,
            })

        # --- Evaluate ---
        if epoch % tc.evaluate_every == 0:
            gpt_model.eval()
            num_iters = max(1, int(factset.vocab_size / (len(train_dataloader) * train_dataloader.batch_size))) * 10
            train_eval_result = evaluate(gpt_model, train_dataloader, device, num_iterations=num_iters)
            train_eval_result["epoch"] = epoch
            train_eval_result["weight_norms"] = compute_weight_norms(gpt_model, trainable_only=True)
            train_eval_metrics.append(train_eval_result)

            if train_eval_result["accuracy"] > best_acc:
                best_acc = train_eval_result["accuracy"]

            # Plot progress
            if tc.figs_dir is not None and epoch % tc.plot_every == 0:
                plot_training_progress(
                    train_metrics=train_metrics,
                    train_eval_metrics=train_eval_metrics,
                    figs_dir=tc.figs_dir,
                    steps_per_epoch=tc.steps_per_dataset,
                )

            # Early stopping
            if tc.early_stop_accuracy is not None:
                if (train_eval_result["accuracy"] >= tc.early_stop_accuracy):
                    print(
                        f"Early stopping at epoch {epoch}: "
                        f"train_acc={train_eval_result['accuracy']:.4f}"
                    )
                    break

    # Final evaluation
    gpt_model.eval()
    num_iters = max(1, int(factset.vocab_size / (len(train_dataloader) * train_dataloader.batch_size))) * 10
    final_result = evaluate(gpt_model, train_dataloader, device, num_iterations=num_iters)
    final_weight_norms = compute_weight_norms(gpt_model, trainable_only=True)
    print(f"\nFinal accuracy: {final_result['accuracy']:.4f}")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Final total weight norm: {final_weight_norms['total_norm']:.4f}")

    # Final plot
    if tc.figs_dir is not None:
        plot_training_progress(
            train_metrics=train_metrics,
            train_eval_metrics=train_eval_metrics,
            figs_dir=tc.figs_dir,
            steps_per_epoch=tc.steps_per_dataset,
        )

    # Save checkpoint
    if tc.save_at_end and tc.save_dir is not None:
        checkpoint_path = os.path.join(tc.save_dir, "last_model.pt")
        torch.save({
            "epoch": epoch,
            "gpt_state_dict": gpt_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "gpt_config": gpt_config,
            "config": config,
            "train_metrics": train_metrics,
            "train_eval_metrics": train_eval_metrics,
            "best_acc": best_acc,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    return {
        "gpt_model": gpt_model,
        "factset": factset,
        "train_metrics": train_metrics,
        "train_eval_metrics": train_eval_metrics,
        "best_acc": best_acc,
        "final_accuracy": final_result["accuracy"],
        "final_weight_norms": final_weight_norms,
        "train_dataloader": train_dataloader,
    }
