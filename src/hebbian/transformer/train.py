"""Associative-recall training pipeline.

Usage:
    from hebbian.transformer.train import train_associative_recall
    from hebbian.transformer.config import AssociativeRecallConfig

    results = train_associative_recall(AssociativeRecallConfig())
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hebbian.transformer.config import AssociativeRecallConfig
from hebbian.transformer.model import GPT
from hebbian.transformer.utils import (
    create_gpt_config,
    insert_mlp_into_gpt,
    copy_embeddings_to_gpt,
    compute_weight_norms,
    evaluate,
)
from hebbian.transformer.data import create_associative_recall_batches
from hebbian.transformer.fact_store import (
    build_factset,
    build_fact_mlp,
    build_token_embeddings,
)
from hebbian.transformer.plotting import plot_training_progress
from hebbian.data.synthetics.factsets import (
    Factset,
    create_random_permutation_mapping,
)


@torch.no_grad()
def _compute_mlp_gamma_min(
    mlp: nn.Module,
    factset: Factset,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Compute minimum margin of mlp(K) scored against value embeddings.

    gamma_min = min_i [ <mlp(k_i), v_{f(i)}> - max_{j!=f(i)} <mlp(k_i), v_j> ]

    For tie_embeddings=True factsets, K == V (same embedding matrix).
    """
    mlp.eval()
    K     = factset.input_embeddings.to(device=device, dtype=dtype)   # (n, d)
    V_all = factset.output_embeddings.to(device=device, dtype=dtype)  # (P, d)
    n     = K.shape[0]

    value_idx = torch.tensor(
        [factset.mapping.get_output(i) for i in range(n)],
        device=device, dtype=torch.long,
    )

    Y      = mlp(K)                                   # (n, d)
    Y      = Y / Y.norm(dim=1, keepdim=True).clamp(min=1e-8)  # normalise to unit norm
    scores = Y @ V_all.T                              # (n, P)

    batch  = torch.arange(n, device=device)
    correct = scores[batch, value_idx]                # (n,)
    scores_masked = scores.clone()
    scores_masked[batch, value_idx] = float("-inf")
    max_wrong = scores_masked.max(dim=1).values       # (n,)

    return (correct - max_wrong).min().item()


def _make_eval_factset(train_factset: Factset, seed: int) -> Factset:
    """Create an eval factset: same embeddings, different random permutation.

    Mirrors the mlps repo pattern where embeddings are shared across
    train/eval but each gets its own random mapping permutation.
    """
    eval_mapping = create_random_permutation_mapping(train_factset.vocab_size, seed=seed)
    eval_factset = object.__new__(Factset)
    eval_factset.input_embeddings = train_factset.input_embeddings
    eval_factset.output_embeddings = train_factset.output_embeddings
    eval_factset.mapping = eval_mapping
    eval_factset.d_model = train_factset.d_model
    eval_factset.vocab_size = train_factset.vocab_size
    return eval_factset


def _make_eval_config(
    config: AssociativeRecallConfig,
) -> AssociativeRecallConfig:
    """Create a config copy with the eval-MLP settings applied."""
    import copy
    eval_config = copy.deepcopy(config)
    eval_config.train_config.mlp_method = config.train_config.eval_mlp_method
    if config.train_config.eval_mlp_hidden_dim is not None:
        eval_config.train_config.mlp_hidden_dim = config.train_config.eval_mlp_hidden_dim
    eval_config.train_config.mlp_dtype = config.train_config.eval_mlp_dtype
    eval_config.train_config.mlp_method_kwargs = config.train_config.eval_mlp_method_kwargs
    return eval_config


def _swap_mlp(gpt_model: GPT, mlp: nn.Module, full_embeddings: nn.Embedding, tc) -> None:
    """Swap the MLP in all transformer blocks."""
    gpt_model.eval()
    insert_mlp_into_gpt(
        gpt_model,
        mlp,
        full_embeddings,
        freeze_mlp=True,
        freeze_wte=tc.transformer_config.freeze_input_embeddings,
        freeze_lm_head=tc.transformer_config.freeze_output_embeddings,
    )


def train_associative_recall(
    config: AssociativeRecallConfig,
) -> Dict[str, Any]:
    """Train the Transformer on the associative-recall task.

    Steps:
        1. Generate factset (embeddings + mapping)
        2. Create MLP from factset using Method interface
        3. Create datasets and dataloaders
        4. Create GPT model, copy embeddings, insert MLP
        5. Train transformer (attention only, MLP frozen)
        6. Return results

    Args:
        config: AssociativeRecallConfig instance.

    Returns:
        Dict with training results including metrics and final accuracy.
    """
    dc = config.dataset_config
    tc = config.train_config

    # Finalize configs (auto-compute vocab_size, dirs, etc.)
    if hasattr(dc, "custom_finalize"):
        dc.custom_finalize()
    if hasattr(tc, "custom_finalize"):
        tc.custom_finalize()
    # tc.custom_finalize doesn't cascade into nested @pydraclass instances; call
    # the embeddings_config finalize explicitly so the LLM-mode d_model override
    # and tie_embeddings check fire even when callers skip config.finalize().
    ec = tc.embeddings_config
    if hasattr(ec, "custom_finalize"):
        ec.custom_finalize()

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

    # LLM-embeddings mode: "frozen_rmsnorm" asserts row-RMS uniformity within
    # rtol=0.2 (see transformer/norms.py); raw LLM activations will almost
    # always violate that. Reject the combo up front rather than crashing inside
    # the first forward pass.
    if tc.embeddings_config.embeddings_dir is not None:
        bad = [
            name
            for name, val in (
                ("mlp_norm_type", tc.transformer_config.mlp_norm_type),
                ("attn_norm_type", tc.transformer_config.attn_norm_type),
                ("lm_head_norm_type", tc.transformer_config.lm_head_norm_type),
            )
            if val == "frozen_rmsnorm"
        ]
        if bad:
            raise ValueError(
                f"embeddings_dir is incompatible with 'frozen_rmsnorm' on {bad}: "
                "raw LLM activation rows fail the row-RMS-uniformity assertion. "
                "Pick 'none', 'unit_rmsnorm', 'rmsnorm', or 'layernorm'."
            )

    # 1. Generate factset
    print("Generating factset...")
    factset = build_factset(config, seed=tc.seed)
    if tc.embeddings_config.embeddings_dir is not None:
        junk_rows = getattr(factset, "_llm_junk_embeddings", None)
        n_junk = 0 if junk_rows is None else int(junk_rows.shape[0])
        print(
            f"  [LLM mode] embeddings_dir={tc.embeddings_config.embeddings_dir!r} "
            f"d={factset.d_model} F={factset.vocab_size} junk+Q={n_junk} "
            f"mapping=identity"
        )

    # 2. Create MLP (train MLP)
    print(f"Creating MLP using method '{tc.mlp_method}'...")
    mlp, mlp_metrics = build_fact_mlp(config, factset)
    mlp_accuracy = mlp_metrics.get("accuracy", mlp_metrics.get("final_accuracy", None))
    print(f"  MLP accuracy: {mlp_accuracy}")

    # MLP-accuracy gate: short-circuit before the transformer training loop if
    # the standalone MLP can't already store the facts above threshold.
    if tc.mlp_acc_threshold is not None:
        acc_val = float(mlp_accuracy) if mlp_accuracy is not None else float("nan")
        if not (acc_val >= float(tc.mlp_acc_threshold)):
            print(
                f"  [mlp_acc_threshold] MLP accuracy {acc_val:.4f} < threshold "
                f"{float(tc.mlp_acc_threshold):.4f}; failing fast "
                "(skipping transformer training)."
            )
            mlp_native = mlp.to(device=device, dtype=tc.dtype)
            gamma = _compute_mlp_gamma_min(mlp_native, factset, device, tc.dtype)
            mlp_metrics["gamma_min"] = gamma
            mlp_metrics["hidden_dim"] = mlp_metrics.get(
                "hidden_dim", tc.mlp_hidden_dim
            )
            mlp_native.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "gpt_model": None,
                "mlp": mlp,
                "mlp_metrics": mlp_metrics,
                "factset": factset,
                "train_metrics": [],
                "eval_metrics": [],
                "train_eval_metrics": [],
                "best_acc": 0.0,
                "best_train_acc": 0.0,
                "final_eval_accuracy": 0.0,
                "final_train_accuracy": 0.0,
                "mlp_gamma_min": gamma,
                "train_dataloader": None,
                "final_weight_norms": {},
                "mlp_str": str(mlp),
                "gpt_str": "(skipped: mlp_acc_threshold gate failed)",
            }

    # 2b. Create eval MLP (optional — separate MLP with different permutation mapping)
    # Mirrors mlps repo: embeddings are shared, but eval gets a fresh random permutation.
    eval_mlp = None
    eval_mlp_metrics = None
    eval_factset = None
    if tc.eval_mlp_method is not None:
        print(f"Creating eval MLP using method '{tc.eval_mlp_method}'...")
        eval_factset = _make_eval_factset(factset, seed=tc.seed + 7777)
        eval_config = _make_eval_config(config)
        eval_mlp, eval_mlp_metrics = build_fact_mlp(eval_config, eval_factset)
        eval_mlp_accuracy = eval_mlp_metrics.get("accuracy", eval_mlp_metrics.get("final_accuracy", None))
        print(f"  Eval MLP accuracy: {eval_mlp_accuracy}")

    # 3. Create full embeddings (facts + junk + Q)
    print("Creating embeddings...")
    full_embeddings = build_token_embeddings(
        factset=factset,
        junk_vocab_size=dc.junk_vocab_size,
        embedding_init=tc.embeddings_config.embedding_init,
        dtype=tc.dtype,
        seed=tc.seed,
        side="input",
    )
    full_embeddings = full_embeddings.to(device=device)
    # When wte and lm_head are independent (LLM mode, or any tie_embeddings=False
    # configuration), also build the output-side table so copy_embeddings_to_gpt
    # can populate lm_head.
    full_output_embeddings = None
    if not tc.embeddings_config.tie_embeddings:
        full_output_embeddings = build_token_embeddings(
            factset=factset,
            junk_vocab_size=dc.junk_vocab_size,
            embedding_init=tc.embeddings_config.embedding_init,
            dtype=tc.dtype,
            seed=tc.seed,
            side="output",
        ).to(device=device)

    # 4. Create datasets
    print("Creating datasets...")
    train_dataloader, _ = create_associative_recall_batches(
        dc, tc, factset.mapping, device=device,
    )
    # Eval dataloaders use eval mapping if eval MLP is configured, else train mapping
    eval_mapping = eval_factset.mapping if eval_factset is not None else factset.mapping
    _, eval_dataloader = create_associative_recall_batches(
        dc, tc, eval_mapping, device=device,
    )

    # 5. Create GPT model
    print("Creating GPT model...")
    gpt_config = create_gpt_config(tc, dc)
    gpt_model = GPT(gpt_config)
    gpt_model.to(device=device, dtype=tc.dtype)

    # Copy embeddings
    copy_embeddings_to_gpt(
        gpt_model,
        full_embeddings,
        output_embeddings=full_output_embeddings,
        freeze_wte=tc.transformer_config.freeze_input_embeddings,
        freeze_lm_head=tc.transformer_config.freeze_output_embeddings,
        tie_embeddings=tc.embeddings_config.tie_embeddings,
    )

    # Insert MLP
    gpt_model.eval()
    mlp = mlp.to(device=device, dtype=tc.dtype)
    if eval_mlp is not None:
        eval_mlp = eval_mlp.to(device=device, dtype=tc.dtype)

    # Compute MLP standalone margin (gamma_min) before transformer training
    mlp_gamma_min = _compute_mlp_gamma_min(mlp, factset, device, tc.dtype)
    mlp_metrics["gamma_min"] = mlp_gamma_min
    print(f"  MLP gamma_min: {mlp_gamma_min:.4f}")

    insert_mlp_into_gpt(
        gpt_model,
        mlp,
        full_embeddings,
        freeze_mlp=True,
        freeze_wte=tc.transformer_config.freeze_input_embeddings,
        freeze_lm_head=tc.transformer_config.freeze_output_embeddings,
    )

    # 6. Create optimizer
    optimizer = gpt_model.configure_optimizers(
        weight_decay=tc.weight_decay,
        learning_rate=tc.lr,
        betas=(0.9, 0.999),
        device_type=device.type,
    )

    # Capture architecture strings for checkpointing
    mlp_str = str(mlp)
    gpt_str = str(gpt_model)

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
    eval_metrics = []
    train_eval_metrics = []
    best_acc = 0.0
    best_train_acc = 0.0

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

            logits, loss = gpt_model(inputs, targets=targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Batch accuracy
            mask = targets != -100
            if mask.any():
                preds = logits.argmax(dim=-1)
                batch_acc = (preds[mask] == targets[mask]).float().mean().item()
            else:
                batch_acc = 0.0

            train_metrics.append({
                "epoch": epoch,
                "step": step,
                "loss": loss.item(),
                "accuracy": batch_acc,
            })

        # --- Evaluate ---
        if epoch % tc.evaluate_every == 0:
            # Eval accuracy: eval MLP + eval dataloader (generalization test)
            if eval_mlp is not None:
                _swap_mlp(gpt_model, eval_mlp, full_embeddings, tc)

            gpt_model.eval()
            eval_num_iters = max(1, int(factset.vocab_size / (len(eval_dataloader) * eval_dataloader.batch_size))) * 10
            eval_result = evaluate(gpt_model, eval_dataloader, device, num_iterations=eval_num_iters)
            eval_result["epoch"] = epoch
            eval_result["weight_norms"] = compute_weight_norms(gpt_model, trainable_only=True)
            eval_metrics.append(eval_result)

            if eval_result["accuracy"] > best_acc:
                best_acc = eval_result["accuracy"]

            # Train accuracy: train MLP + train dataloader
            if eval_mlp is not None:
                _swap_mlp(gpt_model, mlp, full_embeddings, tc)
            train_num_iters = max(1, int(factset.vocab_size / (len(train_dataloader) * train_dataloader.batch_size))) * 10
            train_eval_result = evaluate(gpt_model, train_dataloader, device, num_iterations=train_num_iters)
            train_eval_result["epoch"] = epoch
            train_eval_result["weight_norms"] = compute_weight_norms(gpt_model, trainable_only=True)
            train_eval_metrics.append(train_eval_result)

            if train_eval_result["accuracy"] > best_train_acc:
                best_train_acc = train_eval_result["accuracy"]

            # Plot progress
            if tc.figs_dir is not None and epoch % tc.plot_every == 0:
                plot_training_progress(
                    train_metrics=train_metrics,
                    train_eval_metrics=train_eval_metrics,
                    eval_metrics=eval_metrics,
                    figs_dir=tc.figs_dir,
                    steps_per_epoch=tc.steps_per_dataset,
                )

            # Early stopping
            if tc.early_stop_accuracy is not None:
                if (eval_result["accuracy"] >= tc.early_stop_accuracy
                        and train_eval_result["accuracy"] >= tc.early_stop_accuracy):
                    print(
                        f"Early stopping at epoch {epoch}: "
                        f"eval_acc={eval_result['accuracy']:.4f}, "
                        f"train_acc={train_eval_result['accuracy']:.4f}"
                    )
                    break

    # Final evaluation
    # Eval accuracy: eval MLP + eval dataloader
    if eval_mlp is not None:
        _swap_mlp(gpt_model, eval_mlp, full_embeddings, tc)
    gpt_model.eval()
    eval_num_iters = max(1, int(factset.vocab_size / (len(eval_dataloader) * eval_dataloader.batch_size))) * 10
    final_eval = evaluate(gpt_model, eval_dataloader, device, num_iterations=eval_num_iters)
    # Train accuracy: train MLP + train dataloader
    if eval_mlp is not None:
        _swap_mlp(gpt_model, mlp, full_embeddings, tc)
    train_num_iters = max(1, int(factset.vocab_size / (len(train_dataloader) * train_dataloader.batch_size))) * 10
    final_train_eval = evaluate(gpt_model, train_dataloader, device, num_iterations=train_num_iters)
    final_weight_norms = compute_weight_norms(gpt_model, trainable_only=True)
    print(f"\nFinal eval accuracy: {final_eval['accuracy']:.4f}")
    print(f"Final train accuracy: {final_train_eval['accuracy']:.4f}")
    print(f"Best eval accuracy: {best_acc:.4f}")
    print(f"Final total weight norm: {final_weight_norms['total_norm']:.4f}")

    # Final plot
    if tc.figs_dir is not None:
        plot_training_progress(
            train_metrics=train_metrics,
            train_eval_metrics=train_eval_metrics,
            eval_metrics=eval_metrics,
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
            "eval_metrics": eval_metrics,
            "train_eval_metrics": train_eval_metrics,
            "best_acc": best_acc,
            "best_train_acc": best_train_acc,
            "mlp_metrics": mlp_metrics,
            "mlp_str": mlp_str,
            "gpt_str": gpt_str,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    results = {
        "gpt_model": gpt_model,
        "mlp": mlp,
        "mlp_metrics": mlp_metrics,      # includes gamma_min, final_accuracy, param_count, hidden_dim
        "factset": factset,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "train_eval_metrics": train_eval_metrics,
        "best_acc": best_acc,
        "best_train_acc": best_train_acc,
        "final_eval_accuracy": final_eval["accuracy"],
        "final_train_accuracy": final_train_eval["accuracy"],
        "train_dataloader": train_dataloader,
        "final_weight_norms": final_weight_norms,
        "mlp_gamma_min": mlp_gamma_min,  # convenience alias (also in mlp_metrics["gamma_min"])
        "mlp_str": mlp_str,
        "gpt_str": gpt_str,
    }
    if eval_mlp is not None:
        results["eval_mlp"] = eval_mlp
        results["eval_mlp_metrics"] = eval_mlp_metrics
        results["eval_factset"] = eval_factset
        results["eval_dataloader"] = eval_dataloader
    return results
