"""Plotting utilities for associative-recall training."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_progress(
    train_metrics: List[Dict[str, Any]],
    train_eval_metrics: List[Dict[str, Any]],
    eval_metrics: Optional[List[Dict[str, Any]]] = None,
    figs_dir: str = ".",
    steps_per_epoch: int = 1,
    save_name: str = "training_progress",
) -> None:
    """Plot training loss, accuracy, and weight norms over time.

    Creates a 2x2 figure saved to ``figs_dir/save_name.png``:

    * Top-left:     Loss (per-step train + checkpoint evals)
    * Top-right:    Accuracy (per-step train + checkpoint evals)
    * Bottom-left:  Total weight norm over training
    * Bottom-right: Per-parameter weight norms

    Args:
        train_metrics: Per-step metrics with keys
            ``epoch``, ``step``, ``loss``, ``accuracy``, ``weight_norms``.
        train_eval_metrics: Per-checkpoint eval on train data with keys
            ``epoch``, ``loss``, ``accuracy``, ``weight_norms``.
        eval_metrics: Per-checkpoint eval on eval data (optional).
        figs_dir: Directory to save the figure.
        steps_per_epoch: Number of training steps per epoch.
        save_name: Filename without extension.
    """
    if not train_metrics:
        return
    os.makedirs(figs_dir, exist_ok=True)

    fig, ((ax_loss, ax_acc), (ax_norm, ax_norms)) = plt.subplots(
        2, 2, figsize=(14, 10), dpi=150,
    )

    # X-axes ----------------------------------------------------------------
    train_x = [m["epoch"] + m["step"] / steps_per_epoch for m in train_metrics]
    te_x = [m["epoch"] for m in train_eval_metrics] if train_eval_metrics else []
    ev_x = [m["epoch"] for m in eval_metrics] if eval_metrics else []

    # ---- Loss --------------------------------------------------------------
    ax_loss.plot(
        train_x, [m["loss"] for m in train_metrics],
        alpha=0.3, color="C0", linewidth=0.5, label="Train (per step)",
    )
    if train_eval_metrics:
        ax_loss.plot(
            te_x, [m["loss"] for m in train_eval_metrics],
            color="C0", linewidth=1.5, label="Train (eval)",
        )
    if eval_metrics:
        ax_loss.plot(
            ev_x, [m["loss"] for m in eval_metrics],
            color="C1", linewidth=1.5, linestyle="--", label="Eval",
        )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.set_yscale("log")
    ax_loss.legend(fontsize="small")
    ax_loss.grid(True, alpha=0.3)

    # ---- Accuracy -----------------------------------------------------------
    ax_acc.plot(
        train_x, [m["accuracy"] for m in train_metrics],
        alpha=0.3, color="C0", linewidth=0.5, label="Train (per step)",
    )
    if train_eval_metrics:
        ax_acc.plot(
            te_x, [m["accuracy"] for m in train_eval_metrics],
            color="C0", linewidth=1.5, label="Train (eval)",
        )
    if eval_metrics:
        ax_acc.plot(
            ev_x, [m["accuracy"] for m in eval_metrics],
            color="C1", linewidth=1.5, linestyle="--", label="Eval",
        )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.legend(fontsize="small")
    ax_acc.grid(True, alpha=0.3)

    # ---- Total weight norm --------------------------------------------------
    norm_src = train_eval_metrics if train_eval_metrics else train_metrics
    norm_x = te_x if train_eval_metrics else train_x
    total_norms = [m["weight_norms"]["total_norm"] for m in norm_src]
    ax_norm.plot(norm_x, total_norms, color="C2", linewidth=1.5)
    ax_norm.set_xlabel("Epoch")
    ax_norm.set_ylabel("L2 Norm")
    ax_norm.set_title("Total Weight Norm (trainable)")
    ax_norm.grid(True, alpha=0.3)

    # ---- Per-parameter weight norms -----------------------------------------
    if norm_src:
        last_norms = norm_src[-1]["weight_norms"]
        param_names = [
            k for k in last_norms
            if k not in ("total_norm", "total_norm_squared")
        ]
        for pname in param_names:
            vals = [m["weight_norms"].get(pname, 0.0) for m in norm_src]
            parts = pname.split(".")
            short = ".".join(parts[-2:]) if len(parts) >= 2 else pname
            ax_norms.plot(norm_x, vals, linewidth=1.0, alpha=0.8, label=short)
        ax_norms.set_xlabel("Epoch")
        ax_norms.set_ylabel("L2 Norm")
        ax_norms.set_title("Per-Parameter Weight Norms")
        ax_norms.legend(fontsize="x-small", loc="best")
        ax_norms.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"{save_name}.png"))
    plt.close(fig)
