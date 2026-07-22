"""Gradient-descent baseline for the paper's gated MLP."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn.functional as F

from hebbian.config import pydraclass
from hebbian.mlp_core.blocks import GatedLinearBlock, GatedMLP, LinearBlock
from hebbian.mlp_core.gd_training import (
    GDBatchConfig,
    GDLogConfig,
    GDOptimizerConfig,
    train_with_gd,
)
from hebbian.mlp_core.task import SharedConstructionConfig

if TYPE_CHECKING:
    from hebbian.data.synthetics.factsets import Factset


@pydraclass
class GDMLPConfig:
    """Configuration for fitting the gated MLP baseline with Adam."""

    shared: SharedConstructionConfig = field(default_factory=SharedConstructionConfig)
    bias: bool = True
    expansion_factor: float = 4.0
    m: Optional[int] = None
    num_epochs: int = 10000
    lr: float = 1e-3
    min_lr: float = 1e-6
    cutoff: float = 1e-7
    log_every: int = 100
    eval_every: int = 1000
    loss_fn: str = "ce"
    batch_size: Optional[int] = None
    shuffle: bool = False
    use_bf16: bool = False


def _create_gd_mlp(
    d_model: int,
    intermediate_size: int,
    config: GDMLPConfig,
) -> GatedMLP:
    device = (
        torch.device(config.shared.device)
        if isinstance(config.shared.device, str)
        else config.shared.device
    )
    dtype = config.shared.build_dtype
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.replace("torch.", ""))
    activation = config.shared.mlp_config.activation.get_activation()
    up = GatedLinearBlock(
        d_model,
        intermediate_size,
        activation,
        bias=config.bias,
        dtype=dtype,
        device=device,
    )
    down = LinearBlock(
        intermediate_size,
        d_model,
        bias=config.bias,
        dtype=dtype,
        device=device,
    )
    return GatedMLP(up, down)


def get_gd_mlp(factset: Factset, config: GDMLPConfig) -> tuple[GatedMLP, dict]:
    """Fit a gated MLP to the associations in ``factset``."""

    device = (
        torch.device(config.shared.device)
        if isinstance(config.shared.device, str)
        else config.shared.device
    )
    d_model = factset.input_embeddings.shape[1]
    intermediate_size = (
        int(config.m)
        if config.m is not None
        else int(d_model * config.expansion_factor)
    )
    mlp = _create_gd_mlp(d_model, intermediate_size, config)

    if config.shared.verbose:
        print("=" * 60)
        print("GRADIENT DESCENT MLP CONSTRUCTION")
        mlp.print_shapes()
        print(f"Activation: {config.shared.mlp_config.activation.get_activation().name()}")
        print(f"Loss function: {config.loss_fn}")
        print(f"Batch size: {config.batch_size or 'full batch'}")
        print("=" * 60)

    num_inputs = factset.input_embeddings.shape[0]
    input_indices = np.arange(num_inputs)
    labels = torch.tensor(
        [factset.mapping.get_output(i) for i in input_indices],
        dtype=torch.int64,
        device=device,
    )
    inputs = factset.input_embeddings[input_indices].to(
        dtype=config.shared.build_dtype, device=device
    ).detach()
    values = factset.output_embeddings.to(
        dtype=config.shared.build_dtype, device=device
    ).detach()

    bf16_values_t = None
    if config.loss_fn.lower() == "ce" and config.use_bf16 and device.type == "cuda":
        bf16_values_t = values.to(torch.bfloat16).t().contiguous()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def compute_loss(batch_indices):
        batch_inputs = inputs if batch_indices is None else inputs[batch_indices]
        batch_labels = labels if batch_indices is None else labels[batch_indices]
        outputs = mlp(batch_inputs)
        if config.loss_fn.lower() == "mse":
            return F.mse_loss(outputs, values[batch_labels])
        if bf16_values_t is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = outputs @ bf16_values_t
        else:
            logits = outputs @ values.T
        return F.cross_entropy(logits, batch_labels)

    def compute_acc() -> float:
        with torch.no_grad():
            predictions = torch.argmax(mlp(inputs) @ values.T, dim=-1)
            return (predictions == labels).to(float).mean().item()

    def compute_mse() -> float:
        with torch.no_grad():
            return F.mse_loss(mlp(inputs), values[labels]).item()

    batch_config = (
        GDBatchConfig(batch_size=config.batch_size, shuffle=config.shuffle)
        if config.batch_size
        else None
    )
    result = train_with_gd(
        params=list(mlp.parameters()),
        compute_loss=compute_loss,
        metrics={"accuracy": compute_acc, "mse": compute_mse},
        optimizer_config=GDOptimizerConfig(
            lr=config.lr,
            min_lr=config.min_lr,
            num_epochs=config.num_epochs,
            cutoff=config.cutoff,
            batch_config=batch_config,
        ),
        log_config=GDLogConfig(
            log_every=config.log_every,
            eval_every=config.eval_every,
            verbose=config.shared.verbose,
        ),
        num_samples=num_inputs if config.batch_size else None,
    )
    final_acc = compute_acc()
    final_mse = compute_mse()
    return mlp, {
        "final_accuracy": final_acc,
        "final_mse": final_mse,
        "train_losses": result.train_losses,
        "metrics_history": result.metrics,
    }
