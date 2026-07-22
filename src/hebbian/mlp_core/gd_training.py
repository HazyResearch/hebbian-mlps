"""
General gradient descent training utilities.

This module provides the shared optimization loop for the paper's GD baseline.

Supports optional minibatch training for memory efficiency.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, Iterable, Optional, List, Union
import torch
import warnings
from tqdm import tqdm


__all__ = [
    "GDBatchConfig",
    "GDOptimizerConfig",
    "GDLogConfig",
    "GDTrainingResult",
    "train_with_gd",
]


@dataclass
class GDBatchConfig:
    """Configuration for minibatch training.

    Attributes:
        batch_size: Batch size for minibatch training (None = full batch)
        shuffle: Whether to shuffle data each epoch
        accumulate_loss: Whether to accumulate loss across batches for logging
                        (True = average loss over epoch, False = last batch loss)
    """

    batch_size: Optional[int] = None
    shuffle: bool = True
    accumulate_loss: bool = True


@dataclass
class GDOptimizerConfig:
    """Configuration for gradient descent optimizer.

    Attributes:
        lr: Initial learning rate for Adam optimizer
        min_lr: Minimum learning rate for cosine annealing scheduler
        num_epochs: Maximum number of training epochs
        cutoff: Optional absolute loss threshold - stop if loss < cutoff
        batch_config: Optional batching configuration for minibatch training
    """

    lr: float = 1e-3
    min_lr: float = 1e-6
    num_epochs: int = 10000
    cutoff: Optional[float] = None
    batch_config: Optional[GDBatchConfig] = None


@dataclass
class GDLogConfig:
    """Configuration for logging during training.

    Attributes:
        log_every: Print loss every N epochs
        eval_every: Evaluate metrics every N epochs (None = never evaluate)
        verbose: Whether to print progress during training
    """

    log_every: int = 100
    eval_every: Optional[int] = None
    verbose: bool = True


@dataclass
class GDTrainingResult:
    """Results from gradient descent training.

    Attributes:
        final_loss: Final loss value after training
        train_losses: List of losses per epoch
        metrics: Dict of metric_name -> list of values over epochs
    """

    final_loss: float
    train_losses: List[float]
    metrics: Dict[str, List[float]]


def train_with_gd(
    params: Iterable[torch.Tensor],
    compute_loss: Callable[[Optional[torch.Tensor]], torch.Tensor],
    metrics: Optional[Dict[str, Callable[[], float]]] = None,
    optimizer_config: GDOptimizerConfig = None,
    log_config: GDLogConfig = None,
    num_samples: Optional[int] = None,
) -> GDTrainingResult:
    """
    General gradient descent training loop using Adam optimizer with cosine annealing.

    This function provides a standardized training loop that can be used across
    different gradient descent scenarios by accepting a loss computation callable
    and optional metric computation callables.

    Supports optional minibatch training for memory efficiency. When batching is
    enabled (via optimizer_config.batch_config), the compute_loss function receives
    batch indices. Otherwise, it receives None.

    Parameters are updated in-place, so there's no need to return them.

    Args:
        params: Parameters to optimize (e.g., mlp.parameters() or [M, h]).
                These are updated in-place during training.
        compute_loss: Callable that computes and returns loss tensor.
                     - If batching disabled: receives None, should compute loss on all data
                     - If batching enabled: receives batch_indices (torch.Tensor), should
                       compute loss only on that batch
                     This closure should capture all necessary state (model, data, etc.)
                     BF16/tiling should be handled inside this function if needed.
        metrics: Optional dict of metric_name -> callable that computes the metric.
                Each metric callable should return a float. Metrics are evaluated
                every eval_every epochs if provided. Metrics always operate on full data.
        optimizer_config: Optimizer configuration. Defaults to GDOptimizerConfig()
        log_config: Logging configuration. Defaults to GDLogConfig()
        num_samples: Total number of training samples. Required if batching is enabled.

    Returns:
        GDTrainingResult containing final_loss, train_losses, and metrics

    Example:
        >>> # Full-batch training (no batching)
        >>> def compute_loss(batch_indices):
        ...     # batch_indices will be None
        ...     outputs = mlp(X)
        ...     return F.mse_loss(outputs, Y[Y_labels])
        >>>
        >>> result = train_with_gd(
        ...     params=mlp.parameters(),
        ...     compute_loss=compute_loss,
        ...     optimizer_config=GDOptimizerConfig(lr=1e-3),
        ... )

        >>> # Minibatch training
        >>> def compute_loss_batched(batch_indices):
        ...     if batch_indices is None:
        ...         xb, yb = X, Y_labels
        ...     else:
        ...         xb = X[batch_indices]
        ...         yb = Y_labels[batch_indices]
        ...     outputs = mlp(xb)
        ...     logits = outputs @ Y.T
        ...     return F.cross_entropy(logits, yb)
        >>>
        >>> batch_config = GDBatchConfig(batch_size=256, shuffle=True)
        >>> optimizer_config = GDOptimizerConfig(lr=1e-3, batch_config=batch_config)
        >>> result = train_with_gd(
        ...     params=mlp.parameters(),
        ...     compute_loss=compute_loss_batched,
        ...     num_samples=len(X),
        ...     optimizer_config=optimizer_config,
        ... )
    """
    # Use default configs if not provided
    if optimizer_config is None:
        optimizer_config = GDOptimizerConfig()
    if log_config is None:
        log_config = GDLogConfig()

    # Check batching configuration
    batch_config = optimizer_config.batch_config
    use_batching = batch_config is not None and batch_config.batch_size is not None

    if use_batching and num_samples is None:
        raise ValueError("num_samples must be provided when batching is enabled")

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(params, lr=optimizer_config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimizer_config.num_epochs, eta_min=optimizer_config.min_lr
    )

    # Initialize tracking
    train_losses: List[float] = []
    metric_history: Dict[str, List[float]] = {name: [] for name in (metrics or {})}

    # Get device from first parameter (if available)
    # Convert params to list first to avoid consuming the iterator
    params = list(params)

    if not params:
        raise ValueError("params is empty - no parameters to optimize")

    device = None
    try:
        device = params[0].device
        print(f"[train_with_gd] Using device: {device}")
    except AttributeError as e:
        warnings.warn(
            f"Could not detect device from parameters (error: {e}). "
            f"Defaulting to CPU. This may cause performance issues.",
            RuntimeWarning,
        )
        device = torch.device("cpu")

    # Training loop with progress bar
    show_progress = log_config.verbose
    pbar = tqdm(
        range(optimizer_config.num_epochs),
        desc=f"GD Training ({optimizer_config.num_epochs} epochs)",
        disable=not show_progress,
        unit="epoch",
        file=None,  # Use stderr by default (tqdm default)
    )

    for epoch in pbar:
        if use_batching:
            # Minibatch training
            if batch_config.shuffle:
                perm = torch.randperm(num_samples, device=device)
            else:
                perm = torch.arange(num_samples, device=device)

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, num_samples, batch_config.batch_size):
                end = min(start + batch_config.batch_size, num_samples)
                batch_indices = perm[start:end]

                optimizer.zero_grad()
                loss = compute_loss(batch_indices)
                loss.backward()
                optimizer.step()

                if batch_config.accumulate_loss:
                    epoch_loss += loss.item() * len(batch_indices)
                    num_batches += len(batch_indices)
                else:
                    epoch_loss = loss.item()

            # Scheduler steps once per epoch (not per batch)
            scheduler.step()

            # Compute epoch loss
            if batch_config.accumulate_loss:
                loss_val = epoch_loss / num_batches
            else:
                loss_val = epoch_loss
        else:
            # Full-batch training
            optimizer.zero_grad()
            loss = compute_loss(None)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()

        train_losses.append(loss_val)

        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss_val:.6f}"})

        # Check absolute loss cutoff
        if optimizer_config.cutoff is not None and loss_val <= optimizer_config.cutoff:
            if log_config.verbose:
                pbar.write(
                    f"Converged at epoch {epoch} "
                    f"(loss={loss_val:.6e} < cutoff={optimizer_config.cutoff:.6e})"
                )
            break

        # Log loss (using pbar.write to avoid interfering with progress bar)
        if log_config.verbose and (
            epoch % log_config.log_every == 0
            or epoch == optimizer_config.num_epochs - 1
        ):
            pbar.write(f"Epoch {epoch:5d}: loss={loss_val:.6f}")

        # Evaluate metrics
        if (
            log_config.eval_every is not None
            and metrics
            and (
                epoch % log_config.eval_every == 0
                or epoch == optimizer_config.num_epochs - 1
            )
        ):
            metric_dict = {"loss": f"{loss_val:.6f}"}
            for name, metric_fn in metrics.items():
                metric_val = metric_fn()
                metric_history[name].append(metric_val)
                metric_dict[name] = f"{metric_val:.4f}"
                if log_config.verbose:
                    pbar.write(f"  {name}={metric_val:.4f}")
            # Update progress bar with metrics
            pbar.set_postfix(metric_dict)

    pbar.close()

    return GDTrainingResult(
        final_loss=train_losses[-1] if train_losses else float("inf"),
        train_losses=train_losses,
        metrics=metric_history,
    )
