"""
Metrics computation utilities.

This module provides utilities for computing:
    - Accuracy metrics
    - Margin metrics
    - Parameter counting
"""

from typing import Optional, Tuple

import torch
from torch import nn


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_embeddings: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute classification accuracy.

    If output_embeddings is provided, predictions are assumed to be
    continuous vectors and accuracy is computed via nearest neighbor.

    Args:
        predictions: Predicted values (N, d) or (N,)
        targets: Target values (N,) or target indices
        output_embeddings: Optional output embedding matrix (V, d)

    Returns:
        Accuracy as a float between 0 and 1
    """
    with torch.no_grad():
        if output_embeddings is not None:
            # Compute similarities and find nearest neighbor
            similarities = predictions @ output_embeddings.T
            predicted_indices = torch.argmax(similarities, dim=-1)
            correct = (predicted_indices == targets).float()
        else:
            # Direct comparison for discrete predictions
            if predictions.dim() > 1:
                predicted_indices = torch.argmax(predictions, dim=-1)
            else:
                predicted_indices = predictions
            correct = (predicted_indices == targets).float()

        return correct.mean().item()


def compute_margin(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_embeddings: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute margin metrics.

    The margin is defined as the difference between the similarity to the
    correct class and the similarity to the most similar incorrect class.

    Args:
        predictions: Predicted embeddings (N, d)
        targets: Target indices (N,)
        output_embeddings: Output embedding matrix (V, d)

    Returns:
        Tuple of (min_margin, mean_margin)
    """
    with torch.no_grad():
        # Compute all similarities
        similarities = predictions @ output_embeddings.T  # (N, V)

        # Get correct class similarities
        batch_indices = torch.arange(len(targets), device=targets.device)
        correct_similarities = similarities[batch_indices, targets]  # (N,)

        # Mask out correct class and find max incorrect similarity
        mask = torch.ones_like(similarities, dtype=torch.bool)
        mask[batch_indices, targets] = False
        incorrect_similarities = similarities.clone()
        incorrect_similarities[~mask] = float("-inf")
        max_incorrect = incorrect_similarities.max(dim=-1).values  # (N,)

        # Compute margins
        margins = correct_similarities - max_incorrect

        return margins.min().item(), margins.mean().item()


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: The PyTorch model
        trainable_only: If True, only count trainable parameters

    Returns:
        Total parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def compute_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute mean squared error.

    Args:
        predictions: Predicted values
        targets: Target values

    Returns:
        MSE as a float
    """
    with torch.no_grad():
        return torch.nn.functional.mse_loss(predictions, targets).item()


def compute_coherence(embeddings: torch.Tensor) -> float:
    """
    Compute the coherence of embeddings.

    Coherence is the maximum absolute cosine similarity between any pair
    of embeddings, measuring how "aligned" the embeddings are.

    Args:
        embeddings: Embedding matrix (N, d)

    Returns:
        Coherence as a float
    """
    with torch.no_grad():
        # Normalize embeddings
        embeddings_norm = torch.nn.functional.normalize(embeddings, dim=-1)

        # Compute pairwise cosine similarities
        similarities = embeddings_norm @ embeddings_norm.T

        # Mask out diagonal
        n = similarities.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=similarities.device)
        pairwise_similarities = similarities[mask]

        return pairwise_similarities.abs().max().item()
