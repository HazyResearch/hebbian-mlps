"""
Transformer training infrastructure for associative-recall experiments.

Simplified from mlps repo — removes unused knobs and complexity while
keeping all functionality needed for synthetic scaling experiments.
"""

from hebbian.transformer.model import BinaryMoE, BinaryRouter, GPT, GPTConfig, LowRankLinear
from hebbian.transformer.config import (
    AssociativeRecallConfig,
    DatasetConfig,
    TrainingConfig,
)
from hebbian.transformer.train import train_associative_recall
from hebbian.transformer.train_attention import train_attention
from hebbian.transformer.utils import evaluate, compute_weight_norms

__all__ = [
    "GPT",
    "GPTConfig",
    "BinaryMoE",
    "BinaryRouter",
    "LowRankLinear",
    "AssociativeRecallConfig",
    "TrainingConfig",
    "DatasetConfig",
    "train_associative_recall",
    "train_attention",
    "evaluate",
    "compute_weight_norms",
]
