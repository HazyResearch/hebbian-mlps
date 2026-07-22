"""
Data utilities for Hebbian experiments.

This module provides:
    - Embedding generators (isotropic, anisotropic, tied K=V)
    - Embedding transforms (normalization, conditioning, whitening)
    - Synthetic factset generation (bijective maps, regime templates)
"""

from hebbian.data.embeddings import (
    generate_embeddings,
    normalize_embeddings,
    condition_embeddings,
    spike_embeddings,
)
from hebbian.data.synthetics import (
    generate_factset,
    BijectiveMapping,
)

__all__ = [
    "generate_embeddings",
    "normalize_embeddings",
    "condition_embeddings",
    "spike_embeddings",
    "generate_factset",
    "BijectiveMapping",
]
