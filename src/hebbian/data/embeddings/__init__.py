"""
Embedding generation and transformation utilities.

This module provides:
    - Embedding generators (spherical, Gaussian, uniform)
    - Embedding transforms (normalization, conditioning, spiked covariance)
"""

from hebbian.data.embeddings.generators import (
    generate_embeddings,
    generate_spherical_embeddings,
    generate_gaussian_embeddings,
    generate_uniform_embeddings,
)
from hebbian.data.embeddings.transforms import (
    normalize_embeddings,
    condition_embeddings,
    spike_embeddings,
)

__all__ = [
    "generate_embeddings",
    "generate_spherical_embeddings",
    "generate_gaussian_embeddings",
    "generate_uniform_embeddings",
    "normalize_embeddings",
    "condition_embeddings",
    "spike_embeddings",
]
