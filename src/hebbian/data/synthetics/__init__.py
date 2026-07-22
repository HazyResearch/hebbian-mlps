"""
Synthetic data generation utilities.

This module provides:
    - Factset generation (bijective key-value mappings)
    - Regime templates for different experimental settings
"""

from hebbian.data.synthetics.factsets import (
    generate_factset,
    BijectiveMapping,
    create_identity_mapping,
    create_random_permutation_mapping,
)

__all__ = [
    "generate_factset",
    "BijectiveMapping",
    "create_identity_mapping",
    "create_random_permutation_mapping",
]
