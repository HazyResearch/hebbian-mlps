"""
Synthetic factset generation.

This module provides utilities for generating synthetic fact datasets:
    - Bijective mappings (permutations, identity)
    - Factset generation with configurable parameters
"""

from dataclasses import dataclass
from typing import List, Optional, Literal, Union

import numpy as np
import torch

from hebbian.data.embeddings import generate_embeddings
from hebbian.mlp_core.exceptions import (
    InvalidEmbeddingError,
    InvalidMappingError,
)


@dataclass
class BijectiveMapping:
    """
    A bijective mapping from inputs to outputs.

    Attributes:
        inputs: List of input indices
        outputs: List of output indices (same length as inputs)
    """

    inputs: List[int]
    outputs: List[int]

    def __post_init__(self):
        if len(self.inputs) != len(self.outputs):
            raise ValueError("inputs and outputs must have the same length")
        # Build O(1) lookup dict
        self._input_to_pos: dict[int, int] = {inp: i for i, inp in enumerate(self.inputs)}

    def __len__(self) -> int:
        return len(self.inputs)

    def get_output(self, input_idx: int) -> int:
        """Get the output index for a given input index. O(1) lookup."""
        pos = self._input_to_pos.get(input_idx)
        if pos is None:
            raise KeyError(f"Input index {input_idx} not in mapping")
        return self.outputs[pos]

    @classmethod
    def from_permutation(cls, perm: List[int]) -> "BijectiveMapping":
        """Create mapping from a permutation array."""
        n = len(perm)
        return cls(inputs=list(range(n)), outputs=perm)


def create_identity_mapping(n: int) -> BijectiveMapping:
    """
    Create an identity mapping (output[i] = i).

    Args:
        n: Number of elements

    Returns:
        Identity BijectiveMapping
    """
    return BijectiveMapping(
        inputs=list(range(n)),
        outputs=list(range(n)),
    )


def create_random_permutation_mapping(
    n: int,
    seed: Optional[int] = None,
) -> BijectiveMapping:
    """
    Create a random permutation mapping.

    Args:
        n: Number of elements
        seed: Random seed for reproducibility

    Returns:
        Random permutation BijectiveMapping
    """
    if seed is not None:
        np.random.seed(seed)

    perm = np.random.permutation(n).tolist()
    return BijectiveMapping(
        inputs=list(range(n)),
        outputs=perm,
    )


@dataclass
class Factset:
    """
    A synthetic factset containing embeddings and mapping.

    Attributes:
        input_embeddings: Key embeddings (N, d)
        output_embeddings: Value embeddings (N, d) or (M, d)
        mapping: Bijective mapping from inputs to outputs
        d_model: Embedding dimension
        vocab_size: Number of facts
    """

    input_embeddings: torch.Tensor
    output_embeddings: torch.Tensor
    mapping: BijectiveMapping
    d_model: int
    vocab_size: int

    def __post_init__(self):
        """Validate the factset after initialization."""
        self.validate()

    def validate(self):
        """
        Validate that the factset is well-formed.

        Raises:
            InvalidEmbeddingError: If embeddings are invalid
            InvalidMappingError: If mapping is inconsistent with embeddings
        """
        # Validate input embeddings
        if not isinstance(self.input_embeddings, torch.Tensor):
            raise InvalidEmbeddingError(
                f"input_embeddings must be a torch.Tensor, got {type(self.input_embeddings)}"
            )

        if self.input_embeddings.ndim != 2:
            raise InvalidEmbeddingError(
                f"input_embeddings must be 2D (n, d), got shape {self.input_embeddings.shape}"
            )

        # Validate output embeddings
        if not isinstance(self.output_embeddings, torch.Tensor):
            raise InvalidEmbeddingError(
                f"output_embeddings must be a torch.Tensor, got {type(self.output_embeddings)}"
            )

        if self.output_embeddings.ndim != 2:
            raise InvalidEmbeddingError(
                f"output_embeddings must be 2D (n, d), got shape {self.output_embeddings.shape}"
            )

        # Validate shape consistency
        n_input, d_input = self.input_embeddings.shape
        n_output, d_output = self.output_embeddings.shape

        if d_input != d_output:
            raise InvalidEmbeddingError(
                f"Embedding dimensions must match: input has d={d_input}, output has d={d_output}"
            )

        # Validate mapping
        if not isinstance(self.mapping, BijectiveMapping):
            raise InvalidMappingError(
                f"mapping must be a BijectiveMapping instance, got {type(self.mapping)}"
            )

        # Validate mapping size consistency
        mapping_size = len(self.mapping)
        if mapping_size != n_input:
            raise InvalidMappingError(
                f"Mapping size ({mapping_size}) must match number of input embeddings ({n_input})"
            )

        # Additional checks for NaN/Inf values
        if torch.isnan(self.input_embeddings).any():
            raise InvalidEmbeddingError("input_embeddings contains NaN values")

        if torch.isinf(self.input_embeddings).any():
            raise InvalidEmbeddingError("input_embeddings contains Inf values")

        if torch.isnan(self.output_embeddings).any():
            raise InvalidEmbeddingError("output_embeddings contains NaN values")

        if torch.isinf(self.output_embeddings).any():
            raise InvalidEmbeddingError("output_embeddings contains Inf values")

    def to(
        self,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> "Factset":
        """
        Move factset to a different device and/or dtype.

        Args:
            device: Target device (e.g., 'cuda', 'cpu')
            dtype: Target dtype (e.g., torch.float32)

        Returns:
            New Factset with moved tensors
        """
        # Create new Factset without validation (since we know the original is valid)
        new_factset = object.__new__(Factset)
        new_factset.input_embeddings = self.input_embeddings.to(device=device, dtype=dtype)
        new_factset.output_embeddings = self.output_embeddings.to(device=device, dtype=dtype)
        new_factset.mapping = self.mapping
        new_factset.d_model = self.d_model
        new_factset.vocab_size = self.vocab_size
        return new_factset


def generate_factset(
    d_model: int,
    vocab_size: Optional[int] = None,
    facts_multiplier: float = 0.25,
    embedding_init: Literal["spherical", "gaussian", "uniform"] = "spherical",
    tie_embeddings: bool = True,
    mapping_type: Literal["identity", "random"] = "identity",
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> Factset:
    """
    Generate a synthetic factset.

    Args:
        d_model: Embedding dimension
        vocab_size: Number of facts (if None, computed as facts_multiplier * d_model^2)
        facts_multiplier: Multiplier for computing vocab_size from d_model
        embedding_init: Embedding initialization type
        tie_embeddings: Whether to use same embeddings for input and output
        mapping_type: Type of input-output mapping
        dtype: Data type for embeddings
        device: Device to place embeddings on
        seed: Random seed for reproducibility

    Returns:
        Factset with embeddings and mapping
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Compute vocab size if not specified
    if vocab_size is None:
        vocab_size = int(facts_multiplier * d_model * d_model)

    # Generate input embeddings
    input_embeddings = generate_embeddings(
        vocab_size, d_model, embedding_init, dtype, device
    )

    # Generate output embeddings
    if tie_embeddings:
        output_embeddings = input_embeddings
    else:
        output_embeddings = generate_embeddings(
            vocab_size, d_model, embedding_init, dtype, device
        )

    # Create mapping
    if mapping_type == "identity":
        mapping = create_identity_mapping(vocab_size)
    elif mapping_type == "random":
        mapping = create_random_permutation_mapping(vocab_size, seed)
    else:
        raise ValueError(f"Unknown mapping_type: {mapping_type}")

    # Create Factset without re-validation (we know generated data is valid)
    factset = object.__new__(Factset)
    factset.input_embeddings = input_embeddings
    factset.output_embeddings = output_embeddings
    factset.mapping = mapping
    factset.d_model = d_model
    factset.vocab_size = vocab_size
    return factset
