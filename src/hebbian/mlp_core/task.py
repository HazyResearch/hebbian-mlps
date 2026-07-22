"""Task specification for MLP construction and training."""

from typing import Optional, Union
from dataclasses import dataclass, field
import torch

from hebbian.config import pydraclass
from hebbian.mlp_core.utils import move_config_to_device
from hebbian.mlp_core.mapping import Mapping
from hebbian.mlp_core.exceptions import (
    InvalidEmbeddingError,
    InvalidMappingError,
)
from hebbian.mlp_core.blocks.activations import ActivationConfig


@pydraclass
class MLPConfig:
    """Configuration for the gated MLP used in the paper."""

    activation: ActivationConfig = field(default_factory=ActivationConfig)


@pydraclass
class SharedConstructionConfig:
    """
    Parameters shared across many parts of the construction pipeline.
    """

    build_dtype: torch.dtype = torch.float64
    final_dtype: Optional[torch.dtype] = None

    device: Optional[Union[str, torch.device]] = None
    verbose: bool = True
    rtol: float = 1e-4
    atol: float = 1e-4
    strict_assertions: bool = True
    mlp_config: MLPConfig = field(default_factory=MLPConfig)

    def to(
        self,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> "SharedConstructionConfig":
        """Create a copy of this config with all tensors moved to the specified device and/or dtype."""
        return move_config_to_device(self, device=device, dtype=dtype)

    def custom_finalize(self):
        if self.final_dtype is None:
            self.final_dtype = self.build_dtype


@dataclass
class MLPTask:
    """
    Task specification for MLP construction and training.

    Attributes:
        input_embeddings: Input embedding matrix of shape (n, d)
        output_embeddings: Output embedding matrix of shape (n, d)
        mapping: Mapping from input indices to output indices
    """

    input_embeddings: torch.Tensor
    output_embeddings: torch.Tensor
    mapping: Mapping

    def __post_init__(self):
        """Validate the task after initialization."""
        self.validate()

    def validate(self):
        """
        Validate that the task is well-formed.

        Raises:
            InvalidEmbeddingError: If embeddings are invalid
            InvalidMappingError: If mapping is inconsistent with embeddings
            InvalidTaskError: If there are other consistency issues
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
        if not isinstance(self.mapping, Mapping):
            raise InvalidMappingError(
                f"mapping must be a Mapping instance, got {type(self.mapping)}"
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
    ) -> "MLPTask":
        """
        Move task to a different device and/or dtype.

        Args:
            device: Target device (e.g., 'cuda', 'cpu')
            dtype: Target dtype (e.g., torch.float32)

        Returns:
            New MLPTask with moved tensors
        """
        return MLPTask(
            input_embeddings=self.input_embeddings.to(device=device, dtype=dtype),
            output_embeddings=self.output_embeddings.to(device=device, dtype=dtype),
            mapping=self.mapping,
        )
