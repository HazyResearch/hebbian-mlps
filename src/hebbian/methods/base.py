"""
Base interface for MLP construction methods.

This module defines the abstract base class that all MLP construction
methods must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional

import torch
from torch import nn


class Method(ABC):
    """
    Abstract base class for MLP construction methods.

    All MLP construction methods (GD, NTK, Hebbian) must inherit from this
    class and implement its abstract methods.

    Attributes:
        config: Method-specific configuration
        seed: Random seed for reproducibility
    """

    def __init__(self, config: Any = None, seed: int = 0):
        """
        Initialize the method.

        Args:
            config: Method-specific configuration object
            seed: Random seed for reproducibility
        """
        self._config = config
        self._seed = seed
        self._initialized = False

    @property
    @abstractmethod
    def method_id(self) -> str:
        """
        Return a unique identifier for this method.

        Returns:
            String identifier (e.g., "gd", "ntk", "hebbian")
        """
        ...

    @property
    def config(self) -> Any:
        """Return the method configuration."""
        return self._config

    @config.setter
    def config(self, value: Any) -> None:
        """Set the method configuration."""
        self._config = value

    @property
    def seed(self) -> int:
        """Return the random seed."""
        return self._seed

    @abstractmethod
    def initialize(self, config: Any = None, seed: int = 0) -> None:
        """
        Initialize the method with configuration and seed.

        This method should be called before fit_or_construct to set up
        any method-specific state.

        Args:
            config: Method-specific configuration object
            seed: Random seed for reproducibility
        """
        ...

    @abstractmethod
    def fit_or_construct(
        self,
        task: Any,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Fit or construct the MLP for the given factset.

        For GD methods, this trains the MLP. For construction methods
        (NTK, Hebbian), this directly constructs the MLP weights.

        Args:
            factset: Factset containing input_embeddings, output_embeddings, and mapping

        Returns:
            Tuple of (mlp, metrics_dict) where:
                - mlp: The constructed nn.Module
                - metrics_dict: Dictionary of metrics (accuracy, loss, etc.)
        """
        ...

    @abstractmethod
    def param_count(self, mlp: nn.Module) -> int:
        """
        Count the number of parameters in the MLP.

        Args:
            mlp: The MLP module

        Returns:
            Total number of parameters
        """
        ...

    def get_hidden_dim(self, mlp: nn.Module) -> Optional[int]:
        """
        Extract the hidden dimension from an MLP.

        Args:
            mlp: The MLP module

        Returns:
            Hidden dimension if extractable, None otherwise
        """
        try:
            # Try to extract from common MLP structures
            if hasattr(mlp, "hidden_dim"):
                return mlp.hidden_dim
            # Look for first linear layer output features
            for module in mlp.modules():
                if isinstance(module, nn.Linear):
                    return module.out_features
        except Exception:
            pass
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method_id={self.method_id!r})"
