"""
Neural Tangent Kernel (NTK) MLP construction method.

This module wraps the existing NTKConstructionConfig and get_ntk_mlp from the mlps
codebase to provide a consistent Method interface.
"""

from typing import Any, Dict, Tuple, Optional
from dataclasses import field

import torch
from torch import nn
from hebbian.config import pydraclass

from hebbian.core.registry import Registry
from hebbian.methods.base import Method

# Import from hebbian.mlp_core
from hebbian.mlp_core.constructions.ntk import NTKConstructionConfig, get_ntk_mlp
from hebbian.mlp_core.task import SharedConstructionConfig
from hebbian.data.synthetics.factsets import Factset


@pydraclass
class NTKMethodConfig:
    """
    Configuration for NTK method.

    Wraps NTKConstructionConfig with additional method-level settings.
    """

    # Nested NTK config
    ntk_config: NTKConstructionConfig = field(default_factory=NTKConstructionConfig)

    # Method-level settings
    seed: int = 0
    verbose: bool = False


@Registry.register_method("ntk")
class NTKMethod(Method):
    """
    Neural Tangent Kernel (NTK) MLP construction method.

    This method constructs an MLP using NTK theory, computing weights
    directly without gradient descent training.

    Uses the existing mlps implementation under the hood.
    """

    def __init__(self, config: Optional[NTKMethodConfig] = None, seed: int = 0):
        """
        Initialize the NTK method.

        Args:
            config: NTKMethodConfig or None (uses defaults)
            seed: Random seed for reproducibility
        """
        if config is None:
            config = NTKMethodConfig()
        super().__init__(config, seed)
        self._ntk_config = config.ntk_config

    @property
    def method_id(self) -> str:
        """Return the method identifier."""
        return "ntk"

    def initialize(self, config: Any = None, seed: int = 0) -> None:
        """
        Initialize the method with configuration and seed.

        Args:
            config: NTKMethodConfig, NTKConstructionConfig, or dict
            seed: Random seed for reproducibility
        """
        self._seed = seed

        if config is not None:
            if isinstance(config, NTKMethodConfig):
                self._config = config
                self._ntk_config = config.ntk_config
            elif isinstance(config, NTKConstructionConfig):
                self._ntk_config = config
                self._config = NTKMethodConfig(ntk_config=config, seed=seed)
            elif isinstance(config, dict):
                # Build config from dict
                if "ntk_config" in config:
                    ntk_config = config["ntk_config"]
                    if isinstance(ntk_config, dict):
                        self._ntk_config = NTKConstructionConfig(**ntk_config)
                    else:
                        self._ntk_config = ntk_config
                else:
                    self._ntk_config = NTKConstructionConfig(**config)
                self._config = NTKMethodConfig(
                    ntk_config=self._ntk_config,
                    seed=seed,
                    verbose=config.get("verbose", False),
                )
            else:
                self._ntk_config = config
                self._config = NTKMethodConfig(ntk_config=config, seed=seed)

        self._initialized = True

    def fit_or_construct(
        self,
        factset: Factset,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Construct the MLP using NTK theory.

        Args:
            factset: Factset containing input_embeddings, output_embeddings, and mapping

        Returns:
            Tuple of (mlp, metrics_dict)
        """
        if not self._initialized:
            self.initialize(self._config, self._seed)

        # Set seed for reproducibility
        torch.manual_seed(self._seed)

        # Call the mlps implementation
        mlp, metrics = get_ntk_mlp(factset, self._ntk_config)

        # Compute standalone accuracy on the factset.
        with torch.no_grad():
            output = mlp(factset.input_embeddings)
            predictions = output @ factset.output_embeddings.T
            predicted_indices = torch.argmax(predictions, dim=-1)
            targets = torch.tensor(
                factset.mapping.outputs,
                dtype=torch.long,
                device=factset.input_embeddings.device,
            )
            accuracy = (predicted_indices == targets).float().mean().item()

        metrics["final_accuracy"] = accuracy  # key expected by extract_scalar_metrics
        metrics["accuracy"] = accuracy
        metrics["method"] = self.method_id
        metrics["hidden_dim"] = self._ntk_config.m
        metrics["param_count"] = self.param_count(mlp)

        return mlp, metrics

    def param_count(self, mlp: nn.Module) -> int:
        """
        Count the number of parameters in the MLP.

        Args:
            mlp: The MLP module

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in mlp.parameters())

    def get_hidden_dim(self, mlp: nn.Module) -> Optional[int]:
        """
        Extract the hidden dimension from the MLP.

        Args:
            mlp: The MLP module

        Returns:
            Hidden dimension (m) from config
        """
        return self._ntk_config.m

    @staticmethod
    def get_config_class() -> type:
        """Return the configuration class for this method."""
        return NTKMethodConfig

    @staticmethod
    def get_ntk_config_class() -> type:
        """Return the underlying NTK config class."""
        return NTKConstructionConfig
