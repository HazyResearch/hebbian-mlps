"""
Gradient Descent MLP method implementation.

This module wraps the existing GDMLPConfig and get_gd_mlp from the mlps
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
from hebbian.mlp_core.mlp_gd import GDMLPConfig, get_gd_mlp
from hebbian.mlp_core.task import SharedConstructionConfig
from hebbian.data.synthetics.factsets import Factset


def _coerce_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype_name = dtype.replace("torch.", "")
        if hasattr(torch, dtype_name):
            coerced = getattr(torch, dtype_name)
            if isinstance(coerced, torch.dtype):
                return coerced
    raise ValueError(f"Could not coerce dtype {dtype!r} to torch.dtype")


@pydraclass
class GDMLPMethodConfig:
    """
    Configuration for GD MLP method.

    Wraps GDMLPConfig with additional method-level settings.
    """

    # Nested GD config
    gd_config: GDMLPConfig = field(default_factory=GDMLPConfig)

    # Method-level settings
    seed: int = 0
    verbose: bool = False


@Registry.register_method("gd_mlp")
@Registry.register_method("gd")
class GDMLPMethod(Method):
    """
    Gradient Descent MLP construction method.

    This method trains an MLP using gradient descent to approximate
    the input-output mapping defined by the task.

    Uses the existing mlps implementation under the hood.
    """

    def __init__(self, config: Optional[GDMLPMethodConfig] = None, seed: int = 0):
        """
        Initialize the GD MLP method.

        Args:
            config: GDMLPMethodConfig or None (uses defaults)
            seed: Random seed for reproducibility
        """
        if config is None:
            config = GDMLPMethodConfig()
        super().__init__(config, seed)
        self._gd_config = config.gd_config

    @property
    def method_id(self) -> str:
        """Return the method identifier."""
        return "gd_mlp"

    def initialize(self, config: Any = None, seed: int = 0) -> None:
        """
        Initialize the method with configuration and seed.

        Args:
            config: GDMLPMethodConfig, GDMLPConfig, or dict
            seed: Random seed for reproducibility
        """
        self._seed = seed

        if config is not None:
            if isinstance(config, GDMLPMethodConfig):
                self._config = config
                self._gd_config = config.gd_config
            elif isinstance(config, GDMLPConfig):
                self._gd_config = config
                self._config = GDMLPMethodConfig(gd_config=config, seed=seed)
            elif isinstance(config, dict):
                config = dict(config)
                activation = config.pop("activation", None)
                build_dtype = config.pop("build_dtype", None)
                final_dtype = config.pop("final_dtype", None)
                # Build config from dict
                if "gd_config" in config:
                    gd_config = config["gd_config"]
                    if isinstance(gd_config, dict):
                        self._gd_config = GDMLPConfig(**gd_config)
                    else:
                        self._gd_config = gd_config
                else:
                    self._gd_config = GDMLPConfig(**config)
                if activation is not None:
                    self._gd_config.shared.mlp_config.activation.activation = activation
                if build_dtype is not None:
                    self._gd_config.shared.build_dtype = _coerce_dtype(build_dtype)
                if final_dtype is not None:
                    self._gd_config.shared.final_dtype = _coerce_dtype(final_dtype)
                self._config = GDMLPMethodConfig(
                    gd_config=self._gd_config,
                    seed=seed,
                    verbose=config.get("verbose", False),
                )
            else:
                self._gd_config = config
                self._config = GDMLPMethodConfig(gd_config=config, seed=seed)

        self._initialized = True

    def fit_or_construct(
        self,
        factset: Factset,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Train the MLP using gradient descent.

        Args:
            factset: Factset containing input_embeddings, output_embeddings, and mapping

        Returns:
            Tuple of (trained_mlp, metrics_dict)
        """
        if not self._initialized:
            self.initialize(self._config, self._seed)

        # Set seed for reproducibility
        torch.manual_seed(self._seed)

        # Call the mlps implementation
        mlp, metrics = get_gd_mlp(factset, self._gd_config)

        # Extract additional metrics
        metrics["method"] = self.method_id
        metrics["hidden_dim"] = self.get_hidden_dim(mlp)
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
            Hidden dimension if extractable
        """
        try:
            if hasattr(mlp, "up") and hasattr(mlp.up, "out_dim"):
                return mlp.up.out_dim
            # Fallback to base implementation
            return super().get_hidden_dim(mlp)
        except Exception:
            return None

    @staticmethod
    def get_config_class() -> type:
        """Return the configuration class for this method."""
        return GDMLPMethodConfig

    @staticmethod
    def get_gd_config_class() -> type:
        """Return the underlying GD config class."""
        return GDMLPConfig
