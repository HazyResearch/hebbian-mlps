"""Registry adapter and paper-facing Hebbian configuration."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

import torch

from hebbian.config import pydraclass
from hebbian.core.registry import Registry
from hebbian.data.synthetics.factsets import Factset
from hebbian.methods.base import Method
from hebbian.mlp_core.task import SharedConstructionConfig

from .construction import construct_hebbian_mlp
from .model import HebbianMLP


@pydraclass
class HebbianConfig:
    """Configuration for the three Hebbian constructions in the paper."""

    variant: Literal["unwhitened", "whitened", "data_dependent"] = "unwhitened"
    m: int | None = None
    ridge: float = 1e-6
    shared: SharedConstructionConfig = field(default_factory=SharedConstructionConfig)


@pydraclass
class HebbianMethodConfig:
    hebbian_config: HebbianConfig = field(default_factory=HebbianConfig)
    seed: int = 0
    verbose: bool = False


@Registry.register_method("hebbian")
class HebbianMethod(Method):
    """Construct a paper Hebbian MLP without gradient descent."""

    def __init__(self, config: HebbianMethodConfig | None = None, seed: int = 0):
        config = config or HebbianMethodConfig()
        super().__init__(config, seed)
        self._hebbian_config = config.hebbian_config

    @property
    def method_id(self) -> str:
        return f"hebbian_{self._hebbian_config.variant}"

    def initialize(self, config: Any = None, seed: int = 0) -> None:
        self._seed = int(seed)
        verbose = False
        if config is None:
            hebbian_config = HebbianConfig()
        elif isinstance(config, HebbianMethodConfig):
            self._config = config
            self._hebbian_config = config.hebbian_config
            self._seed = int(config.seed if seed == 0 else seed)
            self._initialized = True
            return
        elif isinstance(config, HebbianConfig):
            hebbian_config = config
        elif isinstance(config, dict):
            values = dict(config)
            verbose = bool(values.pop("verbose", False))
            nested = values.pop("hebbian_config", None)
            if nested is not None:
                if values:
                    raise ValueError(
                        "Top-level Hebbian options cannot accompany hebbian_config"
                    )
                hebbian_config = (
                    HebbianConfig(**nested) if isinstance(nested, dict) else nested
                )
            else:
                hebbian_config = HebbianConfig(**values)
        else:
            raise TypeError(f"Unsupported Hebbian config type: {type(config)}")

        if hebbian_config.ridge < 0:
            raise ValueError(f"ridge must be non-negative; got {hebbian_config.ridge}")
        self._hebbian_config = hebbian_config
        self._config = HebbianMethodConfig(
            hebbian_config=hebbian_config,
            seed=self._seed,
            verbose=verbose,
        )
        self._initialized = True

    def fit_or_construct(
        self,
        factset: Factset,
    ) -> tuple[HebbianMLP, dict[str, Any]]:
        if not self._initialized:
            self.initialize(self._config, self._seed)
        mlp, metrics = construct_hebbian_mlp(
            factset, self._hebbian_config, seed=self._seed
        )
        metrics.update(
            {
                "method": self.method_id,
                "variant": self._hebbian_config.variant,
                "hidden_dim": mlp.feature_map.out_dim,
                "param_count": self.param_count(mlp),
            }
        )
        return mlp, metrics

    def param_count(self, mlp: torch.nn.Module) -> int:
        if isinstance(mlp, HebbianMLP):
            return mlp.weight_count()
        return sum(p.numel() for p in mlp.parameters()) + sum(
            b.numel() for b in mlp.buffers()
        )

    def get_hidden_dim(self, mlp: torch.nn.Module) -> int | None:
        if isinstance(mlp, HebbianMLP):
            return mlp.feature_map.out_dim
        return self._hebbian_config.m

    @staticmethod
    def get_config_class() -> type:
        return HebbianMethodConfig

    @staticmethod
    def get_hebbian_config_class() -> type:
        return HebbianConfig
