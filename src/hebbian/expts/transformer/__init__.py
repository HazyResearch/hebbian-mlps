"""Transformer capacity experiment entry points."""

from hebbian.expts.transformer.config import (
    DEFAULT_RELEASE_METHODS,
    HIDDEN_DIM_PRESETS,
    NUM_FACTS_PRESETS,
    HiddenDimConfig,
    NumFactsConfig,
    resolve_hidden_dim_config,
    resolve_num_facts_config,
)

__all__ = [
    "DEFAULT_RELEASE_METHODS",
    "HIDDEN_DIM_PRESETS",
    "NUM_FACTS_PRESETS",
    "HiddenDimConfig",
    "NumFactsConfig",
    "resolve_hidden_dim_config",
    "resolve_num_facts_config",
]
