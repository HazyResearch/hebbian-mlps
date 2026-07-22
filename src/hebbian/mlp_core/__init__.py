"""Self-contained MLP infrastructure for Hebbian constructions."""

from hebbian.mlp_core.task import MLPTask, SharedConstructionConfig, MLPConfig
from hebbian.mlp_core.mapping import Mapping, PrefixSuffixCorrespondence, SinglePrefixMapping
from hebbian.mlp_core.exceptions import (
    MLPError,
    MLPConstructionError,
    EncodingError,
    DecodingError,
    InvalidMappingError,
    InvalidEmbeddingError,
    InvalidConfigurationError,
    TrainingError,
    ConvergenceError,
    InvalidTaskError,
    BinningError,
    DeviceError,
)
from hebbian.mlp_core.utils import (
    move_config_to_device,
    extract_hidden_dimension,
    generate_mlp_architecture_string,
)
from hebbian.mlp_core.gd_training import (
    GDBatchConfig,
    GDOptimizerConfig,
    GDLogConfig,
    GDTrainingResult,
    train_with_gd,
)
from hebbian.mlp_core.mlp_gd import (
    GDMLPConfig,
    get_gd_mlp,
)

__all__ = [
    # Task
    "MLPTask",
    "SharedConstructionConfig",
    "MLPConfig",
    # Mapping
    "Mapping",
    "PrefixSuffixCorrespondence",
    "SinglePrefixMapping",
    # Exceptions
    "MLPError",
    "MLPConstructionError",
    "EncodingError",
    "DecodingError",
    "InvalidMappingError",
    "InvalidEmbeddingError",
    "InvalidConfigurationError",
    "TrainingError",
    "ConvergenceError",
    "InvalidTaskError",
    "BinningError",
    "DeviceError",
    # Utils
    "move_config_to_device",
    "extract_hidden_dimension",
    "generate_mlp_architecture_string",
    # GD Training
    "GDBatchConfig",
    "GDOptimizerConfig",
    "GDLogConfig",
    "GDTrainingResult",
    "train_with_gd",
    # GD MLP
    "GDMLPConfig",
    "get_gd_mlp",
]
