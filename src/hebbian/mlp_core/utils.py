"""Utility functions for MLP construction and configuration."""

import torch
import math
import traceback
from copy import deepcopy
from typing import Union, TypeVar, Tuple, Callable, Optional, List, Any
from dataclasses import dataclass, field

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

T = TypeVar("T")


# ============================================================================
# Retry utilities
# ============================================================================


@dataclass
class RetryResult:
    """Result from retry attempts with full context."""

    success: bool
    result: Any = None
    attempts: List[Tuple[int, str, str]] = field(
        default_factory=list
    )  # (num, exc_type, exc_msg)
    num_attempts: int = 0


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 1000,
        verbose: bool = True,
        show_progress: bool = True,
        traceback_frequency: int = 100,
        operation_name: str = "Operation",
    ):
        self.max_attempts = max_attempts
        self.verbose = verbose
        self.show_progress = show_progress
        self.traceback_frequency = traceback_frequency
        self.operation_name = operation_name

    def get_traceback_frequency(self) -> int:
        """Get the actual traceback frequency (handle percentage-based)."""
        if self.traceback_frequency <= 0:
            return 0
        # If traceback_frequency looks like it might be a percentage threshold
        if self.traceback_frequency < 1:
            return max(1, self.max_attempts // 100)
        return self.traceback_frequency


def retry_with_logging(
    operation: Callable[[], T],
    config: RetryConfig,
    extra_info_fn: Optional[Callable[[int, Exception], str]] = None,
    result_validator: Optional[Callable[[T], bool]] = None,
) -> Optional[T]:
    """
    Retry an operation multiple times with configurable logging.

    Args:
        operation: Callable that performs the operation and returns a result
        config: RetryConfig specifying retry behavior
        extra_info_fn: Optional function that returns extra info string given (attempt_num, exception)
        result_validator: Optional function that validates the result.
                         Returns True if result is valid, False to retry.
                         If None, any non-None result is considered valid.

    Returns:
        Result of operation if successful, None if all attempts failed
    """
    # Setup progress bar if requested
    iterator = range(config.max_attempts)
    if config.show_progress and tqdm is not None:
        iterator = tqdm(iterator, disable=not config.verbose)

    traceback_freq = config.get_traceback_frequency()

    for i in iterator:
        try:
            result = operation()

            # Validate result if validator provided
            if result_validator is not None:
                if not result_validator(result):
                    if config.verbose:
                        print(
                            f"{config.operation_name} returned invalid result (attempt {i+1}/{config.max_attempts})"
                        )
                    continue
            elif result is None:
                # Default: treat None as failure
                if config.verbose:
                    print(
                        f"{config.operation_name} returned None (attempt {i+1}/{config.max_attempts})"
                    )
                continue

            # Success!
            if config.verbose and i > 0:
                print(f"✓ {config.operation_name} succeeded on attempt {i+1}")
            return result

        except Exception as e:
            if config.verbose:
                print(
                    f"{config.operation_name} failed (attempt {i+1}/{config.max_attempts}): "
                    f"{type(e).__name__}: {e}"
                )

                # Print full traceback at specified frequency
                if traceback_freq > 0 and (i % traceback_freq == 0):
                    print("Full traceback:")
                    traceback.print_exc()

                    # Print extra debugging info if provided
                    if extra_info_fn is not None:
                        extra_info = extra_info_fn(i, e)
                        if extra_info:
                            print(extra_info)

            continue

    if config.verbose:
        print(f"{config.operation_name} failed after {config.max_attempts} attempts")

    # All attempts failed
    return None


def retry_with_logging_simple(
    operation: Callable[[], T],
    max_attempts: int = 1000,
    verbose: bool = True,
    show_progress: bool = True,
    operation_name: str = "Operation",
) -> Optional[T]:
    """
    Simplified version of retry_with_logging with common defaults.

    Args:
        operation: Callable that performs the operation and returns a result
        max_attempts: Maximum number of attempts
        verbose: Whether to print verbose logging
        show_progress: Whether to show tqdm progress bar
        operation_name: Name of operation for logging

    Returns:
        Result of operation if successful, None if all attempts failed
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        verbose=verbose,
        show_progress=show_progress,
        operation_name=operation_name,
    )

    return retry_with_logging(operation, config)


def move_config_to_device(
    config: T,
    device: Union[str, torch.device, None] = None,
    dtype: Union[torch.dtype, None] = None,
) -> T:
    """
    Create a copy of a config with all tensors moved to the specified device and/or dtype.

    This matches PyTorch's .to() behavior - it returns a NEW object, not modifying in-place.

    This function recursively handles:
    - torch.Tensor objects (moved to device/dtype)
    - Objects with their own .to() method (called with device/dtype)
    - Lists and tuples (recursively processed)
    - Dicts (recursively processed)
    - All other objects (copied as-is)

    Args:
        config: The config object to copy and move to device/dtype
        device: Target device (e.g., 'cpu', 'cuda', 'cuda:0'), or None to keep current device
        dtype: Target dtype (e.g., torch.float32, torch.float64), or None to keep current dtype

    Returns:
        A new config object with all tensors moved to the specified device/dtype
    """

    def move_to_device(obj):
        """Recursively move tensors in an object to the specified device/dtype."""
        if isinstance(obj, torch.Tensor):
            # Move tensor to device and/or dtype (returns new tensor)
            return obj.to(device=device, dtype=dtype)
        elif hasattr(obj, "to") and callable(obj.to) and obj is not config:
            # Handle objects with their own to() method (like nested configs)
            # Avoid infinite recursion by checking obj is not config
            # This should return a new object
            return obj.to(device=device, dtype=dtype)
        elif isinstance(obj, (list, tuple)):
            # Return new list/tuple with moved items
            return type(obj)(move_to_device(item) for item in obj)
        elif isinstance(obj, dict):
            # Return new dict with moved values
            return {k: move_to_device(v) for k, v in obj.items()}
        else:
            # Return non-tensor objects as-is
            return obj

    # Deep copy the config first
    new_config = deepcopy(config)

    # Recursively move tensors in the copy
    for attr_name in list(new_config.__dict__.keys()):
        attr_value = getattr(new_config, attr_name)
        setattr(new_config, attr_name, move_to_device(attr_value))

    # Update the device and dtype attributes if they exist
    if device is not None and hasattr(new_config, "device"):
        new_config.device = device
    if dtype is not None:
        if hasattr(new_config, "build_dtype"):
            new_config.build_dtype = dtype
        if hasattr(new_config, "final_dtype"):
            new_config.final_dtype = dtype
        # Some configuration types expose a single dtype field.
        if hasattr(new_config, "dtype"):
            new_config.dtype = dtype

    return new_config


def extract_hidden_dimension(mlp) -> int:
    """
    Extract the actual hidden dimension from a constructed MLP.

    For constructed MLPs, the structure is typically:
    - First layer: input_dim -> hidden_dim (with ReLU)
    - Second layer: hidden_dim -> output_dim

    We extract the hidden dimension from the first layer's output size.

    Args:
        mlp: The MLP block to extract the hidden dimension from

    Returns:
        The hidden dimension size

    Raises:
        ValueError: If the hidden dimension cannot be extracted
    """
    # Handle different MLP structures
    if hasattr(mlp, "linear") and hasattr(mlp.linear, "weight"):
        # Single LinearBlock
        return mlp.linear.weight.shape[0]

    if hasattr(mlp, "up") and hasattr(mlp.up, "out_dim"):
        return mlp.up.out_dim

    # Fallback: search through all parameters
    for name, param in mlp.named_parameters():
        if param.ndim == 2 and "weight" in name:
            return param.shape[0]

    raise ValueError("Could not extract hidden dimension from MLP structure")


def generate_mlp_architecture_string(
    mlp,
    input_size: int,
    output_size: int,
    hidden_dim: int = None,
    fallback_m: int = None,
    fallback_num_embds: int = None,
    fallback_d_model: int = None,
    fallback_multiplier: float = 1.0,
    verbose: bool = False,
) -> Tuple[str, int]:
    """
    Generate an architecture string describing the MLP structure.

    Args:
        mlp: The MLP to describe
        input_size: Input dimension
        output_size: Output dimension
        hidden_dim: Hidden dimension (if None, will be extracted)
        fallback_m: For fallback calculation if extraction fails
        fallback_num_embds: For fallback calculation if extraction fails
        fallback_d_model: For fallback calculation if extraction fails
        fallback_multiplier: For fallback calculation if extraction fails
        verbose: Whether to print debug information

    Returns:
        Tuple of (architecture_string, actual_hidden_dim)
    """
    # Try to extract hidden dimension if not provided
    if hidden_dim is None:
        try:
            if verbose:
                print("DEBUGGING: trying to extract actual hidden dimension")
            actual_hidden_dim = extract_hidden_dimension(mlp)
        except Exception as e:
            # Fallback to formula-based calculation if extraction fails
            if verbose:
                print("DEBUGGING: extraction failed, using formula-based calculation")
            if (
                fallback_m is not None
                and fallback_num_embds is not None
                and fallback_d_model is not None
            ):
                actual_hidden_dim = math.ceil(
                    fallback_m
                    * (fallback_num_embds / fallback_d_model)
                    * fallback_multiplier
                )
            else:
                raise ValueError(
                    "Could not extract hidden dimension and insufficient fallback parameters provided"
                ) from e
    else:
        actual_hidden_dim = hidden_dim

    # Generate architecture string
    total_params = sum(p.numel() for p in mlp.parameters())

    lines = []
    lines.append(f"MLP: {input_size}->{output_size}")
    lines.append(f"  Linear: {input_size}->{actual_hidden_dim} [B,{actual_hidden_dim}]")
    lines.append(
        f"  relu: {actual_hidden_dim}->{actual_hidden_dim} [B,{actual_hidden_dim}]"
    )
    lines.append(f"  Linear: {actual_hidden_dim}->{output_size} [B,{output_size}]")
    lines.append(f"  hidden_dim: {actual_hidden_dim}")
    lines.append(f"  Total parameters: {total_params}")

    architecture_str = "\n".join(lines)

    return architecture_str, actual_hidden_dim
