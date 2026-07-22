"""
Data type utilities.

This module provides utilities for dtype handling:
    - get_dtype: Convert dtype specification to torch.dtype
"""

from typing import Union

import torch


def get_dtype(dtype_spec: Union[str, torch.dtype, None]) -> torch.dtype:
    """
    Convert dtype specification to torch.dtype.

    Args:
        dtype_spec: One of:
            - None: Returns torch.float64 (default)
            - torch.dtype: Returns as-is
            - str: One of "float16", "float32", "float64", "bfloat16"

    Returns:
        Corresponding torch.dtype

    Raises:
        ValueError: If dtype_spec is not a recognized type or string
    """
    if dtype_spec is None:
        return torch.float64

    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec

    if isinstance(dtype_spec, str):
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
        }
        if dtype_spec in dtype_map:
            return dtype_map[dtype_spec]
        raise ValueError(
            f"Unknown dtype string: {dtype_spec}. "
            f"Valid options: {list(dtype_map.keys())}"
        )

    raise ValueError(f"Cannot convert {type(dtype_spec)} to torch.dtype")
