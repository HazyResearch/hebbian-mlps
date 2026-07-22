"""
Embedding generators.

This module provides utilities for generating embeddings:
    - Spherical (uniform on unit sphere)
    - Gaussian (standard normal)
    - Uniform (uniform in unit cube)
    - Kaiming uniform (PyTorch embedding default)
"""

import math
from typing import Literal, Optional

import torch


def generate_embeddings(
    n: int,
    d: int,
    init_type: Literal["spherical", "gaussian", "uniform", "kaiming_uniform"] = "spherical",
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate embeddings with the specified initialization.

    Args:
        n: Number of embeddings (vocabulary size)
        d: Embedding dimension
        init_type: Type of initialization
        dtype: Data type for embeddings
        device: Device to place embeddings on
        seed: Random seed for reproducibility

    Returns:
        Embedding tensor of shape (n, d)
    """
    if seed is not None:
        torch.manual_seed(seed)

    if init_type == "spherical":
        return generate_spherical_embeddings(n, d, dtype, device)
    elif init_type == "gaussian":
        return generate_gaussian_embeddings(n, d, dtype, device)
    elif init_type == "uniform":
        return generate_uniform_embeddings(n, d, dtype, device)
    elif init_type == "kaiming_uniform":
        return generate_kaiming_uniform_embeddings(n, d, dtype, device)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")


def generate_spherical_embeddings(
    n: int,
    d: int,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate embeddings uniformly distributed on the unit sphere.

    Args:
        n: Number of embeddings
        d: Embedding dimension
        dtype: Data type
        device: Device

    Returns:
        Normalized embeddings of shape (n, d)
    """
    embeddings = torch.randn(n, d, dtype=dtype, device=device)
    return torch.nn.functional.normalize(embeddings, dim=1)


def generate_gaussian_embeddings(
    n: int,
    d: int,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate embeddings from standard normal distribution.

    Args:
        n: Number of embeddings
        d: Embedding dimension
        dtype: Data type
        device: Device

    Returns:
        Gaussian embeddings of shape (n, d)
    """
    return torch.randn(n, d, dtype=dtype, device=device)


def generate_uniform_embeddings(
    n: int,
    d: int,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate embeddings uniformly distributed in the unit cube.

    Args:
        n: Number of embeddings
        d: Embedding dimension
        dtype: Data type
        device: Device

    Returns:
        Uniform embeddings of shape (n, d) in range [0, 1]
    """
    return torch.rand(n, d, dtype=dtype, device=device)


def generate_kaiming_uniform_embeddings(
    n: int,
    d: int,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate embeddings with PyTorch's default Kaiming-uniform embedding init."""
    embeddings = torch.empty(n, d, dtype=dtype, device=device)
    torch.nn.init.kaiming_uniform_(embeddings, a=math.sqrt(5))
    return embeddings


def generate_tied_embeddings(
    n: int,
    d: int,
    init_type: Literal["spherical", "gaussian", "uniform", "kaiming_uniform"] = "spherical",
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> tuple:
    """
    Generate tied key and value embeddings (K = V).

    Args:
        n: Number of embeddings
        d: Embedding dimension
        init_type: Type of initialization
        dtype: Data type
        device: Device
        seed: Random seed

    Returns:
        Tuple of (input_embeddings, output_embeddings) where they are the same
    """
    embeddings = generate_embeddings(n, d, init_type, dtype, device, seed)
    return embeddings, embeddings


def generate_untied_embeddings(
    n: int,
    d: int,
    init_type: Literal["spherical", "gaussian", "uniform"] = "spherical",
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> tuple:
    """
    Generate untied key and value embeddings (K != V).

    Args:
        n: Number of embeddings
        d: Embedding dimension
        init_type: Type of initialization
        dtype: Data type
        device: Device
        seed: Random seed

    Returns:
        Tuple of (input_embeddings, output_embeddings)
    """
    if seed is not None:
        torch.manual_seed(seed)

    input_embeddings = generate_embeddings(n, d, init_type, dtype, device)

    # Use different random state for output
    output_embeddings = generate_embeddings(n, d, init_type, dtype, device)

    return input_embeddings, output_embeddings
