"""
I/O utilities for saving and loading experiment results.

This module provides utilities for:
    - Writing results to JSONL and CSV files
    - Saving and loading PyTorch checkpoints
    - Managing experiment artifacts
"""

import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(
    results: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = "json",
) -> None:
    """
    Save results to a file.

    Args:
        results: Dictionary of results to save
        filepath: Path to save to
        format: Output format ("json", "jsonl", or "csv")
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    if format == "json":
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
    elif format == "jsonl":
        with open(filepath, "a") as f:
            f.write(json.dumps(results, default=str) + "\n")
    elif format == "csv":
        # For CSV, results should be a flat dict or list of flat dicts
        if isinstance(results, dict):
            results = [results]
        if not filepath.exists():
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        else:
            with open(filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writerows(results)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json', 'jsonl', or 'csv'.")


def load_results(
    filepath: Union[str, Path],
    format: Optional[str] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load results from a file.

    Args:
        filepath: Path to load from
        format: File format (auto-detected from extension if not specified)

    Returns:
        Loaded results (dict for json, list of dicts for jsonl/csv)
    """
    filepath = Path(filepath)

    if format is None:
        ext = filepath.suffix.lower()
        format = {".json": "json", ".jsonl": "jsonl", ".csv": "csv"}.get(ext, "json")

    if format == "json":
        with open(filepath, "r") as f:
            return json.load(f)
    elif format == "jsonl":
        results = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    elif format == "csv":
        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_checkpoint(
    checkpoint: Dict[str, Any],
    filepath: Union[str, Path],
) -> None:
    """
    Save a PyTorch checkpoint.

    Args:
        checkpoint: Dictionary containing model state and metadata
        filepath: Path to save the checkpoint
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a PyTorch checkpoint.

    Args:
        filepath: Path to the checkpoint
        map_location: Device to map tensors to (e.g., "cpu", "cuda")

    Returns:
        Loaded checkpoint dictionary
    """
    return torch.load(filepath, map_location=map_location)


def save_config(
    config: Any,
    filepath: Union[str, Path],
) -> None:
    """
    Save a configuration object to a JSON file.

    Args:
        config: Configuration object (must have __dict__ or be a dict)
        filepath: Path to save to
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    if hasattr(config, "__dict__"):
        config_dict = config.__dict__
    else:
        config_dict = dict(config)

    with open(filepath, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a configuration from a JSON file.

    Args:
        filepath: Path to load from

    Returns:
        Configuration dictionary
    """
    with open(filepath, "r") as f:
        return json.load(f)
