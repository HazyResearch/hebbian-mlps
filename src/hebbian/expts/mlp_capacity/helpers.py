"""Helpers for Expt 4: MLP-only fact storage binary search over hidden_dim (m)."""

from __future__ import annotations

import os
from typing import Any, Dict, Literal, Optional

import torch


def run_mlp_experiment(
    d_model: int,
    num_facts: int,
    m: int,
    method_label: str,
    method_spec: Dict[str, Any],
    device: str,
    seed: int,
    mapping_type: Literal["identity", "random"] = "random",
    facts_multiplier: Optional[float] = None,
    embedding_init: str = "spherical",
    tie_embeddings: bool = True,
    spike_beta: float = 0.0,
    spike_target: Literal["keys", "values", "both"] = "both",
    spike_seed: int = 42,
    embeddings_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single MLP-only fact-storage experiment.

    Args:
        d_model: Embedding dimension.
        num_facts: Vocabulary size (number of facts to store).
        m: Hidden dimension to evaluate.
        method_label: Human-readable method name (e.g. "hebbian_whitened").
        method_spec: Method spec dict from _METHOD_SPECS (mlp_method, mlp_dtype,
            mlp_method_kwargs).
        device: Torch device string (e.g. "cuda", "cpu").
        seed: Random seed (used for factset generation reproducibility).
        mapping_type: Input-output map type for the synthetic factset.
        facts_multiplier: Optional alpha = F/d^2 metadata passed by the sweep
            config generator. Not used directly by the experiment run.
        embedding_init: Base embedding initialization for synthetic keys/values.
        tie_embeddings: Whether input/output embeddings share the same base table.
        spike_beta: Rank-1 spike strength used to make embeddings anisotropic.
        spike_target: Which embeddings to spike: "keys", "values", or "both".
        spike_seed: Seed used to sample the shared spike direction.
        embeddings_dir: Optional LLM activation bundle root or direct
            ``activations/`` directory. When set, paired activation rows replace
            synthetic factset generation.

    Returns:
        dict with keys: best_acc, m, method, param_count, success.
    """
    from hebbian.data.embeddings.transforms import spike_embeddings
    from hebbian.data.synthetics import generate_factset

    mlp_method: str = method_spec["mlp_method"]
    mlp_dtype: Optional[torch.dtype] = method_spec.get("mlp_dtype")
    mlp_method_kwargs: Optional[Dict[str, Any]] = method_spec.get("mlp_method_kwargs") or {}

    # Determine dtype: use method-specific dtype or float32 for GD.
    dtype = mlp_dtype if mlp_dtype is not None else torch.float32

    # Set seed for reproducibility.
    torch.manual_seed(seed)

    if embeddings_dir is None:
        # Generate factset.
        factset = generate_factset(
            d_model=d_model,
            vocab_size=num_facts,
            embedding_init=embedding_init,
            tie_embeddings=tie_embeddings,
            mapping_type=mapping_type,
        )
        factset = factset.to(device=device, dtype=dtype)
    else:
        from hebbian.expts.llm_embeddings.bundle import make_factset_from_activation_rows

        factset = make_factset_from_activation_rows(
            embeddings_dir,
            num_facts=num_facts,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        d_model = int(factset.d_model)

    if embeddings_dir is not None and spike_beta > 0:
        raise ValueError("spike_beta is synthetic-only when embeddings_dir is set")

    if spike_beta > 0:
        if spike_target not in ("keys", "values", "both"):
            raise ValueError(
                f"spike_target must be one of ('keys', 'values', 'both'); got {spike_target!r}"
            )

        if spike_target in ("keys", "both"):
            factset.input_embeddings = spike_embeddings(
                factset.input_embeddings,
                beta=spike_beta,
                seed=spike_seed,
                normalize=True,
            )

        if spike_target in ("values", "both"):
            factset.output_embeddings = spike_embeddings(
                factset.output_embeddings,
                beta=spike_beta,
                seed=spike_seed,
                normalize=True,
            )

    # Optional GD epoch override for shorter runs.
    gd_num_epochs: Optional[int] = method_spec.get("gd_num_epochs", None)

    # Dispatch to method-specific construction.
    if mlp_method == "gd":
        mlp, metrics = _run_gd(factset, m, dtype, device, num_epochs=gd_num_epochs)
    elif mlp_method == "hebbian":
        mlp, metrics = _run_hebbian(factset, m, mlp_method_kwargs, dtype, device)
    elif mlp_method == "ntk":
        mlp, metrics = _run_ntk(factset, m, mlp_method_kwargs, dtype, device)
    else:
        raise ValueError(f"Unknown mlp_method: {mlp_method!r}")

    # Evaluate accuracy.
    with torch.no_grad():
        output = mlp(factset.input_embeddings)
        predictions = output @ factset.output_embeddings.T
        predicted_indices = torch.argmax(predictions, dim=-1)
        targets = torch.tensor(
            factset.mapping.outputs, dtype=torch.long, device=device
        )
        accuracy = (predicted_indices == targets).float().mean().item()

    param_count = int(
        metrics.get("param_count", sum(p.numel() for p in mlp.parameters()))
    )

    return {
        "best_acc": accuracy,
        "m": m,
        "method": method_label,
        "param_count": param_count,
        "success": accuracy >= 1.0,
    }


# ---------------------------------------------------------------------------
# Method-specific constructors
# ---------------------------------------------------------------------------


def _run_gd(
    factset,
    m: int,
    dtype: torch.dtype,
    device: str,
    num_epochs: Optional[int] = None,
):
    """Train a bilinear gated-identity GD MLP with hidden dim m."""
    from hebbian.mlp_core.mlp_gd import GDMLPConfig, get_gd_mlp

    gd_config = GDMLPConfig()
    gd_config.m = m
    gd_config.shared.device = device
    gd_config.shared.build_dtype = dtype
    gd_config.shared.mlp_config.activation.activation = "identity"
    gd_config.bias = False
    if num_epochs is not None:
        gd_config.num_epochs = num_epochs
    return get_gd_mlp(factset, gd_config)


def _run_hebbian(factset, m: int, kwargs: dict, dtype: torch.dtype, device: str):
    """Construct a Hebbian MLP with hidden dim m."""
    # Import to register the method with the Registry.
    import hebbian.methods.hebbian  # noqa: F401

    from hebbian.core.registry import Registry
    from hebbian.methods.hebbian import HebbianConfig
    from hebbian.mlp_core.task import SharedConstructionConfig

    hebbian_config = HebbianConfig(m=m, **kwargs)
    hebbian_config.shared = SharedConstructionConfig(
        device=device,
        build_dtype=dtype,
        verbose=False,
    )

    method_class = Registry.get_method("hebbian")
    method = method_class()
    method.initialize(hebbian_config)
    return method.fit_or_construct(factset)


def _run_ntk(factset, m: int, kwargs: dict, dtype: torch.dtype, device: str):
    """Construct an NTK MLP with hidden dim m."""
    from hebbian.mlp_core.constructions.ntk import NTKConstructionConfig, get_ntk_mlp

    ntk_config = NTKConstructionConfig()
    ntk_config.m = m
    ntk_config.shared.device = device
    ntk_config.shared.build_dtype = dtype
    for key, val in kwargs.items():
        setattr(ntk_config, key, val)
    return get_ntk_mlp(factset, ntk_config)
