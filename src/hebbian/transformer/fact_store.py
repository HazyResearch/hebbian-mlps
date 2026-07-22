"""Build factsets, fact-storing MLPs, and token tables for Transformers."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from hebbian.core.registry import Registry
from hebbian.data.synthetics.factsets import (
    Factset,
    generate_factset,
    create_identity_mapping,
    create_random_permutation_mapping,
)

# Trigger method registration
import hebbian.methods  # noqa: F401


def build_factset(
    config,
    seed: int = 42,
) -> Factset:
    """Create a Factset from an associative-recall configuration.

    When ``ec.embeddings_dir`` is set, loads (x, y) tables from disk and samples
    ``num_facts`` paired rows for facts plus ``junk_vocab_size + 1`` extra rows
    for junk/Q (stashed for ``build_token_embeddings``).
    The mapping is identity so that the MLP learns the captured (x_i, y_i)
    associations directly (matches the Section 4 MLP-sweep LLM mode).

    Args:
        config: AssociativeRecallConfig instance.
        seed: Random seed.

    Returns:
        Factset with embeddings and mapping.
    """
    dc = config.dataset_config
    tc = config.train_config
    ec = tc.embeddings_config

    if ec.embeddings_dir is not None:
        return _create_factset_from_embeddings(config, seed=seed)

    mapping_type = "identity" if dc.use_identity_fact_mapping else "random"

    # Use mlp_dtype if set (e.g. float64 for Hebbian/NTK precision), else use dtype.
    factset_dtype = getattr(tc, "mlp_dtype", None) or tc.dtype
    factset = generate_factset(
        d_model=ec.d_model,
        vocab_size=dc.num_facts,
        embedding_init=ec.embedding_init,
        tie_embeddings=ec.tie_embeddings,
        mapping_type=mapping_type,
        dtype=factset_dtype,
        device=config.train_config.device,
        seed=seed,
    )
    return factset


def _create_factset_from_embeddings(config, seed: int = 42) -> Factset:
    """Sample fact + junk + Q rows from x.pt/y.pt under a seeded permutation.

    Layout of the permutation:
        perm[:F]                            -> facts (factset rows)
        perm[F : F + J + extra]             -> junk + Q pool (stashed on factset
                                               as ``_llm_junk_embeddings``)

    Identity mapping: factset row i is the captured pair (x_i, y_i).

    Synthetic-only knobs (``embedding_init``, ``tie_embeddings``,
    ``use_identity_fact_mapping``) are ignored in this mode.
    """
    dc = config.dataset_config
    tc = config.train_config
    ec = tc.embeddings_config

    factset_dtype = getattr(tc, "mlp_dtype", None) or tc.dtype
    device = tc.device

    x_path = os.path.join(ec.embeddings_dir, "x.pt")
    y_path = os.path.join(ec.embeddings_dir, "y.pt")
    x = torch.load(x_path, map_location="cpu")
    y = torch.load(y_path, map_location="cpu")
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError(
            f"x.pt and y.pt must be 2D with matching shape; got "
            f"x={tuple(x.shape)}, y={tuple(y.shape)}"
        )
    n_rows, d = int(x.shape[0]), int(x.shape[1])

    F = int(dc.num_facts)
    J = int(dc.junk_vocab_size)
    needed = F + J + 1
    if needed > n_rows:
        raise ValueError(
            f"num_facts + junk_vocab_size + 1 = {needed} "
            f"exceeds available rows N={n_rows} in {ec.embeddings_dir!r}"
        )

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    perm = torch.randperm(n_rows, generator=gen)

    fact_idx = perm[:F]
    junk_idx = perm[F:needed]

    x_fact = x[fact_idx].to(dtype=factset_dtype, device=device)
    y_fact = y[fact_idx].to(dtype=factset_dtype, device=device)
    x_junk = x[junk_idx].to(dtype=factset_dtype, device=device) if needed > F else None

    factset = Factset(
        input_embeddings=x_fact,
        output_embeddings=y_fact,
        mapping=create_identity_mapping(F),
        d_model=d,
        vocab_size=F,
    )
    # Stash junk pool so build_token_embeddings can fill the wte's junk/Q slots
    # from the same LLM activation distribution rather than random vectors.
    factset._llm_junk_embeddings = x_junk
    return factset


def _make_method_config(hidden_dim: int) -> Dict[str, Any]:
    """Create a method-specific config dict that sets hidden dimension.

    Args:
        hidden_dim: Desired hidden dimension.

    Returns:
        Dict suitable for ``method.initialize(config=...)``.
    """
    # GD, NTK, and Hebbian all accept "m" as hidden dim.
    return {"m": hidden_dim}


def build_fact_mlp(
    config,
    factset: Factset,
    method_config: Optional[Any] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Create an MLP using the hebbian Method interface.

    Looks up the Method class from the Registry based on ``config.train_config.mlp_method``,
    initializes it, and calls ``fit_or_construct(factset)``.

    If ``config.train_config.mlp_hidden_dim`` is set and ``method_config`` is None,
    a method-specific config dict is created automatically to set the hidden dimension.

    Args:
        config: AssociativeRecallConfig instance.
        factset: Factset with embeddings and mapping.
        method_config: Optional method-specific configuration. If None, uses defaults
            (or auto-generated from mlp_hidden_dim).

    Returns:
        Tuple of (mlp, metrics_dict).
    """
    method_name = config.train_config.mlp_method
    hidden_dim = config.train_config.mlp_hidden_dim

    # Auto-create method config from mlp_hidden_dim if no explicit config given
    if method_config is None and hidden_dim is not None:
        method_config = _make_method_config(hidden_dim)

    # Merge in any method-specific kwargs from TrainingConfig (e.g. variant, ridge)
    extra_kwargs = getattr(config.train_config, "mlp_method_kwargs", None) or {}
    if extra_kwargs:
        method_config = {**(method_config or {}), **extra_kwargs}

    method_cls = Registry.get_method(method_name)
    method = method_cls()
    method.initialize(config=method_config, seed=config.train_config.seed)

    # Ensure the method's SharedConstructionConfig uses the right device.
    # All method configs (GDMLPConfig, NTKConstructionConfig, HebbianConfig, etc.)
    # have a `shared: SharedConstructionConfig` with a `device` field that defaults
    # to None (CPU). Set it to match train_config.device.
    device = config.train_config.device
    for attr in vars(method).values():
        if hasattr(attr, 'shared') and hasattr(attr.shared, 'device'):
            attr.shared.device = device

    mlp, metrics = method.fit_or_construct(factset)
    return mlp, metrics


def build_token_embeddings(
    factset: Factset,
    junk_vocab_size: int,
    embedding_init: str = "normal",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    side: str = "input",
) -> nn.Embedding:
    """Create full embedding table: fact embeddings + junk + query token.

    Layout:
        [0, num_facts)                        : fact embeddings (from factset)
        [num_facts, num_facts + junk_vocab)    : junk embeddings (random)
        [num_facts + junk_vocab]               : Q token

    Args:
        factset: Factset containing fact embeddings.
        junk_vocab_size: Number of junk tokens.
        embedding_init: Initialization type for junk embeddings.
        dtype: Data type.
        seed: Random seed.
        side: "input" (wte) or "output" (lm_head). For "input", fact slots come
            from ``factset.input_embeddings``; for "output", from
            ``factset.output_embeddings``. In LLM mode, junk/Q slots on the
            input side reuse the stashed x.pt pool (same activation
            distribution); on the output side they are filled with zeros, since
            junk/Q tokens are never valid targets and zero rows can never win
            argmax against any fact row with a positive lm_head dot product.

    Returns:
        nn.Embedding with the full vocabulary.
    """
    if side not in ("input", "output"):
        raise ValueError(f"side must be 'input' or 'output'; got {side!r}")

    num_facts = factset.vocab_size
    d_model = factset.d_model
    n_extra = junk_vocab_size + 1
    total_vocab = num_facts + n_extra

    full_weight = torch.zeros(total_vocab, d_model, dtype=dtype)

    fact_rows = (
        factset.input_embeddings if side == "input" else factset.output_embeddings
    )
    full_weight[:num_facts] = fact_rows.to(dtype=dtype)

    # LLM mode: use stashed pool rows (same activation distribution as facts)
    # for the input-side junk + Q slots; zero out the output-side junk/Q slots.
    junk_source = getattr(factset, "_llm_junk_embeddings", None)
    if junk_source is not None:
        if side == "input":
            if junk_source.shape[0] != n_extra:
                raise ValueError(
                    f"Stashed _llm_junk_embeddings has {junk_source.shape[0]} rows, "
                    f"expected {n_extra} (junk_vocab_size + query token)"
                )
            full_weight[num_facts:] = junk_source.to(dtype=dtype)
        # side == "output": leave junk + Q slots as zeros (initialized above).
    else:
        torch.manual_seed(seed + 1000)  # offset to avoid correlation with fact embeddings
        # Generate junk + Q embeddings
        if embedding_init == "spherical":
            extra_emb = torch.randn(n_extra, d_model, dtype=dtype)
            extra_emb = extra_emb / extra_emb.norm(dim=1, keepdim=True)
        elif embedding_init in ("gaussian", "normal"):
            extra_emb = torch.randn(n_extra, d_model, dtype=dtype)
            # Scale to match factset embedding norms
            fact_rms = fact_rows.to(dtype=dtype).norm(dim=1).mean()
            extra_rms = extra_emb.norm(dim=1).mean()
            if extra_rms > 0:
                extra_emb = extra_emb * (fact_rms / extra_rms)
        else:
            extra_emb = torch.randn(n_extra, d_model, dtype=dtype) * 0.02

        full_weight[num_facts:] = extra_emb

    embedding = nn.Embedding(total_vocab, d_model, _freeze=False)
    embedding.weight.data.copy_(full_weight)
    return embedding
