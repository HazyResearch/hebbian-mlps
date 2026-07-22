"""Expt 4: MLP-only fact storage — binary search over m (hidden_dim).

For each (method, d_model, facts_multiplier) tuple binary-searches for the
minimum hidden_dim m
such that the MLP achieves 100% accuracy (best_acc >= 1.0) on a spherical
factset of size num_facts = facts_multiplier * d_model^2.

By default this sweep uses random permutation fact maps (`mapping_type="random"`)
with isotropic spherical embeddings and tied input/output embedding tables.

    Supported methods:
      gd               — gradient-descent trained MLP (float32)
      hebbian          — random bilinear Hebbian construction (float64)
      hebbian_whitened — random bilinear construction with full ridge readout (float64)
      cf_coord_whitened — data-dependent bilinear construction (float64)
      ntk              — NTK construction, hermite_degree=1 (float64)

    Default methods:
      gd, hebbian, hebbian_whitened, cf_coord_whitened, ntk

Usage:
  python -m hebbian.expts.mlp_capacity.run \\
      "d_models=[64]" "methods=['gd']" "facts_multiplier=0.05" "max_gpus=1"

  # Sweep multiple alpha = F/d^2 values in one run:
  python -m hebbian.expts.mlp_capacity.run \\
      "d_models=(64,90,128)" "facts_multipliers=(0.25,0.5,1.0)"
"""

from __future__ import annotations

import copy
import itertools
import os
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from hebbian.gpu_sweep import run_binary_searches
from hebbian.gpu_sweep import BinarySearchConfig
from hebbian.gpu_sweep import GPUJobResult
from hebbian.config import main as main_decorator
from hebbian.config import pydraclass

from hebbian.expts.llm_embeddings.bundle import inspect_bundle
from hebbian.expts.mlp_capacity.helpers import run_mlp_experiment  # noqa: E402

PAPER_DATA_MODE_FACT_COUNTS = (512, 1024, 2048, 4096, 8192, 16384)


# ---------------------------------------------------------------------------
# Method specs — identical to Expt 3.3 for cross-experiment consistency.
# ---------------------------------------------------------------------------

_METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "gd": dict(
        mlp_method="gd",
        mlp_dtype=None,          # float32 — same as transformer
        mlp_method_kwargs=None,
        capacity_multiplier=1.0,
    ),
    "hebbian": dict(
        mlp_method="hebbian",
        mlp_dtype=torch.float64,
        mlp_method_kwargs={"variant": "unwhitened"},
        capacity_multiplier=1.0,
    ),
    "hebbian_whitened": dict(
        mlp_method="hebbian",
        mlp_dtype=torch.float64,
        mlp_method_kwargs={"variant": "whitened", "ridge": 1e-6},
        capacity_multiplier=1.0,
    ),
    "ntk": dict(
        mlp_method="ntk",
        mlp_dtype=torch.float64,
        mlp_method_kwargs={"hermite_degree": 1},
        capacity_multiplier=1.0,
    ),
    "cf_coord_whitened": dict(
        mlp_method="hebbian",
        mlp_dtype=torch.float64,
        mlp_method_kwargs={"variant": "data_dependent", "ridge": 1e-6},
        capacity_multiplier=1.0,
    ),
}


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------


@pydraclass
class ExperimentConfig:
    """Top-level config for Expt 4 MLP capacity sweep."""

    d_models: tuple[int, ...] = (64, 90, 128)
    # Single override for alpha = F/d^2. If set, this takes precedence.
    facts_multiplier: Optional[float] = None
    # Default alpha sweep matches fig2 launcher defaults.
    facts_multipliers: tuple[float, ...] = (
        0.03125,
        0.0625,
        0.125,
        0.25,
        0.5,
        0.75,
        1.0,
    )
    # Direct fact-count override. Useful for fixed-size LLM activation bundles.
    num_facts_values: Optional[tuple[int, ...]] = None
    methods: tuple[str, ...] = (
        "gd",
        "hebbian",
        "hebbian_whitened",
        "cf_coord_whitened",
        "ntk",
    )
    mapping_type: str = "random"         # fact-map type: "random" or "identity"
    embedding_init: str = "spherical"
    tie_embeddings: bool = True
    spike_beta: float = 0.0
    spike_target: str = "both"
    spike_seed: int = 42
    embeddings_dir: Optional[str] = None
    use_embedding_d_model: bool = True
    success_acc_threshold: float = 1.0

    # Binary search range and precision over m (hidden_dim).
    binary_search_range: tuple[float, float] = (1.0, 65536.0)
    # Optional per-method override for binary-search range, e.g.
    # {"gd": (1.0, 512.0), "ntk": (1.0, 65536.0)}.
    method_binary_search_ranges: Optional[dict[str, tuple[float, float]]] = None
    binary_search_precision: float = 0.05  # relative precision

    # Seeds.
    n_seeds: int = 1
    seed: int = 42

    # Compute.
    device: str = "cuda"
    max_gpus: int = 4
    simultaneous_jobs_per_gpu: int = 1

    # Output.
    base_dir: str = "./artifacts/mlp_capacity"


# ---------------------------------------------------------------------------
# Thin wrapper satisfying the sweep runner's finalize() protocol
# ---------------------------------------------------------------------------


class _RunConfig:
    """Picklable wrapper around a kwargs dict with a finalize hook."""

    def __init__(self, kwargs_dict: dict):
        self._kwargs = dict(kwargs_dict)

    def finalize(self) -> None:
        """No-op called after get_experiment_config_and_base_dir."""

    def to_kwargs(self) -> dict:
        return dict(self._kwargs)


# ---------------------------------------------------------------------------
# Per-(method, d_model) binary search config
# ---------------------------------------------------------------------------


@pydraclass
class MlpCapacityBinarySearchConfig(BinarySearchConfig):
    """Binary search config for one (method_label, d_model) pair.

    Stores context in base_experiment_config as a plain dict so that
    get_experiment_config_and_base_dir can update m and return a _RunConfig.
    """

    success_acc_threshold: float = 1.0

    def get_experiment_config_and_base_dir(self, m, **kwargs):
        m_int = max(1, int(round(float(m))))
        ctx = copy.deepcopy(self.base_experiment_config)
        ctx["m"] = m_int

        method_label = ctx["method_label"]
        d_model = ctx["d_model"]
        num_facts = ctx["num_facts"]
        base_dir = os.path.join(
            self.base_dir,
            method_label,
            f"d{d_model}_F{num_facts}",
            f"m_{m_int}",
        )
        os.makedirs(base_dir, exist_ok=True)
        return _RunConfig(ctx), base_dir

    def run_experiment_config(self, config: _RunConfig):
        return run_mlp_experiment(**config.to_kwargs())

    def agg_results(self, results: List[GPUJobResult]):
        valid = [r for r in results if r.success and isinstance(r.result, dict)]
        if not valid:
            return False, None
        best_idx = int(np.argmax([r.result.get("best_acc", 0.0) for r in valid]))
        best = valid[best_idx]
        return bool(
            best.result.get("best_acc", 0.0) >= float(self.success_acc_threshold)
        ), best


# ---------------------------------------------------------------------------
# Sweep config generation
# ---------------------------------------------------------------------------


def get_sweep_configs(config: ExperimentConfig) -> List[MlpCapacityBinarySearchConfig]:
    """Create one binary-search config per (method, d_model) pair."""
    base_dir_root = config.base_dir
    print(f"Base directory: {base_dir_root}")

    bundle_info = None
    d_models = tuple(int(v) for v in config.d_models)
    if config.embeddings_dir is not None:
        bundle_info = inspect_bundle(config.embeddings_dir)
        print(
            "Using LLM activation rows: "
            f"{bundle_info.activation_dir} "
            f"({bundle_info.num_pairs} rows, d={bundle_info.d_model})"
        )
        if config.use_embedding_d_model:
            d_models = (int(bundle_info.d_model),)

    for label in config.methods:
        if label not in _METHOD_SPECS:
            raise ValueError(
                f"Unknown method label {label!r}. Valid: {sorted(_METHOD_SPECS)}"
            )

    method_binary_search_ranges = config.method_binary_search_ranges or {}
    for label, search_range in method_binary_search_ranges.items():
        if label not in _METHOD_SPECS:
            raise ValueError(
                f"Unknown method in method_binary_search_ranges: {label!r}. "
                f"Valid: {sorted(_METHOD_SPECS)}"
            )
        if len(search_range) != 2:
            raise ValueError(
                f"method_binary_search_ranges[{label!r}] must be a length-2 tuple, "
                f"got {search_range!r}"
            )
        lo, hi = float(search_range[0]), float(search_range[1])
        if lo <= 0 or hi <= lo:
            raise ValueError(
                f"Invalid method search range for {label!r}: {(lo, hi)!r}"
            )

    direct_fact_values = config.num_facts_values
    if config.embeddings_dir is not None and direct_fact_values is None:
        direct_fact_values = PAPER_DATA_MODE_FACT_COUNTS

    num_facts_values: list[int] | None = None
    if direct_fact_values is not None:
        num_facts_values = []
        seen_facts = set()
        for v in direct_fact_values:
            iv = int(v)
            if iv <= 0:
                raise ValueError(f"num_facts_values must be positive, got {iv}")
            if bundle_info is not None and iv > bundle_info.num_pairs:
                raise ValueError(
                    f"num_facts={iv} exceeds activation rows {bundle_info.num_pairs}"
                )
            if iv not in seen_facts:
                num_facts_values.append(iv)
                seen_facts.add(iv)
        if len(num_facts_values) == 0:
            raise ValueError("num_facts_values cannot be empty")
        print(f"Fact counts: {tuple(num_facts_values)}")
    else:
        facts_multiplier_values: List[float]
        if config.facts_multiplier is not None:
            facts_multiplier_values = [float(config.facts_multiplier)]
        else:
            facts_multiplier_values = []
            seen = set()
            for v in config.facts_multipliers:
                fv = float(v)
                if fv <= 0:
                    raise ValueError(f"facts_multipliers must be positive, got {fv}")
                if fv not in seen:
                    facts_multiplier_values.append(fv)
                    seen.add(fv)
            if len(facts_multiplier_values) == 0:
                raise ValueError("facts_multipliers cannot be empty when facts_multiplier is None")
        print(f"Facts multipliers (alpha=F/d^2): {tuple(facts_multiplier_values)}")

    if config.embeddings_dir is not None and direct_fact_values is None:
        raise AssertionError("internal error: data-mode fact counts were not resolved")

    if config.embeddings_dir is not None and config.spike_beta > 0:
        raise ValueError("spike_beta is synthetic-only when embeddings_dir is set")

    if config.embeddings_dir is not None and config.mapping_type != "identity":
        print("WARNING: embeddings_dir uses paired rows with identity mapping; mapping_type is ignored.")

    if config.embeddings_dir is not None and config.embedding_init != "spherical":
        print("WARNING: embeddings_dir uses activation rows; embedding_init is ignored.")

    if config.embeddings_dir is not None and config.tie_embeddings:
        print("WARNING: embeddings_dir uses separate x/y activation rows; tie_embeddings is ignored.")

    sweep_configs: List[MlpCapacityBinarySearchConfig] = []

    for method_label in config.methods:
        spec = _METHOD_SPECS[method_label]
        method_search_range = tuple(
            float(v)
            for v in method_binary_search_ranges.get(
                method_label,
                config.binary_search_range,
            )
        )

        for d_model in d_models:
            if num_facts_values is None:
                fact_specs = [
                    (int(facts_multiplier * d_model ** 2), facts_multiplier)
                    for facts_multiplier in facts_multiplier_values
                ]
            else:
                fact_specs = [
                    (num_facts, float(num_facts) / float(d_model ** 2))
                    for num_facts in num_facts_values
                ]

            for num_facts, facts_multiplier in fact_specs:
                if num_facts <= 0:
                    raise ValueError(
                        f"num_facts must be positive, got d={d_model}, "
                        f"facts_multiplier={facts_multiplier}, num_facts={num_facts}"
                    )

                base_experiment_config = {
                    "d_model": d_model,
                    "num_facts": num_facts,
                    "method_label": method_label,
                    "method_spec": spec,
                    "device": config.device,
                    "seed": config.seed,
                    "mapping_type": config.mapping_type,
                    "facts_multiplier": facts_multiplier,
                    "embedding_init": config.embedding_init,
                    "tie_embeddings": bool(config.tie_embeddings),
                    "spike_beta": float(config.spike_beta),
                    "spike_target": config.spike_target,
                    "spike_seed": int(config.spike_seed),
                    "embeddings_dir": config.embeddings_dir,
                    # "m" will be filled in by get_experiment_config_and_base_dir
                }

                run_dir = os.path.join(
                    base_dir_root,
                    method_label,
                    f"d{d_model}_F{num_facts}",
                )
                os.makedirs(run_dir, exist_ok=True)

                search_cfg = MlpCapacityBinarySearchConfig(
                    base_dir=run_dir,
                    prop="m",
                    range=method_search_range,
                    precision=config.binary_search_precision,
                    success_direction_lower=True,   # success=True at higher m; find minimum m that succeeds
                    sweep_props={},
                    base_experiment_config=base_experiment_config,
                    success_acc_threshold=float(config.success_acc_threshold),
                )

                sweep_configs.append(search_cfg)
                print(
                    f"  [{method_label}] d={d_model} alpha={facts_multiplier:g} "
                    f"F={num_facts} search_range={method_search_range}"
                )

    return sweep_configs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _sweep_prop_product(props_dict: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    """Return the property cross product scheduled by the GPU sweep runner."""

    if not props_dict:
        return [{}]
    prop_names = list(props_dict.keys())
    prop_value_lists = [props_dict[name] for name in prop_names]
    return [
        dict(zip(prop_names, combination))
        for combination in itertools.product(*prop_value_lists)
    ]


def _run_binary_searches_locally(
    configs: list[MlpCapacityBinarySearchConfig],
) -> list[tuple[Any, Any]]:
    """Run binary searches sequentially on the local device."""

    all_results = []
    for binary_config in configs:
        start_time = time.time()
        lo, hi = tuple(float(v) for v in binary_config.range)
        precision = float(binary_config.precision)
        achieved_results = None
        failed_results = None

        print(
            f"Binary search for {binary_config.prop} in [{lo}, {hi}], "
            f"precision={precision}",
            flush=True,
        )
        while (hi - lo) >= precision:
            mid = (lo + hi) / 2
            print(
                f"  Testing {binary_config.prop}={mid} (range: [{lo}, {hi}])",
                flush=True,
            )
            combined_props = dict(binary_config.sweep_props) if binary_config.sweep_props else {}
            combined_props[binary_config.prop] = [mid]
            job_results = []
            for prop_values in _sweep_prop_product(combined_props):
                exp_config, experiment_base_dir = binary_config._get_experiment_config_and_base_dir(
                    **prop_values
                )
                os.makedirs(experiment_base_dir, exist_ok=True)
                try:
                    result = binary_config.run_experiment_config(exp_config)
                    job_results.append(
                        GPUJobResult(
                            success=True,
                            error=None,
                            gpu_id=-1,
                            out_file=None,
                            job=None,
                            result=result,
                        )
                    )
                except Exception as exc:
                    job_results.append(
                        GPUJobResult(
                            success=False,
                            error=str(exc),
                            gpu_id=-1,
                            out_file=None,
                            job=None,
                            result=None,
                        )
                    )

            success, aggregated_result = binary_config.agg_results(job_results)
            if success:
                print(f"Succeeded at {mid}", flush=True)
                achieved_results = (mid, aggregated_result)
                if binary_config.success_direction_lower:
                    hi = mid
                else:
                    lo = mid
            else:
                print(f"Failed at {mid}", flush=True)
                failed_results = (mid, aggregated_result)
                if binary_config.success_direction_lower:
                    lo = mid
                else:
                    hi = mid

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        binary_search_results = {
            "search_range": [lo, hi],
            "precision": precision,
            "achieved_results": achieved_results,
            "failed_results": failed_results,
            "total_time": time.time() - start_time,
            "timestamp": timestamp,
        }
        os.makedirs(binary_config.base_dir, exist_ok=True)
        results_filename = f"{binary_config.base_dir}/binary_search_results_{timestamp}.pkl"
        with open(results_filename, "wb") as f:
            pickle.dump(binary_search_results, f)
        print(f"Binary search results saved to: {results_filename}", flush=True)
        all_results.append((achieved_results, failed_results))
    return all_results


def run(config: ExperimentConfig):
    for label in config.methods:
        if label not in _METHOD_SPECS:
            raise ValueError(f"Unknown method {label!r}. Valid: {sorted(_METHOD_SPECS)}")

    configs = get_sweep_configs(config)
    n_methods = len(config.methods)
    d_model_count = 1 if config.embeddings_dir and config.use_embedding_d_model else len(config.d_models)
    print(
        f"\nRunning {len(configs)} binary-search configs "
        f"({n_methods} methods × {d_model_count} d_models):"
    )

    if int(config.max_gpus) <= 0:
        print("Running sequential local fallback because max_gpus <= 0")
        return _run_binary_searches_locally(configs)

    return run_binary_searches(
        configs,
        max_gpus=config.max_gpus,
        simultaneous_jobs_per_gpu=config.simultaneous_jobs_per_gpu,
    )


@main_decorator(ExperimentConfig)
def main(config: ExperimentConfig):
    return run(config)


if __name__ == "__main__":
    main()
