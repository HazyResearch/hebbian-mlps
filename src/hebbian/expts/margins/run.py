"""
Margin sweep experiment for Hebbian MLPs using the shared GPU grid search.

Sweeps over a grid of (F, M, beta) values and computes margin quantities
for each point. Each grid point is its own GridSearchConfig with seeds as
sweep_props. agg_results picks the best seed per point.

Usage examples:
    # Sweep F with fixed M=256, d=128
    python -m hebbian.expts.margins.run --sweep F --d 128 --M 256 --F_min 128 --F_max 4096

    # Sweep M with fixed F=512, d=128
    python -m hebbian.expts.margins.run --sweep M --d 128 --F 512 --M_min 64 --M_max 2048

    # Sweep F x M grid
    python -m hebbian.expts.margins.run --sweep FM --d 128 --F_min 128 --F_max 2048 --M_min 64 --M_max 1024

    # Sweep beta (spike strength) on both embeddings
    python -m hebbian.expts.margins.run --sweep beta --d 128 --F 512 --M 256 --beta_min 0.1 --beta_max 100

    # Sweep beta on input (key) embeddings only
    python -m hebbian.expts.margins.run --sweep beta --spike_target keys --d 128 --F 512 --M 256

    # Sweep beta on output (value) embeddings only
    python -m hebbian.expts.margins.run --sweep beta --spike_target values --d 128 --F 512 --M 256

Results are saved as search pickles and as JSON for easy loading.
"""

import os
import copy
import json
import datetime
import itertools
import pickle
import time
from typing import Any, List

import numpy as np
import torch
from hebbian.config import pydraclass, main as pydra_main

from hebbian.gpu_sweep import GridSearchConfig
from hebbian.gpu_sweep import GPUJobResult
from hebbian.gpu_sweep import run_grid_searches

from hebbian.data.synthetics import generate_factset
from hebbian.data.embeddings.transforms import spike_embeddings
from hebbian.core.metrics import compute_coherence
from hebbian.methods.hebbian import HebbianMethod, HebbianConfig
from hebbian.mlp_core.task import SharedConstructionConfig
from hebbian.expts.llm_embeddings.bundle import (
    inspect_bundle,
    make_factset_from_activation_rows,
)
from hebbian.metrics.margin_theory import (
    compute_margin_quantities,
    compute_lipschitz_bound_bilinear,
    noisy_query_margin_bound,
    compute_noisy_margin,
)
from hebbian.metrics.value_separability import fit_value_separators
import torch.nn.functional as F_func


# =============================================================================
# Experiment config (one per grid point + seed)
# =============================================================================

@pydraclass
class MarginExperimentConfig:
    """Configuration for a single margin experiment."""
    d: int = 128
    M: int = 256
    F: int = 512
    seed: int = 0
    device: str = "cuda"
    build_dtype: str = "float64"
    beta: float = 0.0        # Spike strength for spiked covariance (0 = no spike)
    spike_target: str = "both"  # Which embeddings to spike: "keys", "values", or "both"
    spike_seed: int = 42     # Seed for spike direction (fixed across sweep)
    use_u_star_codes: bool = True  # Use u*-codes (rho-based) instead of V-codes (mu-based)
    admm_n_iters: int = 300       # ADMM iterations for u* computation
    admm_batch_size: int = 256    # ADMM batch size for u* computation
    epsilon: float = 0.0     # Noise level for noisy-query margin (0 = clean stored-query margin)
    noise_seed: int = 123    # Seed for random query perturbation directions
    embeddings_dir: str | None = None  # Optional LLM activation bundle root or activations/ dir
    gamma_min_percentile: float | None = None


def run_margin_experiment(config: MarginExperimentConfig) -> dict:
    """
    Run a single margin experiment.

    Creates a factset with F random spherical embeddings of dimension d,
    optionally applies a rank-1 spike (spiked covariance model) to control
    coherence, constructs an unwhitened bilinear HebbianMLP with feature
    dimension M,
    and computes all margin quantities (using V as codes).
    """
    if config.embeddings_dir is None:
        factset = generate_factset(
            d_model=config.d,
            vocab_size=config.F,
            embedding_init="spherical",
            tie_embeddings=False,
            mapping_type="identity",
            dtype=torch.float64,
            device=config.device,
            seed=config.seed,
        )
    else:
        if config.beta > 0:
            raise ValueError("beta/spiked covariance sweeps are synthetic-only")
        factset = make_factset_from_activation_rows(
            config.embeddings_dir,
            num_facts=config.F,
            seed=config.seed,
            dtype=torch.float64,
            device=config.device,
        )
        config.d = int(factset.d_model)

    # Apply spiked covariance transform if beta > 0
    if config.beta > 0:
        target = config.spike_target  # "keys", "values", or "both"

        if target in ("keys", "both"):
            factset.input_embeddings = spike_embeddings(
                factset.input_embeddings,
                beta=config.beta,
                seed=config.spike_seed,
                normalize=True,
            )

        if target in ("values", "both"):
            factset.output_embeddings = spike_embeddings(
                factset.output_embeddings,
                beta=config.beta,
                seed=config.spike_seed,
                normalize=True,
            )

    # Measure coherence of input and output embeddings separately
    coherence = compute_coherence(factset.input_embeddings)
    coherence_values = compute_coherence(factset.output_embeddings)

    torch_device = torch.device(config.device)

    hebbian_config = HebbianConfig(
        variant="unwhitened",
        m=config.M,
    )
    hebbian_config.shared = SharedConstructionConfig(
        device=config.device,
        build_dtype=config.build_dtype,
        verbose=False,
    )

    method = HebbianMethod()
    method.initialize(hebbian_config, seed=config.seed)
    mlp, method_metrics = method.fit_or_construct(factset)

    # Compute u*-codes if requested
    U_all = None
    rho_V = None
    rho_per_i = None
    if config.use_u_star_codes:
        V_all_dev = factset.output_embeddings.to(device=torch_device, dtype=torch.float64)
        U_all, rho_V_scalar, rho_per_i_tensor = fit_value_separators(
            V_all_dev,
            batch_size=config.admm_batch_size,
            rho=1.0,
            num_iters=config.admm_n_iters,
        )
        rho_V = rho_V_scalar
        rho_per_i = rho_per_i_tensor

    quantities = compute_margin_quantities(
        mlp, factset, torch_device, torch.float64,
        use_u_star_codes=config.use_u_star_codes,
        U_all=U_all,
        rho_V=rho_V,
        rho_per_i=rho_per_i,
        gamma_min_percentile=config.gamma_min_percentile,
    )

    # signal_matrix already has inf at correct-class entries -> .min() is over valid pairs
    signal_min = quantities["signal_matrix"].min().item()
    # crosstalk_matrix has 0 at correct-class entries; mask them to inf for proper min
    ct = quantities["crosstalk_matrix"].clone()
    ct[ct == 0] = float('inf')  # masked entries were set to exactly 0.0
    # safer: re-mask using correct indices
    n_keys = quantities["n"]
    value_idx = [factset.mapping.get_output(i) for i in range(n_keys)]
    ct[torch.arange(n_keys), torch.tensor(value_idx)] = float('inf')
    crosstalk_min = ct.min().item()

    # ------------------------------------------------------------------
    # Noisy-query margin (when epsilon > 0)
    # ------------------------------------------------------------------
    gamma_min_noisy = None
    noisy_bound_val = None
    L_bil = None

    if config.epsilon > 0:
        n_keys = quantities["n"]
        K_clean = factset.input_embeddings.to(device=torch_device, dtype=torch.float64)
        d_keys = K_clean.shape[1]

        # Sample random unit perturbation directions (fixed by noise_seed)
        noise_gen = torch.Generator(device=torch_device).manual_seed(config.noise_seed)
        raw_noise = torch.randn(n_keys, d_keys, generator=noise_gen,
                                dtype=torch.float64, device=torch_device)
        noise_dir = F_func.normalize(raw_noise, dim=1)  # (n, d) unit vectors
        Z_noisy = K_clean + config.epsilon * noise_dir   # z_i = k_i + epsilon * u_i

        # Noisy margin at perturbed queries
        noisy_result = compute_noisy_margin(
            mlp,
            factset,
            Z_noisy,
            torch_device,
            torch.float64,
            gamma_min_percentile=config.gamma_min_percentile,
        )
        gamma_min_noisy = noisy_result["gamma_min"]

        # Lipschitz constant (bilinear only)
        L_bil = compute_lipschitz_bound_bilinear(mlp.feature_map, K_clean, Z_noisy)
        bound_dict = noisy_query_margin_bound(
            n=n_keys, d=d_keys, m=config.M,
            epsilon=config.epsilon, L_bil=L_bil,
            delta=0.5,
        )
        noisy_bound_val = bound_dict["noisy_bound"]

    return {
        "d": config.d,
        "M": config.M,
        "F": config.F,
        "beta": config.beta,
        "epsilon": config.epsilon,
        "spike_target": config.spike_target,
        "coherence_keys": coherence,
        "coherence_values": coherence_values,
        "seed": config.seed,
        "n": quantities["n"],
        "G": quantities["G"],
        "gamma_min": quantities["gamma_min"],
        "accuracy": quantities["accuracy"],
        "E_col_max": quantities["E_col_max"],
        "K_hat_diag_min": quantities["K_hat_diag_min"],
        "delta_min": quantities["delta_min"],
        "mu_V": quantities["mu_V"],
        "d_int": quantities["d_int"],
        "unified_bound": quantities["unified_bound"],
        "signal_min": signal_min,
        "crosstalk_min": crosstalk_min,
        "signal_per_key": quantities["signal_per_key"].tolist(),
        "crosstalk_per_key": quantities["crosstalk_per_key"].tolist(),
        "K_hat_diag_max": quantities["K_hat_diag_max"],
        "var_max": quantities["var_max"],
        "E_col_i_star": quantities["E_col_i_star"],
        "a_star_norm_sq": quantities["a_star_norm_sq"],
        "var_i_star_j_star": quantities["var_i_star_j_star"],
        "u_hat_int_sq_baseline": quantities["u_hat_int_sq_baseline"],
        "K_hat_diag_i_star": quantities["K_hat_diag_i_star"],
        "signal_dot_i_star": quantities["signal_dot_i_star"],
        "crosstalk_at_i_star": quantities["crosstalk_at_i_star"],
        "K_hat_diag_mean": quantities["K_hat_diag_mean"],
        "mu_V_mean": quantities["mu_V_mean"],
        "a_norm_sq_mean": quantities["a_norm_sq_mean"],
        "E_col_mean": quantities["E_col_mean"],
        "var_mean": quantities["var_mean"],
        "mu_int_sq_mean": quantities["mu_int_sq_mean"],
        "rho_V": quantities.get("rho_V"),
        "signal_dot_mean": quantities.get("signal_dot_mean"),
        "E_Y": quantities.get("E_Y", 0.0),
        "E_Y_mean": quantities.get("E_Y_mean", 0.0),
        "L_v": quantities.get("L_v", 0.0),
        "kappa_max": quantities.get("kappa_max", 0.0),
        "kappa_mean": quantities.get("kappa_mean", 0.0),
        "A_min": quantities.get("A_min", 0.0),
        "B_max": quantities.get("B_max", 0.0),
        "B_Y": quantities.get("B_Y", 0.0),
        "K_max_off": quantities.get("K_max_off", 0.0),
        "param_count": method_metrics.get("param_count", 0),
        # Noisy-query quantities
        "gamma_min_noisy": gamma_min_noisy,
        "noisy_bound": noisy_bound_val,
        "L_bil": L_bil,
    }


# =============================================================================
# Grid search config (one instance per (F, M) grid point)
# =============================================================================

@pydraclass
class MarginSweepGridSearchConfig(GridSearchConfig):
    """
    Grid search for a single (F, M) point across seeds.

    sweep_props should only contain {"seed": [0, 1, 2, ...]}.
    agg_results picks the best seed (highest gamma_min) and returns
    a summary dict for this grid point.
    """

    def get_experiment_config_and_base_dir(self, **prop_values) -> tuple:
        """Create experiment config from seed sweep value."""
        config = copy.deepcopy(self.base_experiment_config)

        if "seed" in prop_values:
            config.seed = int(prop_values["seed"])

        base_dir = f"{self.base_dir}/seed{config.seed}"
        return config, base_dir

    def run_experiment_config(self, config: MarginExperimentConfig) -> dict:
        """Run a single margin experiment."""
        return run_margin_experiment(config)

    def agg_results(self, results: List[GPUJobResult]) -> dict:
        """Average gamma_min across seeds; report mean as gamma_min_best and std."""
        successful = [r.result for r in results if r.success and r.result is not None]
        if not successful:
            return None

        gammas = [r["gamma_min"] for r in successful]
        best_idx = int(np.argmax(gammas))
        best = successful[best_idx]

        return {
            "d": best["d"],
            "M": best["M"],
            "F": best["F"],
            "beta": best.get("beta", 0.0),
            "epsilon": best.get("epsilon", 0.0),
            "spike_target": best.get("spike_target", "both"),
            "coherence_keys": best.get("coherence_keys", 0.0),
            "coherence_values": best.get("coherence_values", 0.0),
            "n": best["n"],
            "G": best["G"],
            "best_seed": best["seed"],
            "gamma_min_best": float(np.mean(gammas)),
            "gamma_min_std": float(np.std(gammas)),
            "gamma_min_all": gammas,
            "accuracy_best": best["accuracy"],
            "E_col_max_best": best["E_col_max"],
            "K_hat_diag_min_best": best["K_hat_diag_min"],
            "delta_min_best": best["delta_min"],
            "mu_V_best": best["mu_V"],
            "d_int_best": best["d_int"],
            "unified_bound": best["unified_bound"],
            "signal_min_best": best["signal_min"],
            "crosstalk_min_best": best["crosstalk_min"],
            "signal_per_key_best": best.get("signal_per_key", []),
            "crosstalk_per_key_best": best.get("crosstalk_per_key", []),
            "K_hat_diag_min": best["K_hat_diag_min"],
            "K_hat_diag_max": best["K_hat_diag_max"],
            "var_max_best": best.get("var_max", 0),
            "E_col_i_star_best": best.get("E_col_i_star", 0),
            "a_star_norm_sq_best": best.get("a_star_norm_sq", 0),
            "var_i_star_j_star_best": best.get("var_i_star_j_star", 0),
            "u_hat_int_sq_baseline_best": best.get("u_hat_int_sq_baseline", 0),
            "K_hat_diag_i_star_best": best.get("K_hat_diag_i_star", 0),
            "signal_dot_i_star_best": best.get("signal_dot_i_star", 0),
            "crosstalk_at_i_star_best": best.get("crosstalk_at_i_star", 0),
            "K_hat_diag_mean_best": best.get("K_hat_diag_mean", 0),
            "mu_V_mean_best": best.get("mu_V_mean", 0),
            "a_norm_sq_mean_best": best.get("a_norm_sq_mean", 0),
            "E_col_mean_best": best.get("E_col_mean", 0),
            "var_mean_best": best.get("var_mean", 0),
            "mu_int_sq_mean_best": best.get("mu_int_sq_mean", 0),
            "rho_V_best": best.get("rho_V"),
            "signal_dot_mean_best": best.get("signal_dot_mean", 0),
            "E_Y_best": best.get("E_Y", 0.0),
            "E_Y_mean_best": best.get("E_Y_mean", 0.0),
            "L_v_best": best.get("L_v", 0.0),
            "kappa_max_best": best.get("kappa_max", 0.0),
            "kappa_mean_best": best.get("kappa_mean", 0.0),
            "A_min_best": best.get("A_min", 0.0),
            "B_max_best": best.get("B_max", 0.0),
            "B_Y_best": best.get("B_Y", 0.0),
            "K_max_off_best": best.get("K_max_off", 0.0),
            "param_count": best.get("param_count", 0),
            # Noisy-query quantities
            "gamma_min_noisy_best": best.get("gamma_min_noisy"),
            "noisy_bound_best": best.get("noisy_bound"),
            "L_bil_best": best.get("L_bil"),
        }


# =============================================================================
# Sweep runner config + main
# =============================================================================

@pydraclass
class MarginSweepRunnerConfig:
    """Top-level configuration for the margin sweep."""

    # Sweep mode
    sweep: str = "F"  # "F", "M", "FM", "beta", or "epsilon"

    # Fixed parameters
    d: int = 128
    device: str = "cuda"
    build_dtype: str = "float64"

    # F-sweep params (also used as fixed F for M-sweep / beta-sweep)
    F: int = 512
    F_min: int = 128
    F_max: int = 4096
    n_F_points: int = 15

    # M-sweep params (also used as fixed M for F-sweep / beta-sweep)
    M: int = 256
    M_min: int = 64
    M_max: int = 2048
    n_M_points: int = 15

    # Beta-sweep params (spike strength for spiked covariance model)
    beta: float = 0.0         # Fixed beta when not sweeping beta
    beta_min: float = 0.1
    beta_max: float = 100.0
    n_beta_points: int = 15
    spike_target: str = "both"  # Which embeddings to spike: "keys", "values", or "both"
    spike_seed: int = 42      # Seed for spike direction (shared across sweep)

    # Epsilon-sweep params (noisy-query perturbation)
    epsilon: float = 0.0      # Fixed epsilon when not sweeping epsilon (0 = clean queries)
    epsilon_min: float = 0.01
    epsilon_max: float = 1.0
    n_epsilon_points: int = 15
    noise_seed: int = 123     # Seed for query perturbation directions (shared across sweep)

    # u*-codes settings
    use_u_star_codes: bool = True  # Use u*-codes (rho-based) instead of V-codes (mu-based)
    admm_n_iters: int = 300       # ADMM iterations for u* computation
    admm_batch_size: int = 256    # ADMM batch size for u* computation

    # Seeds
    n_seeds: int = 1
    seed_offset: int = 0

    # Optional activation-table source for LLM margin sweeps.
    embeddings_dir: str | None = None
    gamma_min_percentile: float | None = None

    # GPU settings
    max_gpus: int = 8
    simultaneous_jobs_per_gpu: int = 8

    # Output
    base_dir: str | None = None
    output_json: str | None = None


def get_grid_points(config: MarginSweepRunnerConfig) -> list[tuple]:
    """Return list of (F, M, beta) grid points to sweep over."""
    beta_fixed = config.beta

    if config.sweep == "F":
        F_values = np.logspace(np.log10(config.F_min), np.log10(config.F_max), config.n_F_points)
        F_values = np.unique(np.round(F_values).astype(int)).tolist()
        return [(F, config.M, beta_fixed) for F in F_values]

    elif config.sweep == "M":
        M_values = np.logspace(np.log10(config.M_min), np.log10(config.M_max), config.n_M_points)
        M_values = np.unique(np.round(M_values).astype(int)).tolist()
        return [(config.F, M, beta_fixed) for M in M_values]

    elif config.sweep == "beta":
        beta_0 = config.beta_min == 0
        if beta_0:
            config.beta_min = 1
            config.n_beta_points = config.n_beta_points - 1
        beta_values = np.logspace(
            np.log10(config.beta_min), np.log10(config.beta_max), config.n_beta_points
        ).tolist()
        if beta_0:
            beta_values = [0] + beta_values
            config.beta_min = 0
            config.n_beta_points = config.n_beta_points + 1
        return [(config.F, config.M, beta) for beta in beta_values]

    elif config.sweep == "epsilon":
        epsilon_values = np.logspace(
            np.log10(config.epsilon_min), np.log10(config.epsilon_max), config.n_epsilon_points
        ).tolist()
        return [(config.F, config.M, beta_fixed, eps) for eps in epsilon_values]

    else:  # FM
        F_values = np.logspace(np.log10(config.F_min), np.log10(config.F_max), config.n_F_points)
        F_values = np.unique(np.round(F_values).astype(int)).tolist()
        M_values = np.logspace(np.log10(config.M_min), np.log10(config.M_max), config.n_M_points)
        M_values = np.unique(np.round(M_values).astype(int)).tolist()
        return [(F, M, beta_fixed) for F in F_values for M in M_values]


def build_grid_search_configs(
    config: MarginSweepRunnerConfig, base_dir: str
) -> list[MarginSweepGridSearchConfig]:
    """Create one GridSearchConfig per grid point, each sweeping over seeds."""
    grid_points = get_grid_points(config)
    seed_props = {"seed": [int(config.seed_offset) + i for i in range(config.n_seeds)]}

    configs = []
    for point in grid_points:
        # Unpack: epsilon sweep emits 4-tuples; all others emit 3-tuples
        if len(point) == 4:
            F, M, beta, epsilon = point
        else:
            F, M, beta = point
            epsilon = config.epsilon

        base_exp_config = MarginExperimentConfig(
            d=config.d,
            M=M,
            F=F,
            seed=0,
            device=config.device,
            build_dtype=config.build_dtype,
            beta=beta,
            spike_target=config.spike_target,
            spike_seed=config.spike_seed,
            use_u_star_codes=config.use_u_star_codes,
            admm_n_iters=config.admm_n_iters,
            admm_batch_size=config.admm_batch_size,
            epsilon=epsilon,
            noise_seed=config.noise_seed,
            embeddings_dir=config.embeddings_dir,
            gamma_min_percentile=config.gamma_min_percentile,
        )

        point_dir = f"{base_dir}/d{config.d}/F{F}/M{M}"
        point_dir += f"/beta{beta:.4g}_{config.spike_target}"
        if epsilon > 0:
            point_dir += f"/eps{epsilon:.4g}"

        grid_config = MarginSweepGridSearchConfig(
            base_dir=point_dir,
            sweep_props=seed_props,
            base_experiment_config=base_exp_config,
        )
        configs.append(grid_config)

    return configs


def aggregate_to_sweep_results(
    per_point_results: list[dict | None], config: MarginSweepRunnerConfig
) -> dict:
    """
    Aggregate per-grid-point results (from agg_results) into a structured sweep result.

    Each element of per_point_results is the dict returned by
    MarginSweepGridSearchConfig.agg_results (best seed summary for one (F, M) point),
    or None if all seeds failed.
    """
    results = [r for r in per_point_results if r is not None]
    if not results:
        return {}

    d = config.d
    sweep_type = config.sweep

    if sweep_type == "F":
        F_values = sorted(set(r["F"] for r in results))
        M = config.M
        sorted_results = sorted(results, key=lambda r: r["F"])
        out = {
            "sweep_type": "F",
            "d": d,
            "M": M,
            "F_values": F_values,
            "gamma_min_best": [r["gamma_min_best"] for r in sorted_results],
            "gamma_min_std": [r["gamma_min_std"] for r in sorted_results],
            "gamma_min_all": [r["gamma_min_all"] for r in sorted_results],
            "accuracy_best": [r["accuracy_best"] for r in sorted_results],
            "E_col_max_best": [r["E_col_max_best"] for r in sorted_results],
            "K_hat_diag_min_best": [r["K_hat_diag_min_best"] for r in sorted_results],
            "K_hat_diag_max_best": [r.get("K_hat_diag_max", r["K_hat_diag_min_best"]) for r in sorted_results],
            "delta_min_best": [r["delta_min_best"] for r in sorted_results],
            "mu_V_best": [r["mu_V_best"] for r in sorted_results],
            "d_int_best": [r["d_int_best"] for r in sorted_results],
            "signal_min_best": [r["signal_min_best"] for r in sorted_results],
            "crosstalk_min_best": [r["crosstalk_min_best"] for r in sorted_results],
            "signal_per_key_best": [r.get("signal_per_key_best", []) for r in sorted_results],
            "crosstalk_per_key_best": [r.get("crosstalk_per_key_best", []) for r in sorted_results],
            "var_max_best": [r.get("var_max_best", 0) for r in sorted_results],
            "E_col_i_star_best": [r.get("E_col_i_star_best", 0) for r in sorted_results],
            "a_star_norm_sq_best": [r.get("a_star_norm_sq_best", 0) for r in sorted_results],
            "var_i_star_j_star_best": [r.get("var_i_star_j_star_best", 0) for r in sorted_results],
            "u_hat_int_sq_baseline_best": [r.get("u_hat_int_sq_baseline_best", 0) for r in sorted_results],
            "K_hat_diag_i_star_best": [r.get("K_hat_diag_i_star_best", 0) for r in sorted_results],
            "signal_dot_i_star_best": [r.get("signal_dot_i_star_best", 0) for r in sorted_results],
            "crosstalk_at_i_star_best": [r.get("crosstalk_at_i_star_best", 0) for r in sorted_results],
            "K_hat_diag_mean_best": [r.get("K_hat_diag_mean_best", 0) for r in sorted_results],
            "mu_V_mean_best": [r.get("mu_V_mean_best", 0) for r in sorted_results],
            "a_norm_sq_mean_best": [r.get("a_norm_sq_mean_best", 0) for r in sorted_results],
            "E_col_mean_best": [r.get("E_col_mean_best", 0) for r in sorted_results],
            "var_mean_best": [r.get("var_mean_best", 0) for r in sorted_results],
            "mu_int_sq_mean_best": [r.get("mu_int_sq_mean_best", 0) for r in sorted_results],
            "rho_V_best": [r.get("rho_V_best") for r in sorted_results],
            "signal_dot_mean_best": [r.get("signal_dot_mean_best", 0) for r in sorted_results],
            "E_Y_best": [r.get("E_Y_best", 0.0) for r in sorted_results],
            "E_Y_mean_best": [r.get("E_Y_mean_best", 0.0) for r in sorted_results],
            "L_v_best": [r.get("L_v_best", 0.0) for r in sorted_results],
            "kappa_max_best": [r.get("kappa_max_best", 0.0) for r in sorted_results],
            "kappa_mean_best": [r.get("kappa_mean_best", 0.0) for r in sorted_results],
            "A_min_best": [r.get("A_min_best", 0.0) for r in sorted_results],
            "B_max_best": [r.get("B_max_best", 0.0) for r in sorted_results],
            "B_Y_best": [r.get("B_Y_best", 0.0) for r in sorted_results],
            "K_max_off_best": [r.get("K_max_off_best", 0.0) for r in sorted_results],
            "n_values": [r["n"] for r in sorted_results],
            "G_values": [r["G"] for r in sorted_results],
            # Noisy-query quantities (present when epsilon > 0)
            "gamma_min_noisy_best": [r.get("gamma_min_noisy_best") for r in sorted_results],
            "noisy_bound_best": [r.get("noisy_bound_best") for r in sorted_results],
            "L_bil_best": [r.get("L_bil_best") for r in sorted_results],
        }
        return out

    elif sweep_type == "M":
        M_values = sorted(set(r["M"] for r in results))
        F = config.F
        num_params = [3 * M * d for M in M_values]
        sorted_results = sorted(results, key=lambda r: r["M"])
        epsilon_val = sorted_results[0].get("epsilon", 0.0) if sorted_results else 0.0
        out = {
            "sweep_type": "M",
            "d": d,
            "F": F,
            "epsilon": epsilon_val,
            "M_values": M_values,
            "num_params": num_params,
            "gamma_min_best": [r["gamma_min_best"] for r in sorted_results],
            "gamma_min_std": [r["gamma_min_std"] for r in sorted_results],
            "gamma_min_all": [r["gamma_min_all"] for r in sorted_results],
            "accuracy_best": [r["accuracy_best"] for r in sorted_results],
            "E_col_max_best": [r["E_col_max_best"] for r in sorted_results],
            "K_hat_diag_min_best": [r["K_hat_diag_min_best"] for r in sorted_results],
            "K_hat_diag_max_best": [r.get("K_hat_diag_max", r["K_hat_diag_min_best"]) for r in sorted_results],
            "delta_min_best": [r["delta_min_best"] for r in sorted_results],
            "mu_V_best": [r["mu_V_best"] for r in sorted_results],
            "d_int_best": [r["d_int_best"] for r in sorted_results],
            "signal_min_best": [r["signal_min_best"] for r in sorted_results],
            "crosstalk_min_best": [r["crosstalk_min_best"] for r in sorted_results],
            "signal_per_key_best": [r.get("signal_per_key_best", []) for r in sorted_results],
            "crosstalk_per_key_best": [r.get("crosstalk_per_key_best", []) for r in sorted_results],
            "var_max_best": [r.get("var_max_best", 0) for r in sorted_results],
            "E_col_i_star_best": [r.get("E_col_i_star_best", 0) for r in sorted_results],
            "a_star_norm_sq_best": [r.get("a_star_norm_sq_best", 0) for r in sorted_results],
            "var_i_star_j_star_best": [r.get("var_i_star_j_star_best", 0) for r in sorted_results],
            "u_hat_int_sq_baseline_best": [r.get("u_hat_int_sq_baseline_best", 0) for r in sorted_results],
            "K_hat_diag_i_star_best": [r.get("K_hat_diag_i_star_best", 0) for r in sorted_results],
            "signal_dot_i_star_best": [r.get("signal_dot_i_star_best", 0) for r in sorted_results],
            "crosstalk_at_i_star_best": [r.get("crosstalk_at_i_star_best", 0) for r in sorted_results],
            "K_hat_diag_mean_best": [r.get("K_hat_diag_mean_best", 0) for r in sorted_results],
            "mu_V_mean_best": [r.get("mu_V_mean_best", 0) for r in sorted_results],
            "a_norm_sq_mean_best": [r.get("a_norm_sq_mean_best", 0) for r in sorted_results],
            "E_col_mean_best": [r.get("E_col_mean_best", 0) for r in sorted_results],
            "var_mean_best": [r.get("var_mean_best", 0) for r in sorted_results],
            "mu_int_sq_mean_best": [r.get("mu_int_sq_mean_best", 0) for r in sorted_results],
            "rho_V_best": [r.get("rho_V_best") for r in sorted_results],
            "signal_dot_mean_best": [r.get("signal_dot_mean_best", 0) for r in sorted_results],
            "E_Y_best": [r.get("E_Y_best", 0.0) for r in sorted_results],
            "E_Y_mean_best": [r.get("E_Y_mean_best", 0.0) for r in sorted_results],
            "L_v_best": [r.get("L_v_best", 0.0) for r in sorted_results],
            "kappa_max_best": [r.get("kappa_max_best", 0.0) for r in sorted_results],
            "kappa_mean_best": [r.get("kappa_mean_best", 0.0) for r in sorted_results],
            "A_min_best": [r.get("A_min_best", 0.0) for r in sorted_results],
            "B_max_best": [r.get("B_max_best", 0.0) for r in sorted_results],
            "B_Y_best": [r.get("B_Y_best", 0.0) for r in sorted_results],
            "K_max_off_best": [r.get("K_max_off_best", 0.0) for r in sorted_results],
            "n_values": [r["n"] for r in sorted_results],
            "G_values": [r["G"] for r in sorted_results],
            # Noisy-query quantities (present when epsilon > 0)
            "gamma_min_noisy_best": [r.get("gamma_min_noisy_best") for r in sorted_results],
            "noisy_bound_best": [r.get("noisy_bound_best") for r in sorted_results],
            "L_bil_best": [r.get("L_bil_best") for r in sorted_results],
        }
        return out

    elif sweep_type == "beta":
        F = config.F
        M = config.M
        sorted_results = sorted(results, key=lambda r: r["beta"])
        beta_values = [r["beta"] for r in sorted_results]
        out = {
            "sweep_type": "beta",
            "spike_target": config.spike_target,
            "d": d,
            "F": F,
            "M": M,
            "beta_values": beta_values,
            "coherence_keys_best": [r.get("coherence_keys", 0.0) for r in sorted_results],
            "coherence_values_best": [r.get("coherence_values", 0.0) for r in sorted_results],
            "gamma_min_best": [r["gamma_min_best"] for r in sorted_results],
            "gamma_min_std": [r["gamma_min_std"] for r in sorted_results],
            "gamma_min_all": [r["gamma_min_all"] for r in sorted_results],
            "accuracy_best": [r["accuracy_best"] for r in sorted_results],
            "E_col_max_best": [r["E_col_max_best"] for r in sorted_results],
            "K_hat_diag_min_best": [r["K_hat_diag_min_best"] for r in sorted_results],
            "K_hat_diag_max_best": [r.get("K_hat_diag_max", r["K_hat_diag_min_best"]) for r in sorted_results],
            "delta_min_best": [r["delta_min_best"] for r in sorted_results],
            "mu_V_best": [r["mu_V_best"] for r in sorted_results],
            "d_int_best": [r["d_int_best"] for r in sorted_results],
            "signal_min_best": [r["signal_min_best"] for r in sorted_results],
            "crosstalk_min_best": [r["crosstalk_min_best"] for r in sorted_results],
            "signal_per_key_best": [r.get("signal_per_key_best", []) for r in sorted_results],
            "crosstalk_per_key_best": [r.get("crosstalk_per_key_best", []) for r in sorted_results],
            "var_max_best": [r.get("var_max_best", 0) for r in sorted_results],
            "E_col_i_star_best": [r.get("E_col_i_star_best", 0) for r in sorted_results],
            "a_star_norm_sq_best": [r.get("a_star_norm_sq_best", 0) for r in sorted_results],
            "var_i_star_j_star_best": [r.get("var_i_star_j_star_best", 0) for r in sorted_results],
            "u_hat_int_sq_baseline_best": [r.get("u_hat_int_sq_baseline_best", 0) for r in sorted_results],
            "K_hat_diag_i_star_best": [r.get("K_hat_diag_i_star_best", 0) for r in sorted_results],
            "signal_dot_i_star_best": [r.get("signal_dot_i_star_best", 0) for r in sorted_results],
            "crosstalk_at_i_star_best": [r.get("crosstalk_at_i_star_best", 0) for r in sorted_results],
            "K_hat_diag_mean_best": [r.get("K_hat_diag_mean_best", 0) for r in sorted_results],
            "mu_V_mean_best": [r.get("mu_V_mean_best", 0) for r in sorted_results],
            "a_norm_sq_mean_best": [r.get("a_norm_sq_mean_best", 0) for r in sorted_results],
            "E_col_mean_best": [r.get("E_col_mean_best", 0) for r in sorted_results],
            "var_mean_best": [r.get("var_mean_best", 0) for r in sorted_results],
            "mu_int_sq_mean_best": [r.get("mu_int_sq_mean_best", 0) for r in sorted_results],
            "rho_V_best": [r.get("rho_V_best") for r in sorted_results],
            "signal_dot_mean_best": [r.get("signal_dot_mean_best", 0) for r in sorted_results],
            "E_Y_best": [r.get("E_Y_best", 0.0) for r in sorted_results],
            "E_Y_mean_best": [r.get("E_Y_mean_best", 0.0) for r in sorted_results],
            "L_v_best": [r.get("L_v_best", 0.0) for r in sorted_results],
            "kappa_max_best": [r.get("kappa_max_best", 0.0) for r in sorted_results],
            "kappa_mean_best": [r.get("kappa_mean_best", 0.0) for r in sorted_results],
            "A_min_best": [r.get("A_min_best", 0.0) for r in sorted_results],
            "B_max_best": [r.get("B_max_best", 0.0) for r in sorted_results],
            "B_Y_best": [r.get("B_Y_best", 0.0) for r in sorted_results],
            "K_max_off_best": [r.get("K_max_off_best", 0.0) for r in sorted_results],
            "n_values": [r["n"] for r in sorted_results],
            "G_values": [r["G"] for r in sorted_results],
            # Noisy-query quantities
            "gamma_min_noisy_best": [r.get("gamma_min_noisy_best") for r in sorted_results],
            "noisy_bound_best": [r.get("noisy_bound_best") for r in sorted_results],
            "L_bil_best": [r.get("L_bil_best") for r in sorted_results],
        }
        return out

    elif sweep_type == "epsilon":
        F = config.F
        M = config.M
        sorted_results = sorted(results, key=lambda r: r["epsilon"])
        epsilon_values = [r["epsilon"] for r in sorted_results]
        out = {
            "sweep_type": "epsilon",
            "d": d,
            "F": F,
            "M": M,
            "epsilon_values": epsilon_values,
            "gamma_min_best": [r["gamma_min_best"] for r in sorted_results],
            "gamma_min_std": [r["gamma_min_std"] for r in sorted_results],
            "gamma_min_all": [r["gamma_min_all"] for r in sorted_results],
            "accuracy_best": [r["accuracy_best"] for r in sorted_results],
            "E_col_max_best": [r["E_col_max_best"] for r in sorted_results],
            "K_hat_diag_min_best": [r["K_hat_diag_min_best"] for r in sorted_results],
            "delta_min_best": [r["delta_min_best"] for r in sorted_results],
            "mu_V_best": [r["mu_V_best"] for r in sorted_results],
            "d_int_best": [r["d_int_best"] for r in sorted_results],
            "n_values": [r["n"] for r in sorted_results],
            "G_values": [r["G"] for r in sorted_results],
            # Noisy-query quantities
            "gamma_min_noisy_best": [r.get("gamma_min_noisy_best") for r in sorted_results],
            "noisy_bound_best": [r.get("noisy_bound_best") for r in sorted_results],
            "L_bil_best": [r.get("L_bil_best") for r in sorted_results],
        }
        return out

    else:
        # FM sweep: return full grid
        F_values = sorted(set(r["F"] for r in results))
        M_values = sorted(set(r["M"] for r in results))
        grid = {}
        for r in results:
            grid[f"{r['F']}_{r['M']}"] = {
                "F": r["F"],
                "M": r["M"],
                "n": r["n"],
                "G": r["G"],
                "gamma_min_best": r["gamma_min_best"],
                "gamma_min_std": r["gamma_min_std"],
                "accuracy_best": r["accuracy_best"],
                "E_col_max_best": r["E_col_max_best"],
                "K_hat_diag_min_best": r["K_hat_diag_min_best"],
                "delta_min_best": r["delta_min_best"],
                "mu_V_best": r["mu_V_best"],
                "d_int_best": r["d_int_best"],
                "signal_min_best": r.get("signal_min_best", 0),
                "crosstalk_min_best": r.get("crosstalk_min_best", 0),
                "var_max_best": r.get("var_max_best", 0),
                "E_col_i_star_best": r.get("E_col_i_star_best", 0),
                "a_star_norm_sq_best": r.get("a_star_norm_sq_best", 0),
                "var_i_star_j_star_best": r.get("var_i_star_j_star_best", 0),
                "u_hat_int_sq_baseline_best": r.get("u_hat_int_sq_baseline_best", 0),
                "K_hat_diag_i_star_best": r.get("K_hat_diag_i_star_best", 0),
                "signal_dot_i_star_best": r.get("signal_dot_i_star_best", 0),
                "crosstalk_at_i_star_best": r.get("crosstalk_at_i_star_best", 0),
                "K_hat_diag_mean_best": r.get("K_hat_diag_mean_best", 0),
                "mu_V_mean_best": r.get("mu_V_mean_best", 0),
                "a_norm_sq_mean_best": r.get("a_norm_sq_mean_best", 0),
                "E_col_mean_best": r.get("E_col_mean_best", 0),
                "var_mean_best": r.get("var_mean_best", 0),
                "mu_int_sq_mean_best": r.get("mu_int_sq_mean_best", 0),
                "rho_V_best": r.get("rho_V_best"),
                "signal_dot_mean_best": r.get("signal_dot_mean_best", 0),
                "E_Y_best": r.get("E_Y_best", 0.0),
                "L_v_best": r.get("L_v_best", 0.0),
                "param_count": r.get("param_count", 3 * r["M"] * r["d"]),
            }
        return {
            "sweep_type": "FM",
            "d": d,
            "F_values": F_values,
            "M_values": M_values,
            "grid": grid,
        }


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


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


def _run_grid_searches_locally(
    configs: list[MarginSweepGridSearchConfig],
) -> list[dict | None]:
    """Run margin sweeps sequentially on the local device."""

    all_results = []
    for grid_config in configs:
        start_time = time.time()
        job_results = []
        for prop_values in _sweep_prop_product(grid_config.sweep_props):
            exp_config, experiment_base_dir = grid_config._get_experiment_config_and_base_dir(
                **prop_values
            )
            os.makedirs(experiment_base_dir, exist_ok=True)
            try:
                result = grid_config.run_experiment_config(exp_config)
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
        aggregated_result = grid_config.agg_results(job_results)
        all_results.append(aggregated_result)

        os.makedirs(grid_config.base_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_filename = f"{grid_config.base_dir}/grid_search_results_{timestamp}.pkl"
        with open(results_filename, "wb") as f:
            pickle.dump(
                {
                    "results": aggregated_result,
                    "total_time": time.time() - start_time,
                    "timestamp": timestamp,
                },
                f,
            )
        print(f"Grid search results saved to: {results_filename}", flush=True)
    return all_results


def run(config: MarginSweepRunnerConfig):
    """Main entry point: build one GridSearchConfig per (F,M) point and run all."""
    bundle_info = None
    if config.embeddings_dir is not None:
        bundle_info = inspect_bundle(config.embeddings_dir)
        config.d = int(bundle_info.d_model)
        print(
            "Using LLM activation rows: "
            f"{bundle_info.activation_dir} "
            f"({bundle_info.num_pairs} rows, d={bundle_info.d_model})"
        )

    # Build base dir
    if config.base_dir is None:
        timestamp = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
        base_dir = f"./results/margin_sweep_{config.sweep}_{timestamp}"
    else:
        base_dir = config.base_dir

    # Build one GridSearchConfig per (F, M) grid point
    grid_configs = build_grid_search_configs(config, base_dir)

    grid_points = get_grid_points(config)
    total_jobs = len(grid_points) * config.n_seeds

    print(f"Margin Sweep ({config.sweep})")
    print(f"  d={config.d}, device={config.device}")
    if config.embeddings_dir is not None:
        print(f"  embeddings_dir={config.embeddings_dir}")
    print(f"  Grid points: {len(grid_points)}, Seeds per point: {config.n_seeds}")
    print(f"  Total jobs: {total_jobs}")
    print(f"  Base dir: {base_dir}")

    # Run all grid searches (one per grid point, each sweeping seeds).
    if int(config.max_gpus) <= 0:
        print("  Running sequential local fallback because max_gpus <= 0")
        all_results = _run_grid_searches_locally(grid_configs)
    else:
        all_results = run_grid_searches(
            grid_configs,
            max_gpus=config.max_gpus,
            simultaneous_jobs_per_gpu=config.simultaneous_jobs_per_gpu,
        )

    # all_results is a list with one element per config (i.e., per grid point)
    # Each element is what agg_results returned: a dict with best-seed summary
    sweep_results = aggregate_to_sweep_results(all_results, config)
    if bundle_info is not None:
        sweep_results["source"] = "llm_embeddings"
        sweep_results["embeddings_dir"] = str(bundle_info.activation_dir)

    # Save JSON
    output_json = config.output_json
    if output_json is None:
        os.makedirs(base_dir, exist_ok=True)
        output_json = f"{base_dir}/margin_sweep_results.json"

    serializable = convert_to_serializable(sweep_results)
    with open(output_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nJSON results saved to: {output_json}")

    return sweep_results


@pydra_main(MarginSweepRunnerConfig)
def main(config: MarginSweepRunnerConfig):
    return run(config)


if __name__ == "__main__":
    main()
