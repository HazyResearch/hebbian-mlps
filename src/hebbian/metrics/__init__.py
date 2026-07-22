"""
Metrics for Hebbian memory analysis.

This module provides utilities for computing:
    - Margin theory bounds and quantities
    - Value-set separability and optimal separator witnesses
"""

from hebbian.metrics.value_separability import fit_value_separators

from hebbian.metrics.margin_theory import (
    # Building blocks
    compute_d_int,
    compute_rho_and_u_star,
    # Main entry point
    compute_margin_quantities,
    # Bounds
    compute_delta_min_fast,
    compute_mu_V,
    unified_bound,
    unified_bound_from_experiment,
    unified_bound_fitted,
    fit_unified_bound_F_sweep,
    fit_unified_bound_M_sweep,
    schematic_bound,
    fit_schematic_bound_F_sweep,
    fit_schematic_bound_M_sweep,
    # Noisy-query
    compute_lipschitz_bound_bilinear,
    noisy_query_margin_bound,
    compute_noisy_margin,
)

__all__ = [
    "fit_value_separators",
    # Building blocks
    "compute_d_int",
    "compute_rho_and_u_star",
    # Main entry point
    "compute_margin_quantities",
    # Bounds
    "compute_delta_min_fast",
    "compute_mu_V",
    "unified_bound",
    "unified_bound_from_experiment",
    "unified_bound_fitted",
    "fit_unified_bound_F_sweep",
    "fit_unified_bound_M_sweep",
    "schematic_bound",
    "fit_schematic_bound_F_sweep",
    "fit_schematic_bound_M_sweep",
    # Noisy-query
    "compute_lipschitz_bound_bilinear",
    "noisy_query_margin_bound",
    "compute_noisy_margin",
]
