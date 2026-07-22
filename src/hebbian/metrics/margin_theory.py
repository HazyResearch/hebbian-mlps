"""
Theoretical margin bounds for Hebbian MLPs.

Implements the coherence-based margin bound:

  gamma_{i->j} >= min_i(K_hat_{ii}) * (1 - mu(V))
                   - 2*sqrt(2) * sqrt(E_col * L / (G * d_int))

where:
  - K_hat_diag_min = min_i K_hat_{ii} (minimum diagonal of feature Gram matrix)
  - mu(V)   = max_{a!=b} |<v_a, v_b>| (coherence of value embeddings)
  - E_col    = max_i sum_{t!=i} K_hat_{ti}^2 (maximum crowding energy)
  - L        = log(n^2 / delta) (union-bound log factor)
  - G        = number of micro-heads (=1 when no head averaging)
  - d_int    = effective interference dimension (1/mu_int^2)

Also implements the schematic bound for uniform spherical keys/values.
"""

import numpy as np
import torch
import torch.nn.functional as F_func
from typing import Tuple, Optional, Dict
import warnings

try:
    from scipy.optimize import curve_fit
except ModuleNotFoundError:
    # Optional dependency: bound-fitting helpers can gracefully fall back to
    # initial guesses when SciPy is unavailable.
    def curve_fit(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("scipy is not installed")

from hebbian.data.synthetics.factsets import Factset
from hebbian.methods.hebbian.model import HebbianMLP
from hebbian.metrics.value_separability import fit_value_separators

def _reduce_gamma_min(
    gamma_per_key: torch.Tensor,
    percentile: Optional[float] = None,
) -> float:
    """Return strict min margin or a requested percentile of per-key margins."""

    if percentile is None:
        return float(gamma_per_key.min().item())
    if not (0.0 <= float(percentile) <= 100.0):
        raise ValueError(
            f"gamma_min_percentile must be in [0, 100]; got {percentile}"
        )
    return float(np.percentile(gamma_per_key.detach().cpu().numpy(), float(percentile)))


# =============================================================================
# Effective Interference Dimension
# =============================================================================


def compute_d_int(
    K_hat: torch.Tensor,
    code_vectors: torch.Tensor,
    V_all: torch.Tensor,
    value_indices: list,
) -> float:
    """
    Compute the effective interference dimension d_int.

    Using codes c_{f(t)} (either values v or separators u*):
      w_{t,i} = K_hat_{ti}^2 / E_col(i)
      Sigma_i = sum_{t!=i} w_{t,i} * c_{f(t)} @ c_{f(t)}^T
      mu_int^2 = max_i max_{j!=f(i)} a_hat_{i->j}^T @ Sigma_i @ a_hat_{i->j}
      d_int = 1 / mu_int^2

    Args:
        K_hat: (n, n) feature Gram matrix
        code_vectors: (n, d_v) per-key code vectors c_{f(t)} (values or separators)
        V_all: (P, d_v) all value embeddings
        value_indices: list of length n, value_indices[t] = class index f(t)

    Returns:
        d_int: effective interference dimension (in [1, d_v])
    """
    n = K_hat.shape[0]
    P = V_all.shape[0]

    # Off-diagonal K_hat^2
    K_hat_sq = K_hat ** 2
    mask = torch.eye(n, dtype=torch.bool, device=K_hat.device)
    K_hat_sq_offdiag = K_hat_sq.clone()
    K_hat_sq_offdiag.masked_fill_(mask, 0.0)
    E_col = K_hat_sq_offdiag.sum(dim=0)  # (n,)

    mu_int_sq = 0.0

    for i in range(n):
        if E_col[i].item() < 1e-30:
            continue

        # Weights w_{t,i} = K_hat_{ti}^2 / E_col(i) for t != i
        w = K_hat_sq_offdiag[:, i] / E_col[i]  # (n,), w[i]=0

        # Sigma_i = sum_{t!=i} w_{t,i} * c_{f(t)} @ c_{f(t)}^T
        # Efficient: Sigma_i = C_w^T @ C_w where C_w[t,:] = sqrt(w[t]) * code_vectors[t,:]
        sqrt_w = torch.sqrt(w).unsqueeze(1)  # (n, 1)
        V_w = sqrt_w * code_vectors  # (n, d_v)
        Sigma_i = V_w.T @ V_w  # (d_v, d_v)

        fi = value_indices[i]
        for j in range(P):
            if j == fi:
                continue
            a_ij = V_all[fi] - V_all[j]
            norm_a = torch.norm(a_ij)
            if norm_a < 1e-30:
                continue
            a_hat = a_ij / norm_a
            quad = (a_hat @ Sigma_i @ a_hat).item()
            if quad > mu_int_sq:
                mu_int_sq = quad

    if mu_int_sq < 1e-30:
        return float(V_all.shape[1])  # fallback to d_v

    return 1.0 / mu_int_sq


# =============================================================================
# Decodability (rho) and Optimal Separators (u*)
# =============================================================================


def compute_rho_and_u_star(
    V_all: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    admm_rho: float = 1.0,
    admm_n_iters: int = 300,
    admm_batch_size: int = 256,
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Compute optimal separators u* and decodability rho(V).

    Solves: max_{u_i} min_{j!=i} <v_i - v_j, u_i> / ||v_i - v_j||
            s.t. ||u_i|| <= 1

    Args:
        V_all: (P, d_v) all distinct value embeddings (unit normalized)
        device, dtype: computation device/dtype
        admm_rho: ADMM penalty parameter
        admm_n_iters: number of ADMM iterations
        admm_batch_size: batch size for batched ADMM

    Returns:
        U_all: (P, d_v) optimal separator directions (unit normalized)
        rho_V: scalar rho(V) = min_i rho_i(V)
        rho_per_i: (P,) per-class decodability rho_i(V)
    """
    V_dev = V_all.to(device=device, dtype=dtype)
    U_all, rho_V, rho_per_i = fit_value_separators(
        V_dev,
        batch_size=admm_batch_size,
        rho=admm_rho,
        num_iters=admm_n_iters,
    )
    return U_all, rho_V, rho_per_i


# =============================================================================
# Margin Quantity Computation from HebbianMLP + Factset
# =============================================================================


def _reduce_gamma_min(gamma_per_key: torch.Tensor, percentile: Optional[float]) -> float:
    """Reduce per-key margins to a single ``gamma_min``.

    When ``percentile`` is None, returns the strict minimum (the default
    "worst-key" margin). When set to a value in [0, 100], returns the
    requested percentile of gamma_per_key — useful when a few outlier keys
    dominate the strict min (e.g., on highly anisotropic captured data).
    """
    if percentile is None:
        return float(gamma_per_key.min().item())
    if not (0.0 <= float(percentile) <= 100.0):
        raise ValueError(
            f"gamma_min_percentile must be in [0, 100]; got {percentile}"
        )
    gpk = gamma_per_key.detach().cpu().numpy()
    return float(np.percentile(gpk, float(percentile)))


def compute_margin_quantities(
    mlp: HebbianMLP,
    factset: Factset,
    device: torch.device,
    dtype: torch.dtype,
    delta: float = 0.5,
    G: int = 1,
    use_u_star_codes: bool = False,
    U_all: Optional[torch.Tensor] = None,
    rho_V: Optional[float] = None,
    rho_per_i: Optional[torch.Tensor] = None,
    gamma_min_percentile: Optional[float] = None,
) -> Dict[str, object]:
    """
    Compute all margin-related quantities from a HebbianMLP and Factset.

    When use_u_star_codes=False (default): uses V as codes (U = V).
    When use_u_star_codes=True: uses optimal separators u* as codes.

    Returns dict with:
      Scalars: gamma_min, accuracy, E_col_max, K_hat_diag_min, delta_min, mu_V, d_int,
               n, G, L, unified_bound, K_hat_diag_max, rho_V, signal_dot_mean
      Tensors: k_hat (n,n), V (n,d_v), gamma_per_key (n,),
               signal_per_key (n,), crosstalk_per_key (n,),
               signal_matrix (n,P), crosstalk_matrix (n,P)

    Signal/cross-talk decomposition (with codes c = u* or v):
      gamma_{i,j} = s_{i,j} + c_{i,j}
      s_{i,j} = K_hat_{ii} * <v_{f(i)} - v_j, c_{f(i)}>
      c_{i,j} = sum_{t!=i} K_hat_{ti} * <v_{f(i)} - v_j, c_{f(t)}>

    signal_per_key[i] and crosstalk_per_key[i] are at the worst competitor j*(i).
    signal_matrix[i,j] and crosstalk_matrix[i,j] give the full (n, P) decomposition
    (correct-class entries are masked: signal=inf, crosstalk=0).
    """
    with torch.no_grad():
        K = factset.input_embeddings.to(device=device, dtype=dtype)
        n = K.shape[0]
        value_indices = [factset.mapping.get_output(i) for i in range(n)]
        V_map = factset.output_embeddings[value_indices].to(device=device, dtype=dtype)
        V_all = factset.output_embeddings.to(device=device, dtype=dtype)

        # Compute features and Gram matrix
        Phi = mlp.feature_map(K)  # (n, m)
        K_hat = Phi @ Phi.T  # (n, n)

        diag = torch.diag(K_hat)
        K_hat_diag_min = diag.min().item()
        K_hat_diag_mean = diag.mean().item()

        # E_col_max = max_i sum_{t!=i} K_hat_{ti}^2
        K_hat_sq = K_hat ** 2
        eye_mask = torch.eye(n, dtype=torch.bool, device=device)
        K_hat_sq.masked_fill_(eye_mask, 0.0)
        E_col = K_hat_sq.sum(dim=0)  # (n,)
        E_col_max = E_col.max().item()
        E_col_mean = E_col.mean().item()

        # delta_min = min_{a!=b} ||v_a - v_b|| over the P distinct value classes
        V_gram_all = V_all @ V_all.T  # (P, P)

        # mu_V = max_{a!=b} |<v_a, v_b>| (coherence of value embeddings)
        V_gram_offdiag = V_gram_all.clone()
        V_gram_offdiag.fill_diagonal_(0.0)
        mu_V = V_gram_offdiag.abs().max().item()

        # Mean quantities for diagnostic pseudo-bounds
        P_val = V_all.shape[0]
        P_pairs = P_val * (P_val - 1)  # number of ordered off-diagonal pairs
        mu_V_mean = (V_gram_offdiag.abs().sum() / P_pairs).item()
        mean_inner_prod = (V_gram_offdiag.sum() / P_pairs).item()
        a_norm_sq_mean = 2.0 - 2.0 * mean_inner_prod  # mean ||v_a - v_b||^2

        V_gram_all.fill_diagonal_(-float('inf'))
        max_inner = V_gram_all.max().item()
        delta_min = np.sqrt(max(0.0, 2 - 2 * max_inner))

        # =================================================================
        # Code vector selection: V_map (U=V) or U_map (u*-codes)
        # =================================================================
        if use_u_star_codes:
            if U_all is None:
                U_all_local, rho_V_local, rho_per_i_local = \
                    compute_rho_and_u_star(V_all, device, dtype)
            else:
                U_all_local = U_all.to(device=device, dtype=dtype)
                rho_V_local = rho_V if rho_V is not None else 0.0
                rho_per_i_local = rho_per_i
            U_map = U_all_local[value_indices]  # (n, d_v)
            code_vectors = U_map
        else:
            code_vectors = V_map
            rho_V_local = None
            rho_per_i_local = None

        # d_int: effective interference dimension using chosen codes
        d_int = compute_d_int(K_hat, code_vectors, V_all, value_indices)

        # L = log(n^2 / delta)
        L = np.log(n**2 / delta)

        # Coherence-based bound (always computed for reference)
        bound = unified_bound(
            K_hat_diag_min, mu_V, E_col_max, n, G, d_int, delta
        )

        # =================================================================
        # Compute predictions and per-key margins using chosen codes
        #
        # y_code(k_i) = sum_t K_hat_{ti} * code_vectors[t]
        # scores_{i,j} = <y_code(k_i), v_j>
        # gamma_{i,j} = scores_{i,f(i)} - scores_{i,j}
        # =================================================================
        correct_indices = torch.tensor(value_indices, device=device)
        batch_idx = torch.arange(n, device=device)
        P = V_all.shape[0]

        Y_code = K_hat @ code_vectors  # (n, d_v)
        scores = Y_code @ V_all.T  # (n, P)
        correct_scores = scores[batch_idx, correct_indices]  # (n,)

        # Mask correct class and find max incorrect
        scores_masked = scores.clone()
        scores_masked[batch_idx, correct_indices] = float('-inf')
        max_incorrect_scores = scores_masked.max(dim=1).values  # (n,)

        gamma_per_key = correct_scores - max_incorrect_scores
        gamma_min = _reduce_gamma_min(gamma_per_key, gamma_min_percentile)
        accuracy = (gamma_per_key > 0).float().mean().item()

        # =================================================================
        # Signal / cross-talk decomposition (with codes c_{f(t)})
        #
        # a_{i->j} = v_{f(i)} - v_j
        # s_{i,j} = K_hat_{ii} * <a_{i->j}, c_{f(i)}>
        # c_{i,j} = sum_{t!=i} K_hat_{ti} * <a_{i->j}, c_{f(t)}>
        #
        # Vectorized via R_i = sum_{t!=i} K_hat_{ti} c_{f(t)}:
        #   s_{i,j} = K_hat_{ii} * (<v_{f(i)}, c_{f(i)}> - <v_j, c_{f(i)}>)
        #   c_{i,j} = <v_{f(i)}, R_i> - <v_j, R_i>
        # =================================================================

        # Off-diagonal K_hat (zero diagonal)
        K_hat_offdiag = K_hat.clone()
        K_hat_offdiag.fill_diagonal_(0.0)

        # R[i] = sum_{t!=i} K_hat_{ti} * code_vectors[t]
        R = K_hat_offdiag @ code_vectors  # (n, d_v)

        # Signal: s_{i,j} = K_hat_{ii} * (<v_{f(i)}, c_{f(i)}> - <v_j, c_{f(i)}>)
        correct_code_dot = (V_map * code_vectors).sum(dim=1)  # (n,) <v_{f(i)}, c_{f(i)}>
        all_code_dot = code_vectors @ V_all.T  # (n, P) <c_{f(i)}, v_j>
        signal_matrix = diag.unsqueeze(1) * (correct_code_dot.unsqueeze(1) - all_code_dot)  # (n, P)

        # Cross-talk: c_{i,j} = <v_{f(i)}, R_i> - <v_j, R_i>
        correct_R_dot = (V_map * R).sum(dim=1)  # (n,) <v_{f(i)}, R_i>
        all_R_dot = R @ V_all.T  # (n, P) <v_j, R_i>
        crosstalk_matrix = correct_R_dot.unsqueeze(1) - all_R_dot  # (n, P)

        # Mask out the correct class (j = f(i))
        signal_matrix[batch_idx, correct_indices] = float('inf')
        crosstalk_matrix[batch_idx, correct_indices] = 0.0

        # Per-key: signal and cross-talk at the worst (min-margin) competitor
        gamma_matrix = signal_matrix + crosstalk_matrix
        gamma_matrix[batch_idx, correct_indices] = float('inf')
        worst_j = gamma_matrix.argmin(dim=1)  # (n,)
        signal_per_key = signal_matrix[batch_idx, worst_j]  # (n,)
        crosstalk_per_key = crosstalk_matrix[batch_idx, worst_j]  # (n,)

        # =================================================================
        # Variance term: var_{i,j} = sum_{t!=i} K_hat_{ti}^2 * <a_hat_{i->j}, c_{f(t)}>^2
        # where a_hat_{i->j} = (v_{f(i)} - v_j) / ||v_{f(i)} - v_j||
        # =================================================================
        A = V_map.unsqueeze(1) - V_all.unsqueeze(0)  # (n, P, d_v)
        A_norms = A.norm(dim=2, keepdim=True).clamp(min=1e-12)  # (n, P, 1)
        A_hat = A / A_norms  # (n, P, d_v) normalized directions

        # <a_hat_{i,j}, c_{f(t)}> for each t
        AC = torch.einsum('ipd,td->ipt', A_hat, code_vectors)  # (n, P, n)
        AC_sq = AC ** 2  # (n, P, n)

        # var_{i,j} = sum_t K_hat_sq[t,i] * AC_sq[i,j,t]
        var_matrix = torch.einsum('ti,ipt->ip', K_hat_sq, AC_sq)  # (n, P)

        # Mask correct-class entries
        var_matrix[batch_idx, correct_indices] = 0.0
        var_max = var_matrix.max().item()
        var_mean = (var_matrix.sum() / (n * (P - 1))).item()

        # mu_int^2 per (i,j): var_{i,j} / (||a_{ij}||^2 * E_col(i))
        safe_E_col = E_col.clamp(min=1e-30)  # (n,)
        mu_int_sq_matrix = var_matrix / safe_E_col.unsqueeze(1)  # (n, P)
        mu_int_sq_matrix[batch_idx, correct_indices] = 0.0  # re-mask
        mu_int_sq_mean = (mu_int_sq_matrix.sum() / (n * (P - 1))).item()

        # =================================================================
        # Baseline quantities at worst-margin pair (i*, j*)
        # =================================================================
        i_star = gamma_per_key.argmin().item()
        j_star = worst_j[i_star].item()

        # E_col(i*)
        E_col_i_star = E_col[i_star].item()

        # ||v_{f(i*)} - v_{j*}||^2
        a_star = V_map[i_star] - V_all[j_star]
        a_star_norm_sq = (a_star @ a_star).item()

        # V_{i*,j*} — variance at worst pair
        var_i_star_j_star = var_matrix[i_star, j_star].item()

        # Baseline u_hat_int^2 = V_{i*j*} / (||a*||^2 * E_col(i*))
        denom = a_star_norm_sq * E_col_i_star
        u_hat_int_sq_baseline = var_i_star_j_star / max(denom, 1e-30)

        # K_hat_{i*,i*}
        K_hat_diag_i_star = diag[i_star].item()

        # (v_{f(i*)} - v_{j*})^T c_{f(i*)} — exact signal dot product
        signal_dot_i_star = (a_star @ code_vectors[i_star]).item()

        # Exact cross-talk at (i*, j*)
        crosstalk_at_i_star = crosstalk_per_key[i_star].item()

        # =================================================================
        # E_Y: Y-energy = max_{i, j!=f(i)} ||Y^{(ij)}||^2
        #
        # Y^{(ij)}_t = <v_{f(i)} - v_j, c_{f(t)}> for t != i
        # ||Y^{(ij)}||^2 = sum_{t!=i} (<a_ij, c_{f(t)}>)^2
        #               = a_ij^T M_full a_ij - (a_ij^T c_{f(i)})^2
        # where M_full = sum_t c_{f(t)} c_{f(t)}^T = code_vectors^T @ code_vectors
        # a_ij = v_{f(i)} - v_j = V_map[i] - V_all[j]
        # =================================================================
        M_full = code_vectors.T @ code_vectors  # (d_v, d_v)
        VM_M = V_map @ M_full                   # (n, d_v)
        VA_M = V_all @ M_full                   # (P, d_v)
        VM_sq = (VM_M * V_map).sum(1)           # (n,)  <v_i, M_full v_i>
        VA_sq = (VA_M * V_all).sum(1)           # (P,)  <v_j, M_full v_j>
        cross_M = VM_M @ V_all.T               # (n, P)  <v_i, M_full v_j>
        Q_EY = VM_sq.unsqueeze(1) + VA_sq.unsqueeze(0) - 2 * cross_M  # (n, P)

        # Correction: subtract (a_ij^T c_{f(i)})^2 = (<v_i, c_i> - <v_j, c_i>)^2
        alpha_c = (code_vectors * V_map).sum(1)    # (n,)  <v_i, c_i>
        beta_c = code_vectors @ V_all.T            # (n, P) <c_i, v_j>
        corr_EY = (alpha_c.unsqueeze(1) - beta_c) ** 2  # (n, P)

        EY_mat = (Q_EY - corr_EY).clamp(min=0.0)
        EY_mat[batch_idx, correct_indices] = 0.0
        E_Y = EY_mat.max().item()
        E_Y_mean = (EY_mat.sum() / (n * (P - 1))).item()

        # =================================================================
        # L_v: max_{i, j!=f(i)} ||Y^{(ij)}||_1^2 / ||Y^{(ij)}||_2^2
        #
        # Y^{(ij)}_t = <v_{f(i)} - v_j, c_{f(t)}> for t != i
        # Computed via Y_full[i,j,t] = <v_i, c_t> - <v_j, c_t>,
        # then zeroing t=i to exclude the self-term.
        # =================================================================
        VmC = V_map @ code_vectors.T        # (n, n): <v_i, c_t>
        VaC = V_all @ code_vectors.T        # (P, n): <v_j, c_t>
        Y_full = VmC.unsqueeze(1) - VaC.unsqueeze(0)  # (n, P, n)
        range_n = torch.arange(n, device=device)
        Y_full[range_n, :, range_n] = 0.0  # zero out t=i self-term
        Y_l1_sq = Y_full.abs().sum(dim=2) ** 2  # (n, P)
        Y_l1_sq[batch_idx, correct_indices] = 0.0
        EY_safe = EY_mat.clamp(min=1e-30)
        Lv_mat = torch.where(EY_mat > 0, Y_l1_sq / EY_safe,
                             torch.zeros_like(Y_l1_sq))
        Lv_mat[batch_idx, correct_indices] = 0.0
        L_v = Lv_mat.max().item()

        # =================================================================
        # kappa: alignment between key-column X^(i) and value-diff Y^(ij)
        #
        # kappa^{ij} = max{0, -<X^(i), Y^(ij)> / (||X^(i)||_2 * ||Y^(ij)||_2)}
        #            = max{0, -crosstalk(i,j) / (sqrt(E_K(i)) * sqrt(EY(i,j)))}
        #
        # kappa   = max_{i, j!=f(i)} kappa^{ij}          (used in deterministic bounds)
        # kappa_mean = mean_{i, j!=f(i)} kappa^{ij}      (used in heuristic bounds)
        # =================================================================
        sqrt_EK = torch.sqrt(safe_E_col)          # (n,)
        sqrt_EY = torch.sqrt(EY_mat.clamp(min=0)) # (n, P)
        denom_kappa = (sqrt_EK.unsqueeze(1) * sqrt_EY).clamp(min=1e-30)  # (n, P)
        # crosstalk_matrix[i, correct_indices[i]] == 0 already, so kappa is 0 there
        kappa_mat = torch.clamp(-crosstalk_matrix / denom_kappa, min=0.0)  # (n, P)
        kappa_mat[batch_idx, correct_indices] = 0.0
        kappa_max = kappa_mat.max().item()
        kappa_mean = (kappa_mat.sum() / (n * (P - 1))).item()

        # =================================================================
        # Mean signal dot: mean_{i, j!=f(i)} (v_i - v_j)^T c_i
        # (replaces mu_V_mean for EV signal bounds when using u*-codes)
        # =================================================================
        if use_u_star_codes:
            # A_hat: (n, P, d_v) normalized; code_vectors: (n, d_v)
            signal_dots = torch.einsum('ipd,id->ip', A_hat, code_vectors)  # (n, P)
            signal_dots[batch_idx, correct_indices] = 0.0
            # mean of unnormalized signal: mean (v_i - v_j)^T c_i
            # = mean ||v_i - v_j|| * (a_hat^T c_i)
            unnorm_signal_dots = torch.einsum('ipd,id->ip', A, code_vectors)  # (n, P)
            unnorm_signal_dots[batch_idx, correct_indices] = 0.0
            signal_dot_mean = (unnorm_signal_dots.sum() / (n * (P - 1))).item()
        else:
            # V-code approximation based on mean value coherence.
            signal_dot_mean = 1.0 - mu_V_mean

        # =================================================================
        # Signal/noise quantities used by the paper bounds.
        #
        # A_min   = min_{i≠j} <v_i - v_j, v_i>
        # B_max   = max_{i≠j} |<v_i - v_j, v_j>|
        # B_Y     = max_{i≠j} |sum_{t≠i} <v_i - v_j, v_t>|
        # K_max_off = max_{i≠j} |K_hat_{ij}|
        # =================================================================
        V_norms_sq_P = (V_all * V_all).sum(dim=1)   # (P,)  ||v_i||^2
        eye_P = torch.eye(P, dtype=torch.bool, device=device)

        # A_min: A_mat[i,j] = ||v_i||^2 - <v_j, v_i>  (= <v_i-v_j, v_i> for unit vectors)
        A_mat = (V_norms_sq_P.unsqueeze(1) - V_gram_offdiag.T).masked_fill(eye_P, float('inf'))
        A_min_val = A_mat.min().item()

        # B_max: B_mat[i,j] = <v_i,v_j> - ||v_j||^2  (= <v_i-v_j, v_j>)
        B_mat = (V_gram_offdiag - V_norms_sq_P.unsqueeze(0)).masked_fill(eye_P, 0.0)
        B_max_val = B_mat.abs().max().item()

        # B_Y: BY_mat[i,j] = <v_i-v_j, S_i>  where S_i = sum_{t≠i} v_t
        # Derivation: <v_i-v_j, S_i> = <v_i,S_i> - <v_j,S-v_i>
        #           = (<v_i,S> - ||v_i||^2) - <v_j,S> + <v_i,v_j>
        S_sum = V_all.sum(dim=0)                               # (d_v,)
        V_S_dot = V_all @ S_sum                                # (P,) <v_i, S>
        V_Si_dot = V_S_dot - V_norms_sq_P                     # (P,) <v_i, S_i>
        BY_mat = (V_Si_dot.unsqueeze(1) - V_S_dot.unsqueeze(0) + V_gram_offdiag.T).masked_fill(eye_P, 0.0)
        B_Y_val = BY_mat.abs().max().item()

        # K_max_off: max off-diagonal absolute value of kernel matrix
        K_hat_abs_off = K_hat.abs().clone()
        K_hat_abs_off.fill_diagonal_(0.0)
        K_max_off_val = K_hat_abs_off.max().item()

    return {
        "gamma_min": gamma_min,
        "accuracy": accuracy,
        "E_col_max": E_col_max,
        "K_hat_diag_min": K_hat_diag_min,
        "delta_min": delta_min,
        "mu_V": mu_V,
        "d_int": d_int,
        "n": n,
        "G": G,
        "L": L,
        "unified_bound": bound,
        "k_hat": K_hat.detach().cpu(),
        "V": V_map.detach().cpu(),
        "gamma_per_key": gamma_per_key.cpu(),
        "signal_per_key": signal_per_key.cpu(),
        "crosstalk_per_key": crosstalk_per_key.cpu(),
        "signal_matrix": signal_matrix.cpu(),
        "crosstalk_matrix": crosstalk_matrix.cpu(),
        "K_hat_diag_max": diag.max().item(),
        "var_max": var_max,
        "E_col_i_star": E_col_i_star,
        "a_star_norm_sq": a_star_norm_sq,
        "var_i_star_j_star": var_i_star_j_star,
        "u_hat_int_sq_baseline": u_hat_int_sq_baseline,
        "K_hat_diag_i_star": K_hat_diag_i_star,
        "signal_dot_i_star": signal_dot_i_star,
        "crosstalk_at_i_star": crosstalk_at_i_star,
        # Mean quantities for diagnostic pseudo-bounds
        "K_hat_diag_mean": K_hat_diag_mean,
        "mu_V_mean": mu_V_mean,
        "a_norm_sq_mean": a_norm_sq_mean,
        "E_col_mean": E_col_mean,
        "var_mean": var_mean,
        "mu_int_sq_mean": mu_int_sq_mean,
        # Decodability quantities (u*-codes)
        "rho_V": rho_V_local,
        "signal_dot_mean": signal_dot_mean,
        # Y-energy (Sec 2.3 bound)
        "E_Y": E_Y,
        "E_Y_mean": E_Y_mean,
        "L_v": L_v,
        # kappa: alignment between key-column and value-diff vectors (Sec 2.3/2.4 bounds)
        "kappa_max": kappa_max,
        "kappa_mean": kappa_mean,
        # tab:2x2 signal/noise quantities
        "A_min": A_min_val,
        "B_max": B_max_val,
        "B_Y": B_Y_val,
        "K_max_off": K_max_off_val,
    }


# =============================================================================
# Coherence-Based Margin Bound
# =============================================================================


def compute_delta_min_fast(V: np.ndarray) -> float:
    """
    Compute minimum pairwise separation of unit value vectors (vectorized).

    delta_min(V) = min_{i!=j} ||v_i - v_j||
    For unit vectors: ||v_i - v_j||^2 = 2 - 2<v_i, v_j>
    """
    G = V @ V.T
    np.fill_diagonal(G, -np.inf)
    max_inner = G.max()
    return np.sqrt(max(0.0, 2 - 2 * max_inner))


def compute_mu_V(V: np.ndarray) -> float:
    """
    Compute the coherence of value embeddings.

    mu(V) = max_{a!=b} |<v_a, v_b>|
    """
    G = V @ V.T
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))


def unified_bound(
    K_hat_diag_min: float,
    mu_V: float,
    E_col_max: float,
    n: int,
    G: int = 1,
    d_int: float = 1.0,
    delta: float = 0.5,
) -> float:
    """
    Compute the coherence-based margin bound.

    When U = V the signal term is:
      s_{i,j} = K_hat_{ii} * <v_{f(i)} - v_j, v_{f(i)}>
              = K_hat_{ii} * (1 - <v_{f(i)}, v_j>)
              >= K_hat_{ii} * (1 - mu(V))

    gamma_min >= min_i(K_hat_{ii}) * (1 - mu(V))
                 - 2*sqrt(2) * sqrt(E_col_max * L / (G * d_int))

    where L = log(n^2 / delta).
    """
    L = np.log(n**2 / delta)
    signal_term = K_hat_diag_min * (1.0 - mu_V)
    noise_term = 2 * np.sqrt(2) * np.sqrt(E_col_max * L / (G * d_int))
    return signal_term - noise_term


def decodability_bound(
    K_hat_diag_min: float,
    rho_V: float,
    delta_min: float,
    E_col_max: float,
    n: int,
    G: int = 1,
    d_int: float = 1.0,
    delta: float = 0.5,
) -> float:
    """
    Compute the decodability-based margin bound (using u* codes).

    gamma_min >= rho(V) * delta_min(V) * min_i(K_hat_{ii})
                 - 2*sqrt(2) * sqrt(E_col_max * L / (G * d_int))

    where rho(V) = min_i max_{||u||=1} min_{j!=i} <v_i-v_j, u>/||v_i-v_j||.
    """
    L = np.log(n**2 / delta)
    signal_term = rho_V * delta_min * K_hat_diag_min
    noise_term = 2 * np.sqrt(2) * np.sqrt(E_col_max * L / (G * d_int))
    return signal_term - noise_term


def unified_bound_fitted(
    K_hat_diag_min: float,
    mu_V: float,
    E_col_max: float,
    n: int,
    G: int,
    d_int: float,
    delta: float,
    C0: float = 1.0,
    C1: float = 2.828,
) -> float:
    """
    Compute the coherence-based margin bound with fitting constants C0, C1.

    gamma ~ C0 * min_i(K_hat_{ii}) * (1 - mu(V))
            - C1 * sqrt(E_col_max * L / (G * d_int))

    The exact theorem has C0=1, C1=2*sqrt(2)~2.828.
    """
    L = np.log(n**2 / delta)
    signal_term = C0 * K_hat_diag_min * (1.0 - mu_V)
    noise_term = C1 * np.sqrt(E_col_max * L / (G * d_int))
    return signal_term - noise_term


def unified_bound_from_experiment(
    V: np.ndarray,
    K_hat_diag: np.ndarray,
    E_col_max: float,
    n: int,
    G: int = 1,
    d_int: float = 1.0,
    delta: float = 0.5,
    V_all: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the coherence-based bound from experiment data.

    Args:
        V: (n, d) per-key value embeddings v_{f(i)} (unit normalized)
        K_hat_diag: (n,) diagonal of feature Gram matrix
        E_col_max: Maximum crowding energy
        n: number of stored keys
        G: number of micro-heads
        d_int: effective interference dimension
        delta: confidence parameter
        V_all: (P, d) all distinct value embeddings. If None, uses V
               (correct only when n = P with no duplicate values).
    """
    K_hat_diag_min = float(np.min(K_hat_diag))

    # Coherence and delta_min over distinct value classes
    V_ref = V_all if V_all is not None else V
    mu_V = compute_mu_V(V_ref)
    delta_min_val = compute_delta_min_fast(V_ref)

    L = np.log(n**2 / delta)
    bound = unified_bound(K_hat_diag_min, mu_V, E_col_max, n, G, d_int, delta)

    signal = K_hat_diag_min * (1.0 - mu_V)
    noise = 2 * np.sqrt(2) * np.sqrt(E_col_max * L / (G * d_int))

    components = {
        "K_hat_diag_min": K_hat_diag_min,
        "mu_V": mu_V,
        "delta_min": delta_min_val,
        "E_col_max": E_col_max,
        "d_int": d_int,
        "G": G,
        "L": L,
        "signal_term": signal,
        "noise_term": noise,
        "bound": bound,
    }
    return bound, components


# Global variables for unified bound fitting (used by curve_fit)
_unified_fit_data = {}


def _unified_fit_func(x_values, C0, C1):
    """Fitting function for unified bound (used by both F-sweep and M-sweep)."""
    global _unified_fit_data
    d = _unified_fit_data
    result = np.zeros(len(x_values))
    for i in range(len(x_values)):
        result[i] = unified_bound_fitted(
            d["K_hat_diag_min_values"][i], d["mu_V_values"][i],
            d["E_col_max_values"][i],
            d["n_values"][i], d["G_values"][i], d["d_int_values"][i],
            d["delta"], C0, C1,
        )
    return result


def _unified_fit_func_C0_fixed(x_values, C1):
    """Fitting function with C0=1 fixed, only fitting C1."""
    global _unified_fit_data
    d = _unified_fit_data
    result = np.zeros(len(x_values))
    for i in range(len(x_values)):
        result[i] = unified_bound_fitted(
            d["K_hat_diag_min_values"][i], d["mu_V_values"][i],
            d["E_col_max_values"][i],
            d["n_values"][i], d["G_values"][i], d["d_int_values"][i],
            d["delta"], 1.0, C1,
        )
    return result


def _fit_unified_bound(
    x_values: np.ndarray,
    gamma_values: np.ndarray,
    K_hat_diag_min_values: np.ndarray,
    mu_V_values: np.ndarray,
    E_col_max_values: np.ndarray,
    n_values: np.ndarray,
    G_values: np.ndarray,
    d_int_values: np.ndarray,
    delta: float = 0.5,
    initial_guess: Tuple[float, float] = (1.0, 2.828),
) -> Tuple[Dict[str, float], np.ndarray]:
    """Core fitting routine for the unified bound."""
    global _unified_fit_data
    _unified_fit_data = {
        "K_hat_diag_min_values": K_hat_diag_min_values,
        "mu_V_values": mu_V_values,
        "E_col_max_values": E_col_max_values,
        "n_values": n_values,
        "G_values": G_values,
        "d_int_values": d_int_values,
        "delta": delta,
    }

    bounds = ([0, 0], [5, 20])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _unified_fit_func, x_values.astype(float), gamma_values,
                p0=initial_guess, bounds=bounds, maxfev=5000,
            )
        C0, C1 = popt
    except RuntimeError:
        C0, C1 = initial_guess

    predicted = np.array([
        unified_bound_fitted(
            K_hat_diag_min_values[i], mu_V_values[i],
            E_col_max_values[i], n_values[i], G_values[i], d_int_values[i],
            delta, C0, C1,
        )
        for i in range(len(x_values))
    ])

    formula = "gamma = C0*K_hat_diag_min*(1-mu_V) - C1*sqrt(E_col*L/(G*d_int))"
    return {"C0": C0, "C1": C1, "formula": formula}, predicted


def fit_unified_bound_F_sweep(
    F_values: np.ndarray,
    gamma_values: np.ndarray,
    K_hat_diag_min_values: np.ndarray,
    mu_V_values: np.ndarray,
    E_col_max_values: np.ndarray,
    n_values: np.ndarray,
    G_values: np.ndarray,
    d_int_values: np.ndarray,
    delta: float = 0.5,
    initial_guess: Tuple[float, float] = (1.0, 2.828),
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit the unified bound constants to empirical F-sweep data."""
    return _fit_unified_bound(
        F_values, gamma_values, K_hat_diag_min_values, mu_V_values,
        E_col_max_values, n_values, G_values,
        d_int_values, delta, initial_guess,
    )


def fit_unified_bound_M_sweep(
    M_values: np.ndarray,
    gamma_values: np.ndarray,
    K_hat_diag_min_values: np.ndarray,
    mu_V_values: np.ndarray,
    E_col_max_values: np.ndarray,
    n_values: np.ndarray,
    G_values: np.ndarray,
    d_int_values: np.ndarray,
    delta: float = 0.5,
    initial_guess: Tuple[float, float] = (1.0, 2.828),
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit the unified bound constants to empirical M-sweep data."""
    return _fit_unified_bound(
        M_values, gamma_values, K_hat_diag_min_values, mu_V_values,
        E_col_max_values, n_values, G_values,
        d_int_values, delta, initial_guess,
    )


def _fit_unified_bound_C0_fixed(
    x_values: np.ndarray,
    gamma_values: np.ndarray,
    K_hat_diag_min_values: np.ndarray,
    mu_V_values: np.ndarray,
    E_col_max_values: np.ndarray,
    n_values: np.ndarray,
    G_values: np.ndarray,
    d_int_values: np.ndarray,
    delta: float = 0.5,
    initial_guess: float = 2.828,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit only C1 with C0=1 fixed."""
    global _unified_fit_data
    _unified_fit_data = {
        "K_hat_diag_min_values": K_hat_diag_min_values,
        "mu_V_values": mu_V_values,
        "E_col_max_values": E_col_max_values,
        "n_values": n_values,
        "G_values": G_values,
        "d_int_values": d_int_values,
        "delta": delta,
    }

    bounds = ([0], [20])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _unified_fit_func_C0_fixed, x_values.astype(float), gamma_values,
                p0=[initial_guess], bounds=bounds, maxfev=5000,
            )
        C1 = popt[0]
    except RuntimeError:
        C1 = initial_guess

    predicted = np.array([
        unified_bound_fitted(
            K_hat_diag_min_values[i], mu_V_values[i],
            E_col_max_values[i], n_values[i], G_values[i], d_int_values[i],
            delta, 1.0, C1,
        )
        for i in range(len(x_values))
    ])

    formula = "gamma = K_hat_diag_min*(1-mu_V) - C1*sqrt(E_col*L/(G*d_int))"
    return {"C0": 1.0, "C1": C1, "formula": formula}, predicted


def fit_unified_bound_C0_fixed_F_sweep(
    F_values: np.ndarray,
    gamma_values: np.ndarray,
    K_hat_diag_min_values: np.ndarray,
    mu_V_values: np.ndarray,
    E_col_max_values: np.ndarray,
    n_values: np.ndarray,
    G_values: np.ndarray,
    d_int_values: np.ndarray,
    delta: float = 0.5,
    initial_guess: float = 2.828,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit C1 only (C0=1 fixed) to empirical F-sweep data."""
    return _fit_unified_bound_C0_fixed(
        F_values, gamma_values, K_hat_diag_min_values, mu_V_values,
        E_col_max_values, n_values, G_values,
        d_int_values, delta, initial_guess,
    )


def fit_unified_bound_C0_fixed_M_sweep(
    M_values: np.ndarray,
    gamma_values: np.ndarray,
    K_hat_diag_min_values: np.ndarray,
    mu_V_values: np.ndarray,
    E_col_max_values: np.ndarray,
    n_values: np.ndarray,
    G_values: np.ndarray,
    d_int_values: np.ndarray,
    delta: float = 0.5,
    initial_guess: float = 2.828,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit C1 only (C0=1 fixed) to empirical M-sweep data."""
    return _fit_unified_bound_C0_fixed(
        M_values, gamma_values, K_hat_diag_min_values, mu_V_values,
        E_col_max_values, n_values, G_values,
        d_int_values, delta, initial_guess,
    )


def unified_bound_exact_sweep(
    K_hat_diag_min_values: np.ndarray,
    mu_V_values: np.ndarray,
    E_col_max_values: np.ndarray,
    n_values: np.ndarray,
    G_values: np.ndarray,
    d_int_values: np.ndarray,
    delta: float = 0.5,
) -> np.ndarray:
    """Evaluate the exact coherence-based bound at each sweep point."""
    return np.array([
        unified_bound(
            K_hat_diag_min_values[i], mu_V_values[i],
            E_col_max_values[i], int(n_values[i]), int(G_values[i]),
            d_int_values[i], delta,
        )
        for i in range(len(K_hat_diag_min_values))
    ])


# =============================================================================
# Noisy-Query Lipschitz Constant and Margin Bound
# =============================================================================


def compute_lipschitz_bound_bilinear(
    feature_map,
    K: torch.Tensor,
    Z: Optional[torch.Tensor] = None,
) -> float:
    """
    Upper-bound the Lipschitz constant L_bil of the bilinear kernel hat_k(x, z) = <g(x), g(z)>.

    For the bilinear feature map g(z) = (1/sqrt(m)) * (A0 z) odot (A1 z),
    the kernel Lipschitz constant satisfies (Appendix of noisy_query_margins_bilinear_mlp.tex):

        L_bil <= B_g * sup_{z in Z} ||Jg(z)||_op

    where
        B_g = max_t ||g(k_t)||_2                      (max feature norm over stored keys)
        Jg(z) = (1/sqrt(m)) * (diag(A1 z) A0 + diag(A0 z) A1)  (m x d Jacobian)

    and the cheap analytical upper bound on the operator norm is:
        ||Jg(z)||_op <= (1/sqrt(m)) * (||A1 z||_inf * ||A0||_op + ||A0 z||_inf * ||A1||_op)

    The supremum is approximated by evaluating this upper bound over K and Z.

    Args:
        feature_map: ProductRFFeatureMap with p=2, gate='raw' (bilinear map)
        K: (n, d) stored key embeddings
        Z: (nz, d) optional additional query points (e.g. noisy queries)

    Returns:
        L_bil_ub: upper bound on the bilinear kernel Lipschitz constant
    """
    A0 = feature_map.A0  # (m, d)
    A1 = feature_map.A1  # (m, d)
    m = float(A0.shape[0])

    with torch.no_grad():
        # B_g = max_t ||g(k_t)||_2
        g_K = feature_map(K)  # (n, m)
        B_g = g_K.norm(dim=1).max().item()

        # Operator norms of A0 and A1 (largest singular value)
        A0_op = torch.linalg.svdvals(A0.float())[0].item()
        A1_op = torch.linalg.svdvals(A1.float())[0].item()

        # Evaluation points: stored keys + optional extra queries
        Z_eval = K if Z is None else torch.cat([K, Z], dim=0)

        # (A0 z) and (A1 z) for each evaluation point
        A0_z = Z_eval @ A0.T  # (nz, m): rows are (A0 z_i)
        A1_z = Z_eval @ A1.T  # (nz, m): rows are (A1 z_i)

        # Cheap upper bound: ||Jg(z)||_op <= (1/sqrt(m)) * (||A1 z||_inf ||A0||_op + ||A0 z||_inf ||A1||_op)
        A0_z_inf = A0_z.abs().max(dim=1).values  # (nz,)
        A1_z_inf = A1_z.abs().max(dim=1).values  # (nz,)

        Jg_op_bound = (A1_z_inf * A0_op + A0_z_inf * A1_op) / (m ** 0.5)
        Jg_op_max = Jg_op_bound.max().item()

    return B_g * Jg_op_max


def noisy_query_margin_bound(
    n: int,
    d: int,
    m: int,
    epsilon: float,
    L_bil: float,
    delta: float = 0.5,
) -> dict:
    """
    Compute the noisy-query margin bound from Thm 2 of noisy_query_margins_bilinear_mlp.tex.

    For the iso-iso bilinear-K2 model with noisy queries ||z_i - k_i||_2 <= epsilon:

        gamma_min(z) >= [clean_bound] - L_bil * epsilon * (1 + 2*sqrt(2)*sqrt(n*L/d))

    where the clean_bound is exactly the iso-iso bilinear-K2 bound (Thm cor:bilinear-scaling):

        clean_bound = 1
                      - 2*sqrt(6)*sqrt(n*L/d^3)
                      - 8*sqrt(n*L/(m*d))
                      - sqrt(2)*sqrt(L/d)          [mu_v value coherence]
                      - sqrt(18)*sqrt(log(4n/delta)/m)  [diagonal term]
                      - 8*log(4n/delta)/m           [column tail]
                      - 4*L/d^2                     [rho_4 deviation]

    and L = log(4*n^2 / delta).

    Args:
        n: number of stored facts
        d: embedding dimension
        m: MLP hidden width (number of bilinear features)
        epsilon: noise level ||z_i - k_i||_2 <= epsilon
        L_bil: Lipschitz constant of the bilinear kernel (from compute_lipschitz_bound_bilinear)
        delta: failure probability

    Returns:
        dict with keys: clean_bound, noise_penalty, noisy_bound, L_log
    """
    L = np.log(4 * n ** 2 / delta)
    L_n = np.log(4 * n / delta)

    # Clean bound terms (iso-iso bilinear-K2 theorem)
    clean_main = (
        1.0
        - 2 * np.sqrt(6) * np.sqrt(n * L / d ** 3)
        - 8 * np.sqrt(n * L / (m * d))
    )
    lot = (
        np.sqrt(2) * np.sqrt(L / d)
        + np.sqrt(18) * np.sqrt(L_n / m)
        + 8 * L_n / m
        + 4 * L / d ** 2
    )
    clean_bound = clean_main - lot

    # Noise penalty: L_bil * epsilon * (1 + 2*sqrt(2)*sqrt(n*L/d))
    noise_penalty = L_bil * epsilon * (1.0 + 2 * np.sqrt(2) * np.sqrt(n * L / d))

    return {
        "clean_bound": clean_bound,
        "noise_penalty": noise_penalty,
        "noisy_bound": clean_bound - noise_penalty,
        "L_log": L,
    }


def compute_noisy_margin(
    mlp,
    factset,
    Z_noisy: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    gamma_min_percentile: Optional[float] = None,
) -> dict:
    """
    Compute the retrieval margin at noisy queries Z_noisy.

    The MLP was built from stored keys K; we now query it at Z_noisy instead.
    The mixed Gram K_mixed[t, i] = hat_k(k_t, z_i) = g(k_t)^T g(z_i),
    and the output at z_i is y(z_i) = K_mixed^T @ code_vectors.

    Args:
        mlp: HebbianMLP with feature map and weight matrix
        factset: Factset with input_embeddings (stored keys), output_embeddings, mapping
        Z_noisy: (n, d) noisy query embeddings
        device: computation device
        dtype: computation dtype

    Returns:
        dict with gamma_min (float) and gamma_per_key (Tensor, n,)
    """
    with torch.no_grad():
        K_stored = factset.input_embeddings.to(device=device, dtype=dtype)  # (n, d)
        V_all = factset.output_embeddings.to(device=device, dtype=dtype)    # (P, d_v)
        n = K_stored.shape[0]
        value_indices = [factset.mapping.get_output(i) for i in range(n)]
        code_vectors = V_all[value_indices]  # (n, d_v): c_{f(t)} for each stored key

        Z = Z_noisy.to(device=device, dtype=dtype)  # (n, d)

        # Features at stored keys and noisy queries
        Phi_K = mlp.feature_map(K_stored)  # (n, m)
        Phi_Z = mlp.feature_map(Z)          # (n, m)

        # Mixed Gram: K_mixed[t, i] = <g(k_t), g(z_i)>
        K_mixed = Phi_K @ Phi_Z.T  # (n, n)

        # Output at each noisy query: y(z_i) = sum_t c_{f(t)} * K_mixed[t, i]
        Y = K_mixed.T @ code_vectors  # (n, d_v)

        # Scores and margins
        scores = Y @ V_all.T  # (n, P)
        correct_indices = torch.tensor(value_indices, device=device)
        batch_idx = torch.arange(n, device=device)

        correct_scores = scores[batch_idx, correct_indices]
        scores_masked = scores.clone()
        scores_masked[batch_idx, correct_indices] = float('-inf')
        max_incorrect = scores_masked.max(dim=1).values

        gamma_per_key = correct_scores - max_incorrect
        gamma_min = _reduce_gamma_min(gamma_per_key, gamma_min_percentile)

    return {"gamma_min": gamma_min, "gamma_per_key": gamma_per_key.cpu()}


# =============================================================================
# Schematic Bound for Uniform Spherical Keys/Values (Theorem 3.8)
# =============================================================================


def schematic_bound(
    n: int,
    d: int,
    M: int,
    C0: float = 1.0,
    C1: float = 1.0,
    C2: float = 1.0,
    delta: float = 0.5,
) -> float:
    """
    Compute the schematic margin bound for uniform spherical keys/values.

    gamma_min ~ C0 - sqrt(n/d) * (C1 * log(n^2/delta) / d + C2 * sqrt(log(n^2/delta) / M))
    """
    log_term = np.log(n**2 / delta)
    key_term = C1 * log_term / d
    feature_term = C2 * np.sqrt(log_term / M)
    cross_talk = np.sqrt(n / d) * (key_term + feature_term)
    return C0 - cross_talk


def schematic_bound_F_sweep(
    F_values: np.ndarray, d: int, M: int,
    C0: float, C1: float, C2: float, delta: float = 0.5,
) -> np.ndarray:
    """Vectorized schematic bound for F-sweep."""
    return np.array([schematic_bound(F, d, M, C0, C1, C2, delta) for F in F_values])


def schematic_bound_M_sweep(
    M_values: np.ndarray, d: int, F: int,
    C0: float, C1: float, C2: float, delta: float = 0.5,
) -> np.ndarray:
    """Vectorized schematic bound for M-sweep."""
    return np.array([schematic_bound(F, d, M, C0, C1, C2, delta) for M in M_values])


# Global variables for schematic bound fitting
_fit_d = None
_fit_M = None
_fit_F = None
_fit_delta = None


def _fit_func_F(F, C0, C1, C2):
    """Fitting function for F-sweep."""
    global _fit_d, _fit_M, _fit_delta
    return schematic_bound_F_sweep(F, _fit_d, _fit_M, C0, C1, C2, _fit_delta)


def _fit_func_M(M, C0, C1, C2):
    """Fitting function for M-sweep."""
    global _fit_d, _fit_F, _fit_delta
    return schematic_bound_M_sweep(M, _fit_d, _fit_F, C0, C1, C2, _fit_delta)


def fit_schematic_bound_F_sweep(
    F_values: np.ndarray,
    gamma_values: np.ndarray,
    d: int,
    M: int,
    delta: float = 0.5,
    initial_guess: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit the schematic bound constants to empirical F-sweep data."""
    global _fit_d, _fit_M, _fit_delta
    _fit_d, _fit_M, _fit_delta = d, M, delta

    bounds = ([0, 0, 0], [2, 10, 10])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _fit_func_F, F_values.astype(float), gamma_values,
                p0=initial_guess, bounds=bounds, maxfev=5000,
            )
        C0, C1, C2 = popt
    except RuntimeError:
        C0, C1, C2 = initial_guess

    predicted = schematic_bound_F_sweep(F_values, d, M, C0, C1, C2, delta)
    return {"C0": C0, "C1": C1, "C2": C2, "delta": delta}, predicted


def fit_schematic_bound_M_sweep(
    M_values: np.ndarray,
    gamma_values: np.ndarray,
    d: int,
    F: int,
    delta: float = 0.5,
    initial_guess: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit the schematic bound constants to empirical M-sweep data."""
    global _fit_d, _fit_F, _fit_delta
    _fit_d, _fit_F, _fit_delta = d, F, delta

    bounds = ([0, 0, 0], [2, 10, 10])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _fit_func_M, M_values.astype(float), gamma_values,
                p0=initial_guess, bounds=bounds, maxfev=5000,
            )
        C0, C1, C2 = popt
    except RuntimeError:
        C0, C1, C2 = initial_guess

    predicted = schematic_bound_M_sweep(M_values, d, F, C0, C1, C2, delta)
    return {"C0": C0, "C1": C1, "C2": C2, "delta": delta}, predicted
