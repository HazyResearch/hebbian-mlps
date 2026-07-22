"""Fit the paper's theoretical margin forms to sweep results."""

from __future__ import annotations

import json
import pickle

import numpy as np
from scipy.optimize import curve_fit


CASE_TITLES = {
    "rkrv": "Isotropic keys, isotropic values",
    "akrv": "Arbitrary keys, isotropic values",
    "rkav": "Isotropic keys, arbitrary values",
    "akav": "Arbitrary keys, arbitrary values",
}

CASE_EQUATIONS = {
    "rkrv": r"Fitted bound: $\gamma_{\min}=C_s-C_x\sqrt{F\hat L/(md)}$",
    "akrv": (
        r"Fitted bound: $\gamma_{\min}=C_sK_{\min}^{\rm diag}"
        r"-C_oK_{\max}^{\rm off}-C_x\sqrt{E_K\hat L/d}$"
    ),
    "rkav": (
        r"Fitted bound: $\gamma_{\min}=C_sV_{\min}"
        r"-(C_BB_Y+C_v\sqrt{E_vL_v})\sqrt{\hat L/m}$"
    ),
    "akav": (
        r"Fitted bound: $\gamma_{\min}=C_1K_{\min}^{\rm diag}V_{\min}"
        r"-C_x\sqrt{E_KE_v}\kappa$"
    ),
}


def load_results(path: str) -> dict:
    if path.endswith(".pkl"):
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        return data
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def detect_case(results: dict) -> str:
    """Infer the key/value regime from sweep metadata."""

    if results.get("sweep_type", "F") != "beta":
        return "rkrv"
    spike_target = results.get("spike_target", "both")
    return {"keys": "akrv", "values": "rkav"}.get(spike_target, "akav")


def _array(results: dict, key: str, length: int, default: float) -> np.ndarray:
    return np.asarray(results.get(key, [default] * length), dtype=float)


def extract_quantities(results: dict) -> dict:
    """Extract the empirical and theorem quantities needed by the fit."""

    sweep_type = results.get("sweep_type", "F")
    dimension = float(results["d"])
    facts = np.asarray(results["n_values"], dtype=float)
    gamma = np.asarray(results["gamma_min_best"], dtype=float)
    length = len(gamma)

    if sweep_type == "F":
        x = np.asarray(results["F_values"], dtype=float)
        x_label = "Number of facts $F$"
        width = float(results.get("M", results.get("M_values", [1])[0]))
        widths = np.full(length, width)
    elif sweep_type == "M":
        x = np.asarray(results["M_values"], dtype=float)
        x_label = "Feature dimension $M$"
        widths = x.copy()
    elif sweep_type == "beta":
        beta_values = np.asarray(results["beta_values"], dtype=float)
        key_energy = _array(results, "E_col_max_best", len(beta_values), 0.0)
        value_energy = _array(results, "E_Y_best", len(beta_values), 0.0)
        kappa = _array(results, "kappa_max_best", len(beta_values), 0.0)
        spike_target = results.get("spike_target", "both")
        if spike_target == "keys":
            x, x_label = key_energy, "$E_K$"
        elif spike_target == "values":
            value_sparsity = _array(results, "L_v_best", len(beta_values), 1.0)
            x, x_label = value_energy * value_sparsity, "$E_v L_v$"
        else:
            x = np.sqrt(np.maximum(key_energy, 0.0))
            x *= np.sqrt(np.maximum(value_energy, 0.0)) * kappa
            x_label = r"$\sqrt{E_K}\sqrt{E_v}\,\kappa$"
        widths = np.full(length, float(results.get("M", 256)))
    else:
        x = np.arange(length, dtype=float)
        x_label = "Index"
        widths = np.full(length, float(results.get("M", 256)))

    return {
        "sweep_type": sweep_type,
        "d": dimension,
        "x": x,
        "xlabel": x_label,
        "n": facts,
        "M_arr": widths,
        "gamma": gamma,
        "gamma_std": _array(results, "gamma_min_std", length, 0.0),
        "K_min_diag": _array(results, "K_hat_diag_min_best", length, 1.0),
        "E_K": _array(results, "E_col_max_best", length, 0.0),
        "E_v": _array(results, "E_Y_best", length, 0.0),
        "L_v": _array(results, "L_v_best", length, 0.0),
        "kappa": _array(results, "kappa_max_best", length, 0.0),
        "A_min": _array(results, "A_min_best", length, 1.0),
        "B_Y": _array(results, "B_Y_best", length, 0.0),
        "K_max_off": _array(results, "K_max_off_best", length, 0.0),
    }


def r2_mse(
    empirical: np.ndarray, predicted: np.ndarray
) -> tuple[float, float]:
    valid = np.isfinite(empirical) & np.isfinite(predicted)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    empirical = empirical[valid]
    predicted = predicted[valid]
    residual_sum = np.sum((empirical - predicted) ** 2)
    total_sum = np.sum((empirical - empirical.mean()) ** 2)
    r2 = 1.0 - residual_sum / total_sum if total_sum > 1e-30 else float("nan")
    return float(r2), float(np.mean((empirical - predicted) ** 2))


def _fit(
    model_fn,
    gamma: np.ndarray,
    initial: list[float],
    bounds: tuple[list[float], list[float]],
) -> list[float]:
    """Fit constants with scaled restarts to avoid a degenerate zero fit."""

    valid = np.isfinite(gamma)
    if valid.sum() < len(initial) + 1:
        return initial
    target = gamma[valid]
    x_dummy = np.zeros(valid.sum())
    lower, upper = bounds
    best_error = float("inf")
    best = list(initial)
    for scale in (1.0, 1e4, 1e8):
        start = [
            min(
                max(value * scale, lower[index]),
                upper[index] if np.isfinite(upper[index]) else value * scale,
            )
            for index, value in enumerate(initial)
        ]
        try:
            fitted, _ = curve_fit(
                model_fn,
                x_dummy,
                target,
                p0=start,
                bounds=bounds,
                maxfev=20000,
            )
            predicted = model_fn(x_dummy, *fitted)
            error = float(np.sum((target - predicted) ** 2))
            if np.all(np.isfinite(predicted)) and error < best_error:
                best_error = error
                best = [float(value) for value in fitted]
        except Exception:
            continue
    return best


def compute_fitted_bound(case: str, quantities: dict) -> tuple[np.ndarray, str]:
    """Fit the constants in the theorem form for one key/value regime."""

    dimension = quantities["d"]
    facts = quantities["n"]
    widths = quantities["M_arr"]
    gamma = quantities["gamma"]

    def log_factor(constant: float) -> np.ndarray:
        return np.log(np.maximum(constant * facts**2 / 0.5, 1e-30))

    if case == "rkrv":
        def model(_, signal, log_constant, crosstalk):
            noise = np.sqrt(
                np.maximum(
                    facts * log_factor(max(log_constant, 1e-6))
                    / (widths * dimension),
                    0.0,
                )
            )
            return signal - crosstalk * noise

        constants = _fit(
            model,
            gamma,
            [1.0, 4.0, 1.0],
            ([0, 1e-6, 0], [np.inf, np.inf, np.inf]),
        )
        label = r"$\hat C_{\rm sig}-\hat C_{\rm xtalk}\sqrt{F\hat L/(md)}$"
    elif case == "akrv":
        def model(_, signal, off_diagonal, log_constant, crosstalk):
            noise = np.sqrt(
                np.maximum(
                    quantities["E_K"] * log_factor(max(log_constant, 1e-6))
                    / dimension,
                    0.0,
                )
            )
            return (
                signal * quantities["K_min_diag"]
                - off_diagonal * quantities["K_max_off"]
                - crosstalk * noise
            )

        constants = _fit(
            model,
            gamma,
            [1.0, 1.0, 4.0, 1.0],
            ([0, 0, 1e-6, 0], [np.inf, np.inf, np.inf, np.inf]),
        )
        label = (
            r"$\hat C_{\rm sig}K_{\min}^{\rm diag}"
            r"-\hat C_{\rm off}K_{\max}^{\rm off}"
            r"-\hat C_{\rm xtalk}\sqrt{E_K\hat L/d}$"
        )
    elif case == "rkav":
        value_noise = np.sqrt(
            np.maximum(quantities["E_v"] * quantities["L_v"], 0.0)
        )

        def model(_, signal, log_constant, value_bound, value_energy):
            scale = np.sqrt(
                np.maximum(log_factor(max(log_constant, 1e-6)) / widths, 0.0)
            )
            noise = (
                value_bound * quantities["B_Y"] + value_energy * value_noise
            ) * scale
            return signal * quantities["A_min"] - noise

        constants = _fit(
            model,
            gamma,
            [1.0, 4.0, 1.0, 1.0],
            ([0, 1e-6, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
        )
        label = (
            r"$\hat C_{\rm sig}V_{\min}"
            r"-(\hat C_{B_Y}B_Y+\hat C_{E_v}\sqrt{E_vL_v})\sqrt{\hat L/m}$"
        )
    elif case == "akav":
        signal_term = quantities["K_min_diag"] * quantities["A_min"]
        noise = (
            np.sqrt(np.maximum(quantities["E_K"], 0.0))
            * np.sqrt(np.maximum(quantities["E_v"], 0.0))
            * quantities["kappa"]
        )

        def model(_, signal, crosstalk):
            return signal * signal_term - crosstalk * noise

        constants = _fit(
            model,
            gamma,
            [1.0, 1.0],
            ([0, 0], [np.inf, np.inf]),
        )
        label = (
            r"$\hat C_{\rm sig}K_{\min}^{\rm diag}V_{\min}"
            r"-\hat C_{\rm xtalk}\sqrt{E_KE_v}\,\kappa$"
        )
    else:
        raise ValueError(f"Unknown margin case: {case}")

    return model(None, *constants), label


__all__ = [
    "CASE_EQUATIONS",
    "CASE_TITLES",
    "compute_fitted_bound",
    "detect_case",
    "extract_quantities",
    "load_results",
    "r2_mse",
]
