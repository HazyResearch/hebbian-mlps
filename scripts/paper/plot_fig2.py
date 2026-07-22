#!/usr/bin/env python3
"""Generate the three Figure 2 panels and the unified 1x3 figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hebbian.expts.margins.theory import (  # noqa: E402
    CASE_EQUATIONS,
    CASE_TITLES,
    compute_fitted_bound,
    extract_quantities,
    load_results,
    r2_mse,
)

COLOR_MLP = "#2E86AB"
COLOR_TFM = "#E63946"
COLOR_GAMMA = "#F18F01"
CAPACITY_METHOD_ORDER = [
    "gd",
    "hebbian",
    "hebbian_whitened",
    "cf_coord_whitened",
    "ntk",
]
CAPACITY_METHOD_DISPLAY = {
    "gd": "GD",
    "hebbian": "Ours (no whitening)",
    "hebbian_whitened": "Ours (whitened)",
    "cf_coord_whitened": "Ours (data-dependent)",
    "ntk": "NTK",
}
CAPACITY_METHOD_COLOR = {
    "gd": "#E63946",
    "hebbian": "#2E86AB",
    "hebbian_whitened": "#16A085",
    "cf_coord_whitened": "#6D28D9",
    "ntk": "#F18F01",
}
CAPACITY_METHOD_LINESTYLE = {
    "hebbian": ":",
    "hebbian_whitened": ":",
    "ntk": ":",
}
CAPACITY_D_MARKER = {64: "o", 90: "s", 128: "^"}
CAPACITY_D_LINESTYLE = {64: "-", 90: "--", 128: "-."}
CAPACITY_FIT_METHODS = {"gd", "cf_coord_whitened"}
PANEL_TITLES = [
    "Hidden-dim usability",
    "Margin validation",
    "MLP capacity",
]


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_hidden_dim_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                {
                    "hidden_dim": float(raw["hidden_dim"]),
                    "param_count": float(raw["param_count"]),
                    "mlp_accuracy": float(raw["mlp_accuracy"]),
                    "transformer_accuracy": float(raw["transformer_accuracy"]),
                    "gamma_min": float(raw["gamma_min"]),
                }
            )
    rows.sort(key=lambda r: r["param_count"])
    if not rows:
        raise ValueError(f"No hidden-dim rows found in {path}")
    return rows


def _setup_hidden_dim_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.labelsize": 20,
            "axes.labelweight": "bold",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
            "lines.linewidth": 2.7,
            "lines.markeredgewidth": 1.1,
            "lines.markeredgecolor": "black",
            "savefig.dpi": 300,
        }
    )


def _hidden_dim_series(
    rows: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float | None, float | None]:
    x = np.array([r["param_count"] for r in rows], dtype=float)
    mlp = np.array([r["mlp_accuracy"] for r in rows], dtype=float)
    tfm = np.array([r["transformer_accuracy"] for r in rows], dtype=float)
    gamma = np.array([r["gamma_min"] for r in rows], dtype=float)
    mlp_thresh = next((xi for xi, yi in zip(x, mlp) if yi >= 1.0), None)
    tfm_thresh = next((xi for xi, yi in zip(x, tfm) if yi >= 1.0), None)
    return x, mlp, tfm, gamma, mlp_thresh, tfm_thresh


def _draw_hidden_dim_panel(
    ax1: plt.Axes,
    rows: list[dict[str, float]],
    *,
    combined: bool = False,
) -> tuple[plt.Axes, dict[str, float | int | None]]:
    x, mlp, tfm, gamma, mlp_thresh, tfm_thresh = _hidden_dim_series(rows)
    ax2 = ax1.twinx()
    marker_size = math.sqrt(82 if combined else 112)
    line_width = 2.25 if combined else 2.7
    annotation_fontsize = 8.0 if combined else 12.5

    l1, = ax1.plot(
        x,
        mlp,
        "o-",
        color=COLOR_MLP,
        markersize=marker_size,
        linewidth=line_width,
        label="MLP accuracy",
        alpha=0.92,
    )
    l2, = ax1.plot(
        x,
        tfm,
        "s-",
        color=COLOR_TFM,
        markersize=marker_size,
        linewidth=line_width,
        label="Transformer accuracy",
        alpha=0.92,
    )
    l3, = ax2.plot(
        x,
        gamma,
        "^--",
        color=COLOR_GAMMA,
        markersize=marker_size,
        linewidth=line_width,
        label="MLP margin",
        alpha=0.88,
    )

    if mlp_thresh is not None:
        ax1.axvline(
            mlp_thresh,
            color=COLOR_MLP,
            linestyle=":",
            linewidth=1.7 if combined else 2.0,
            alpha=0.78,
        )
        ax1.annotate(
            f"{int(mlp_thresh):,} params",
            xy=(mlp_thresh, 0.12),
            xycoords=ax1.get_xaxis_transform(),
            xytext=(4 if combined else 6, 0),
            textcoords="offset points",
            color=COLOR_MLP,
            fontsize=annotation_fontsize,
            fontweight="bold",
            ha="left",
            va="bottom",
            rotation=90,
            rotation_mode="anchor",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        )
    if tfm_thresh is not None:
        ax1.axvline(
            tfm_thresh,
            color=COLOR_TFM,
            linestyle=":",
            linewidth=1.7 if combined else 2.0,
            alpha=0.78,
        )
        ax1.annotate(
            f"{int(tfm_thresh):,} params",
            xy=(tfm_thresh, 0.34),
            xycoords=ax1.get_xaxis_transform(),
            xytext=(4 if combined else 6, 0),
            textcoords="offset points",
            color=COLOR_TFM,
            fontsize=annotation_fontsize,
            fontweight="bold",
            ha="left",
            va="bottom",
            rotation=90,
            rotation_mode="anchor",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("# MLP parameters")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.28, linestyle="--", which="major")
    ax2.set_ylabel("" if combined else r"$\gamma_{\min}$", color=COLOR_GAMMA, labelpad=1.5)
    ax2.tick_params(axis="y", labelcolor=COLOR_GAMMA, pad=1.5)

    legend_handles = [l1, l3, l2]
    ax1.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower right",
        fontsize=8.3 if combined else None,
        framealpha=0.9,
        borderpad=0.34 if combined else 0.45,
        labelspacing=0.20 if combined else 0.35,
        handlelength=1.35 if combined else 1.8,
        handletextpad=0.42 if combined else 0.8,
    )
    return ax2, {
        "mlp_threshold_params": float(mlp_thresh) if mlp_thresh is not None else None,
        "transformer_threshold_params": float(tfm_thresh) if tfm_thresh is not None else None,
        "row_count": len(rows),
    }


def plot_hidden_dim_candidate(csv_path: Path, output_dir: Path) -> dict[str, str | float | None]:
    rows = _load_hidden_dim_rows(csv_path)

    _setup_hidden_dim_style()
    fig, ax1 = plt.subplots(figsize=(7.0, 6.35))
    ax2, hidden_summary = _draw_hidden_dim_panel(ax1, rows)
    ax2.set_ylabel(r"$\gamma_{\min}$ (normalized margin)", color=COLOR_GAMMA)
    fig.tight_layout(pad=0.35)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig2a_hidden_dim_usability.png"
    pdf_path = output_dir / "fig2a_hidden_dim_usability.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        **hidden_summary,
    }


def render_rf_margin_panel(rf_json: Path, output_dir: Path) -> dict[str, str]:
    scratch = output_dir / "_rf_margin_render"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    plotter = REPO_ROOT / "src/hebbian/expts/margins/plot.py"
    subprocess.run(
        [
            sys.executable,
            str(plotter),
            str(rf_json),
            "--output",
            str(scratch),
            "--case",
            "rkrv",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    stem = rf_json.stem
    src_png = scratch / f"{stem}_rkrv_M.png"
    src_pdf = scratch / f"{stem}_rkrv_M.pdf"
    dst_png = output_dir / "fig2b_rf_margin_m.png"
    dst_pdf = output_dir / "fig2b_rf_margin_m.pdf"
    shutil.copyfile(src_png, dst_png)
    shutil.copyfile(src_pdf, dst_pdf)
    return {"png": str(dst_png), "pdf": str(dst_pdf)}


def _draw_rf_margin_panel(
    ax: plt.Axes,
    rf_json: Path,
    *,
    combined: bool = False,
) -> dict[str, float | str]:
    results = load_results(str(rf_json))
    results["_source_path"] = str(rf_json)
    case = "rkrv"
    q = extract_quantities(results)
    gamma = q["gamma"]
    gamma_std = q["gamma_std"]
    x = q["x"]
    fitted, _ = compute_fitted_bound(case, q)
    r2, mse = r2_mse(gamma, fitted)

    marker_size = math.sqrt(52 if combined else 70)
    ax.plot(
        x,
        gamma,
        "o-",
        color="#202020",
        markersize=marker_size,
        linewidth=2.0 if combined else 2.2,
        label=r"Empirical $\gamma_{\min}$",
    )
    ax.fill_between(
        x,
        gamma - 2 * gamma_std,
        gamma + 2 * gamma_std,
        color="#202020",
        alpha=0.10,
        linewidth=0,
    )
    ax.plot(
        x,
        fitted,
        "s--",
        color="#2E86AB",
        markersize=marker_size,
        linewidth=2.0 if combined else 2.2,
        label="Fitted theory",
    )

    ax.axhline(0, color="0.55", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.grid(True, alpha=0.25, linestyle="--", which="major")
    if q["sweep_type"] in ("F", "M") and np.all(x > 0):
        ax.set_xscale("log")
    ax.margins(y=0.18)
    ax.set_xlabel(q["xlabel"])
    ax.set_ylabel(r"Minimum margin $\gamma_{\min}$")
    ax.legend(
        loc="lower right" if combined else "upper right",
        fontsize=8.2 if combined else 10,
        framealpha=0.92,
        borderpad=0.32 if combined else 0.45,
        labelspacing=0.20 if combined else 0.35,
        handlelength=1.35 if combined else 1.8,
    )

    info_text = "\n".join(
        [
            CASE_TITLES[case],
            CASE_EQUATIONS[case],
            rf"$R^2={r2:.3f}$, MSE={mse:.2e}",
        ]
    )
    ax.text(
        0.5,
        0.965,
        info_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.6 if combined else 11.8,
        linespacing=1.10 if combined else 1.14,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
        },
        zorder=20,
    )
    return {
        "json": str(rf_json),
        "case": case,
        "r2": float(r2),
        "mse": float(mse),
        "point_count": int(len(x)),
    }


def _load_capacity_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            method = str(raw["method"])
            rows.append(
                {
                    "method": method,
                    "method_display": raw.get(
                        "method_display",
                        CAPACITY_METHOD_DISPLAY.get(method, method),
                    ),
                    "d_model": int(float(raw["d_model"])),
                    "num_facts": float(raw["num_facts"]),
                    "num_parameters": float(raw["num_parameters"]),
                }
            )
    if not rows:
        raise ValueError(f"No capacity rows found in {path}")
    return rows


def _fit_capacity_f_log_f(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit W = C * F * (log2(F) + b) in log space."""
    log_y = np.log2(y)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(log_y) & (x > 0) & (y > 1)
    if mask.sum() == 0:
        raise ValueError("Cannot fit capacity curve: no positive finite points.")
    x_fit = x[mask]
    y_fit = y[mask]
    log_y_fit = log_y[mask]

    min_b = -float(np.min(log_y_fit)) + 1e-3
    b_grid = np.linspace(min_b, 24.0, 2000)
    best_c = 1.0
    best_b = 0.0
    best_loss = float("inf")
    for b_value in b_grid:
        basis = y_fit * (log_y_fit + b_value)
        if np.any(~np.isfinite(basis)) or np.any(basis <= 0):
            continue
        log2_c = float(np.mean(np.log2(x_fit) - np.log2(basis)))
        residual = np.log2(x_fit) - (log2_c + np.log2(basis))
        loss = float(np.mean(residual * residual))
        if loss < best_loss:
            best_loss = loss
            best_c = float(np.power(2.0, log2_c))
            best_b = float(b_value)
    return best_c, best_b


def _capacity_f_log_f_curve(facts: np.ndarray, coeff: float, log_offset: float) -> np.ndarray:
    return coeff * facts * (np.log2(facts) + log_offset)


def _format_ratio(value: float) -> str:
    if value >= 10:
        return f"{value:.0f}x"
    return f"{value:.1f}x"


def _format_ratio_range(values: list[float]) -> str:
    finite_values = [float(v) for v in values if np.isfinite(v) and v > 0]
    if not finite_values:
        return ""
    lo = min(finite_values)
    hi = max(finite_values)
    if hi < 3:
        lo_label = f"{lo:.1f}"
        hi_label = f"{hi:.1f}"
    else:
        lo_label = str(int(math.floor(lo + 0.5)))
        hi_label = str(int(math.floor(hi + 0.5)))
    if lo_label == hi_label:
        return f"{lo_label}x"
    return f"{lo_label}-{hi_label}x"


def _add_log_bracket(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    label: str,
    *,
    color: str,
    fontsize: float = 11.5,
    linewidth: float = 1.6,
) -> None:
    if not np.isfinite([x0, x1, y]).all() or min(x0, x1, y) <= 0:
        return
    x_left, x_right = sorted([float(x0), float(x1)])
    tick_bottom = y / 1.08
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [tick_bottom, y, y, tick_bottom],
        color=color,
        linewidth=linewidth,
        alpha=0.9,
        solid_capstyle="round",
        zorder=4,
    )
    ax.text(
        math.sqrt(x_left * x_right),
        y * 1.025,
        label,
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        zorder=5,
    )


def _capacity_savings_reference(
    rows: list[dict[str, float | int | str]],
) -> tuple[int, float, dict[str, float]] | None:
    methods = ["gd", "cf_coord_whitened", "ntk"]
    d_values = sorted({int(row["d_model"]) for row in rows}, reverse=True)
    for d_model in d_values:
        facts_by_method: dict[str, set[float]] = {}
        for method in methods:
            facts_by_method[method] = {
                float(row["num_facts"])
                for row in rows
                if row["method"] == method and int(row["d_model"]) == d_model
            }
        common_facts = set.intersection(*facts_by_method.values())
        if not common_facts:
            continue
        fact = max(common_facts)
        x_by_method: dict[str, float] = {}
        for method in methods:
            candidates = [
                float(row["num_parameters"])
                for row in rows
                if row["method"] == method
                and int(row["d_model"]) == d_model
                and float(row["num_facts"]) == fact
            ]
            if not candidates:
                break
            x_by_method[method] = candidates[0]
        if len(x_by_method) == len(methods):
            return d_model, fact, x_by_method
    return None


def _capacity_savings_ranges(
    rows: list[dict[str, float | int | str]],
) -> dict[str, list[float]]:
    methods = ["gd", "cf_coord_whitened", "ntk"]
    ratios: dict[str, list[float]] = {"data_vs_gd": [], "ntk_vs_data": []}
    d_values = sorted({int(row["d_model"]) for row in rows})
    for d_model in d_values:
        facts_by_method: dict[str, set[float]] = {}
        for method in methods:
            facts_by_method[method] = {
                float(row["num_facts"])
                for row in rows
                if row["method"] == method and int(row["d_model"]) == d_model
            }
        common_facts = set.intersection(*facts_by_method.values())
        for fact in sorted(common_facts):
            x_by_method: dict[str, float] = {}
            for method in methods:
                candidates = [
                    float(row["num_parameters"])
                    for row in rows
                    if row["method"] == method
                    and int(row["d_model"]) == d_model
                    and float(row["num_facts"]) == fact
                ]
                if candidates:
                    x_by_method[method] = candidates[0]
            if len(x_by_method) != len(methods):
                continue
            gd_x = x_by_method["gd"]
            data_x = x_by_method["cf_coord_whitened"]
            ntk_x = x_by_method["ntk"]
            if min(gd_x, data_x, ntk_x) <= 0:
                continue
            ratios["data_vs_gd"].append(data_x / gd_x)
            ratios["ntk_vs_data"].append(ntk_x / data_x)
    return ratios


def _draw_mlp_capacity_panel(
    ax: plt.Axes,
    rows: list[dict[str, float | int | str]],
    *,
    combined: bool = False,
) -> dict[str, str | float | int]:
    methods_present = [
        method for method in CAPACITY_METHOD_ORDER if any(row["method"] == method for row in rows)
    ]
    d_values = sorted({int(row["d_model"]) for row in rows})
    x_max = 0.0
    y_max = 0.0
    y_min = float("inf")
    point_size = 46 if combined else 66
    marker_size = 6.7 if combined else 7.7
    line_width = 1.85 if combined else 2.0

    for method in methods_present:
        method_rows = [row for row in rows if row["method"] == method]
        color = CAPACITY_METHOD_COLOR.get(method, "#666666")
        for d_model in d_values:
            subset = [row for row in method_rows if int(row["d_model"]) == d_model]
            if not subset:
                continue
            subset = sorted(subset, key=lambda row: float(row["num_parameters"]))
            x = np.array([float(row["num_parameters"]) for row in subset], dtype=float)
            y = np.array([float(row["num_facts"]) for row in subset], dtype=float)
            x_max = max(x_max, float(np.max(x)))
            y_max = max(y_max, float(np.max(y)))
            y_min = min(y_min, float(np.min(y)))

            marker = CAPACITY_D_MARKER.get(d_model, "o")
            base_linestyle = CAPACITY_D_LINESTYLE.get(d_model, "-")
            if method in CAPACITY_FIT_METHODS and len(subset) >= 3:
                ax.scatter(
                    x,
                    y,
                    marker=marker,
                    color=color,
                    s=point_size,
                    edgecolor="black",
                    linewidth=0.85 if combined else 0.95,
                    alpha=0.95,
                    zorder=3,
                )
                fit_c, fit_b = _fit_capacity_f_log_f(x, y)
                fit_log_y = np.linspace(float(np.min(np.log2(y))), float(np.max(np.log2(y))), 180)
                fit_y = np.power(2.0, fit_log_y)
                fit_x = _capacity_f_log_f_curve(fit_y, fit_c, fit_b)
                ax.plot(
                    fit_x,
                    fit_y,
                    linestyle=base_linestyle,
                    color=color,
                    linewidth=2.0 if combined else 2.15,
                    alpha=0.92,
                    zorder=2,
                )
            else:
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linestyle=CAPACITY_METHOD_LINESTYLE.get(method, base_linestyle),
                    color=color,
                    linewidth=line_width,
                    markersize=marker_size,
                    markeredgewidth=0.85 if combined else 0.95,
                    markeredgecolor="black",
                    alpha=0.93,
                )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of Parameters (W)")
    ax.set_ylabel("Number of Facts (F)")
    ax.grid(True, alpha=0.3, linestyle="--", which="major")

    summary: dict[str, str | float | int] = {}
    reference = _capacity_savings_reference(rows)
    if reference is not None:
        d_model, fact, x_by_method = reference
        gd_x = x_by_method["gd"]
        data_x = x_by_method["cf_coord_whitened"]
        ntk_x = x_by_method["ntk"]
        ranges = _capacity_savings_ranges(rows)
        gd_label = _format_ratio_range(ranges["data_vs_gd"]) or _format_ratio(data_x / gd_x)
        ntk_label = _format_ratio_range(ranges["ntk_vs_data"]) or _format_ratio(ntk_x / data_x)
        ax.set_ylim(y_min / 1.25, max(y_max * 1.12, fact * 1.82))
        ax.set_xlim(min(float(row["num_parameters"]) for row in rows) / 1.35, x_max * 1.4)
        _add_log_bracket(
            ax,
            gd_x,
            data_x,
            fact * 1.28,
            f"{gd_label} gap to GD",
            color=CAPACITY_METHOD_COLOR["cf_coord_whitened"],
            fontsize=7.4 if combined else 11.5,
            linewidth=1.25 if combined else 1.6,
        )
        _add_log_bracket(
            ax,
            data_x,
            ntk_x,
            fact * 1.68,
            f"{ntk_label} fewer than NTK",
            color=CAPACITY_METHOD_COLOR["ntk"],
            fontsize=7.4 if combined else 11.5,
            linewidth=1.25 if combined else 1.6,
        )
        summary.update(
            {
                "bracket_d_model": d_model,
                "bracket_num_facts": fact,
                "data_dependent_vs_gd_ratio": data_x / gd_x,
                "ntk_vs_data_dependent_ratio": ntk_x / data_x,
                "data_dependent_vs_gd_range": gd_label,
                "ntk_vs_data_dependent_range": ntk_label,
            }
        )

    method_handles = [
        Line2D(
            [0],
            [0],
            color=CAPACITY_METHOD_COLOR.get(method, "#666666"),
            linestyle="-",
            marker="o",
            markeredgecolor="black",
            markeredgewidth=0.85 if combined else 0.9,
            markersize=6.4 if combined else 7.3,
        )
        for method in methods_present
    ]
    if method_handles:
        method_legend = ax.legend(
            handles=method_handles,
            labels=[CAPACITY_METHOD_DISPLAY.get(method, method) for method in methods_present],
            title="Method" if not combined else None,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.015),
            fontsize=7.35 if combined else 10.2,
            title_fontsize=10.8,
            framealpha=0.88,
            borderpad=0.28 if combined else 0.4,
            labelspacing=0.15 if combined else 0.27,
            handlelength=1.10 if combined else 1.5,
            handletextpad=0.38 if combined else 0.8,
            borderaxespad=0.10 if combined else 0.18,
        )
        ax.add_artist(method_legend)
    d_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=CAPACITY_D_LINESTYLE.get(d_model, "-"),
            marker=CAPACITY_D_MARKER.get(d_model, "o"),
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.85 if combined else 0.9,
            markersize=6.4 if combined else 7.3,
        )
        for d_model in d_values
    ]
    if d_handles:
        ax.legend(
            handles=d_handles,
            labels=[f"d={d_model}" for d_model in d_values],
            loc="upper left",
            fontsize=7.1 if combined else 8.6,
            framealpha=0.88,
            borderpad=0.22 if combined else 0.3,
            labelspacing=0.10 if combined else 0.16,
            handlelength=1.10 if combined else 1.8,
            handletextpad=0.36 if combined else 0.8,
        )
    return summary


def plot_mlp_capacity_candidate(points_csv: Path, output_dir: Path) -> dict[str, str | float | int]:
    rows = _load_capacity_rows(points_csv)
    methods_present = [
        method for method in CAPACITY_METHOD_ORDER if any(row["method"] == method for row in rows)
    ]
    d_values = sorted({int(row["d_model"]) for row in rows})

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 9.5,
            "savefig.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(7.5, 6.25))

    x_max = 0.0
    y_max = 0.0
    y_min = float("inf")
    for method in methods_present:
        method_rows = [row for row in rows if row["method"] == method]
        color = CAPACITY_METHOD_COLOR.get(method, "#666666")
        for d_model in d_values:
            subset = [row for row in method_rows if int(row["d_model"]) == d_model]
            if not subset:
                continue
            subset = sorted(subset, key=lambda row: float(row["num_parameters"]))
            x = np.array([float(row["num_parameters"]) for row in subset], dtype=float)
            y = np.array([float(row["num_facts"]) for row in subset], dtype=float)
            x_max = max(x_max, float(np.max(x)))
            y_max = max(y_max, float(np.max(y)))
            y_min = min(y_min, float(np.min(y)))

            marker = CAPACITY_D_MARKER.get(d_model, "o")
            base_linestyle = CAPACITY_D_LINESTYLE.get(d_model, "-")
            if method in CAPACITY_FIT_METHODS and len(subset) >= 3:
                ax.scatter(
                    x,
                    y,
                    marker=marker,
                    color=color,
                    s=66,
                    edgecolor="black",
                    linewidth=0.95,
                    alpha=0.95,
                    zorder=3,
                )
                fit_c, fit_b = _fit_capacity_f_log_f(x, y)
                fit_log_y = np.linspace(float(np.min(np.log2(y))), float(np.max(np.log2(y))), 180)
                fit_y = np.power(2.0, fit_log_y)
                fit_x = _capacity_f_log_f_curve(fit_y, fit_c, fit_b)
                ax.plot(
                    fit_x,
                    fit_y,
                    linestyle=base_linestyle,
                    color=color,
                    linewidth=2.15,
                    alpha=0.92,
                    zorder=2,
                )
            else:
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linestyle=CAPACITY_METHOD_LINESTYLE.get(method, base_linestyle),
                    color=color,
                    linewidth=2.0,
                    markersize=7.7,
                    markeredgewidth=0.95,
                    markeredgecolor="black",
                    alpha=0.93,
                )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of Parameters (W)")
    ax.set_ylabel("Number of Facts (F)")
    ax.grid(True, alpha=0.3, linestyle="--", which="major")

    reference = _capacity_savings_reference(rows)
    summary: dict[str, str | float | int] = {
        "png": str(output_dir / "fig2c_mlp_capacity_isotropic_fit_brackets.png"),
        "pdf": str(output_dir / "fig2c_mlp_capacity_isotropic_fit_brackets.pdf"),
        "points_csv": str(points_csv),
    }
    if reference is not None:
        d_model, fact, x_by_method = reference
        gd_x = x_by_method["gd"]
        data_x = x_by_method["cf_coord_whitened"]
        ntk_x = x_by_method["ntk"]
        ranges = _capacity_savings_ranges(rows)
        gd_label = _format_ratio_range(ranges["data_vs_gd"]) or _format_ratio(data_x / gd_x)
        ntk_label = _format_ratio_range(ranges["ntk_vs_data"]) or _format_ratio(ntk_x / data_x)
        ax.set_ylim(y_min / 1.25, max(y_max * 1.12, fact * 1.82))
        ax.set_xlim(min(float(row["num_parameters"]) for row in rows) / 1.35, x_max * 1.4)
        _add_log_bracket(
            ax,
            gd_x,
            data_x,
            fact * 1.28,
            f"{gd_label} gap to GD",
            color=CAPACITY_METHOD_COLOR["cf_coord_whitened"],
        )
        _add_log_bracket(
            ax,
            data_x,
            ntk_x,
            fact * 1.68,
            f"{ntk_label} fewer than NTK",
            color=CAPACITY_METHOD_COLOR["ntk"],
        )
        summary.update(
            {
                "bracket_d_model": d_model,
                "bracket_num_facts": fact,
                "data_dependent_vs_gd_ratio": data_x / gd_x,
                "ntk_vs_data_dependent_ratio": ntk_x / data_x,
                "data_dependent_vs_gd_range": gd_label,
                "ntk_vs_data_dependent_range": ntk_label,
            }
        )

    method_handles = [
        Line2D(
            [0],
            [0],
            color=CAPACITY_METHOD_COLOR.get(method, "#666666"),
            linestyle="-",
            marker="o",
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=7.3,
        )
        for method in methods_present
    ]
    if method_handles:
        method_legend = ax.legend(
            handles=method_handles,
            labels=[CAPACITY_METHOD_DISPLAY.get(method, method) for method in methods_present],
            title="Method",
            loc="lower right",
            bbox_to_anchor=(0.992, 0.02),
            fontsize=10.2,
            title_fontsize=10.8,
            framealpha=0.88,
            borderpad=0.4,
            labelspacing=0.27,
            handlelength=1.5,
            borderaxespad=0.18,
        )
        ax.add_artist(method_legend)
    d_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=CAPACITY_D_LINESTYLE.get(d_model, "-"),
            marker=CAPACITY_D_MARKER.get(d_model, "o"),
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=7.3,
        )
        for d_model in d_values
    ]
    if d_handles:
        ax.legend(
            handles=d_handles,
            labels=[f"d={d_model}" for d_model in d_values],
            loc="upper left",
            fontsize=8.6,
            framealpha=0.88,
            borderpad=0.3,
            labelspacing=0.16,
        )

    fig.tight_layout(pad=0.4)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig2c_mlp_capacity_isotropic_fit_brackets.png"
    pdf_path = output_dir / "fig2c_mlp_capacity_isotropic_fit_brackets.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    summary["png"] = str(png_path)
    summary["pdf"] = str(pdf_path)
    return summary


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _trim_whitespace(image: Image.Image, padding: int = 18) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    nonwhite = np.any(arr < 248, axis=2)
    if not np.any(nonwhite):
        return image
    ys, xs = np.where(nonwhite)
    left = max(int(xs.min()) - padding, 0)
    right = min(int(xs.max()) + padding + 1, image.width)
    top = max(int(ys.min()) - padding, 0)
    bottom = min(int(ys.max()) + padding + 1, image.height)
    return image.crop((left, top, right, bottom))


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    center_x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (20, 20, 20),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    draw.text((center_x - width / 2, y), text, fill=fill, font=font)


def stitch_panels(
    panel_paths: list[Path],
    output_path: Path,
    *,
    target_height: int = 1500,
    add_labels: bool = False,
    panel_titles: list[str] | None = None,
) -> str:
    images = [_trim_whitespace(Image.open(p).convert("RGB")) for p in panel_paths]
    resized = []
    for image in images:
        scale = target_height / image.height
        width = int(round(image.width * scale))
        resized.append(image.resize((width, target_height), Image.Resampling.LANCZOS))

    gap = 28
    pad_x = 30
    pad_y = 30
    title_band = 138 if panel_titles else 0
    total_width = sum(img.width for img in resized) + gap * (len(resized) - 1) + 2 * pad_x
    total_height = target_height + 2 * pad_y + title_band
    canvas = Image.new("RGB", (total_width, total_height), "white")

    draw = ImageDraw.Draw(canvas)
    label_font = _load_font(72)
    title_font = _load_font(72)
    x_cursor = pad_x
    for idx, image in enumerate(resized):
        panel_y = pad_y + title_band
        if panel_titles:
            title = panel_titles[idx]
            if add_labels:
                title = f"{chr(ord('A') + idx)}. {title}"
            _draw_centered_text(
                draw,
                title,
                center_x=x_cursor + image.width // 2,
                y=pad_y + 10,
                font=title_font,
            )
        canvas.paste(image, (x_cursor, panel_y))
        if add_labels and not panel_titles:
            draw.text(
                (x_cursor + 20, pad_y + 14),
                chr(ord("A") + idx),
                fill=(20, 20, 20),
                font=label_font,
            )
        x_cursor += image.width + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)


def plot_native_combined_1x3(
    hidden_dim_csv: Path,
    rf_margin_json: Path,
    mlp_capacity_points_csv: Path,
    output_dir: Path,
) -> dict[str, str | dict[str, str | float | int | None]]:
    hidden_rows = _load_hidden_dim_rows(hidden_dim_csv)
    capacity_rows = _load_capacity_rows(mlp_capacity_points_csv)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10.5,
            "axes.titlesize": 15.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 12.4,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 5.65))
    _, hidden_summary = _draw_hidden_dim_panel(axes[0], hidden_rows, combined=True)
    margin_summary = _draw_rf_margin_panel(axes[1], rf_margin_json, combined=True)
    capacity_summary = _draw_mlp_capacity_panel(axes[2], capacity_rows, combined=True)

    for idx, (ax, title) in enumerate(zip(axes, PANEL_TITLES)):
        ax.set_title(f"{chr(ord('A') + idx)}. {title}", pad=13)

    fig.subplots_adjust(left=0.057, right=0.992, bottom=0.18, top=0.82, wspace=0.34)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig2_margin_capacity_1x3_native.png"
    pdf_path = output_dir / "fig2_margin_capacity_1x3_native.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "fig2a": hidden_summary,
        "fig2b": margin_summary,
        "fig2c": capacity_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/paper/figures/fig2_margin_capacity",
        help="Directory for generated Figure 2 panels.",
    )
    parser.add_argument(
        "--hidden-dim-csv",
        default="artifacts/paper/figures/fig_hidden_dim_margin/figure_points.csv",
        help="Figure-points CSV generated by the fig_hidden_dim_margin target.",
    )
    parser.add_argument(
        "--rf-margin-json",
        default="artifacts/paper/results/margins/rf_margin_m_d64_f256_30pts.json",
        help="Margin JSON generated by the fig_rf_margin_m target.",
    )
    parser.add_argument(
        "--mlp-capacity-points-csv",
        default="artifacts/paper/figures/fig_mlp_capacity_isotropic/figure_points.csv",
        help="Figure-points CSV generated by the fig_mlp_capacity_isotropic target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _repo_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden = plot_hidden_dim_candidate(_repo_path(args.hidden_dim_csv), output_dir)
    rf_margin = render_rf_margin_panel(_repo_path(args.rf_margin_json), output_dir)
    mlp_capacity = plot_mlp_capacity_candidate(
        _repo_path(args.mlp_capacity_points_csv),
        output_dir,
    )

    composite = stitch_panels(
        [
            Path(hidden["png"]),
            Path(rf_margin["png"]),
            Path(mlp_capacity["png"]),
        ],
        output_dir / "fig2_margin_capacity_1x3.png",
        add_labels=False,
    )
    labeled_composite = stitch_panels(
        [
            Path(hidden["png"]),
            Path(rf_margin["png"]),
            Path(mlp_capacity["png"]),
        ],
        output_dir / "fig2_margin_capacity_1x3_labeled.png",
        add_labels=True,
    )
    titled_composite = stitch_panels(
        [
            Path(hidden["png"]),
            Path(rf_margin["png"]),
            Path(mlp_capacity["png"]),
        ],
        output_dir / "fig2_margin_capacity_1x3_titled.png",
        add_labels=False,
        panel_titles=PANEL_TITLES,
    )
    labeled_titled_composite = stitch_panels(
        [
            Path(hidden["png"]),
            Path(rf_margin["png"]),
            Path(mlp_capacity["png"]),
        ],
        output_dir / "fig2_margin_capacity_1x3_labeled_titled.png",
        add_labels=True,
        panel_titles=PANEL_TITLES,
    )
    native_composite = plot_native_combined_1x3(
        _repo_path(args.hidden_dim_csv),
        _repo_path(args.rf_margin_json),
        _repo_path(args.mlp_capacity_points_csv),
        output_dir,
    )

    summary = {
        "status": {
            "fig2a": "generated_from_experiment_outputs",
            "fig2b": "generated_from_experiment_outputs",
            "fig2c": "generated_from_experiment_outputs",
        },
        "outputs": {
            "fig2a": hidden,
            "fig2b": rf_margin,
            "fig2c": mlp_capacity,
            "composite": composite,
            "labeled_composite": labeled_composite,
            "titled_composite": titled_composite,
            "labeled_titled_composite": labeled_titled_composite,
            "native_composite": native_composite,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
