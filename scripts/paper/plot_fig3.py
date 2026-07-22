#!/usr/bin/env python3
"""Regenerate the individual and composite Figure 3 panels from run outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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
CAPACITY_FIT_METHODS = {"gd", "cf_coord_whitened"}
CAPACITY_PLOT_PARAM_BRACKET_MIDPOINTS = {("gd", 256)}

ATTENTION_COLOR_BY_D = {
    64: "#1f77b4",
    96: "#2ca02c",
    128: "#d62728",
}
ATTENTION_MARKER_BY_D = {
    64: "o",
    96: "s",
    128: "D",
}

FACT_METHOD_ORDER = ["memit", "alpha_edit", "rome", "gd_construction"]
FACT_METHOD_DISPLAY = {
    "memit": "MEMIT",
    "alpha_edit": "AlphaEdit",
    "rome": "ROME",
    "gd_construction": "MLP Swapping",
}
FACT_METHOD_COLOR = {
    "memit": "#E63946",
    "alpha_edit": "#2E86AB",
    "rome": "#F18F01",
    "gd_construction": "#16A085",
}
FACT_METHOD_MARKER = {
    "memit": "o",
    "alpha_edit": "s",
    "rome": "^",
    "gd_construction": "D",
}
COMBINED_PANEL_TITLES = [
    "A. Attention noise floor",
    "B. Transformer capacity",
    "C. Fact editing",
]


def _default_attention_csv() -> Path:
    return Path("artifacts/paper/figures/fig_attention_noise_floor/figure_points.csv")


def _default_transformer_csv() -> Path:
    return Path("artifacts/paper/results/transformer_storage_capacity/capacity_points.csv")


def _default_fact_editing_csv() -> Path:
    return Path("artifacts/paper/results/fact_editing/results/best_results.csv")


def _default_output_dir() -> Path:
    return Path("artifacts/paper/figures/fig3_transformer_integration")


def _save_panel(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _pow2_label(value: int) -> str:
    if value > 0 and (value & (value - 1)) == 0:
        return rf"$2^{{{int(np.log2(value))}}}$"
    return str(value)


def _format_percent(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 0.15:
        return f"{rounded:.0f}%"
    return f"{value:.1f}%"


def _capacity_plot_param_count(row: pd.Series) -> float:
    """Return the x-coordinate used for the paper plot."""
    method = str(row["method"])
    facts = int(row["capacity_num_facts"])
    if (method, facts) in CAPACITY_PLOT_PARAM_BRACKET_MIDPOINTS:
        lower = float(row.get("source_lower_w", float("nan")))
        upper = float(row.get("source_upper_w", float("nan")))
        if np.isfinite(lower) and np.isfinite(upper) and lower > 0 and upper > 0:
            return float(math.sqrt(lower * upper))
    return float(row["mlp_param_count"])


def _fit_capacity_f_log_f(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit the theory-aligned curve W = C * F * (log2(F) + b) in log space."""
    log_y = np.log2(y)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(log_y) & (x > 0) & (y > 1)
    if mask.sum() == 0:
        raise ValueError("Cannot fit W = C F (log F + b): no positive finite points.")
    x_fit = x[mask]
    y_fit = y[mask]
    log_y_fit = log_y[mask]

    min_b = -float(np.min(log_y_fit)) + 1e-3
    # Enough room to model finite-size constants without degenerating into an
    # unconstrained empirical smoother.
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
    basis = facts * (np.log2(facts) + log_offset)
    return coeff * basis


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
) -> None:
    if not np.isfinite([x0, x1, y]).all() or min(x0, x1, y) <= 0:
        return
    x_left, x_right = sorted([float(x0), float(x1)])
    tick_bottom = y / 1.08
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [tick_bottom, y, y, tick_bottom],
        color=color,
        linewidth=1.45,
        alpha=0.9,
        solid_capstyle="round",
        zorder=4,
    )
    ax.text(
        math.sqrt(x_left * x_right),
        y * 1.025,
        label,
        color=color,
        fontsize=8.6,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
        zorder=5,
    )


def _capacity_savings_reference(
    points: pd.DataFrame,
) -> tuple[float, dict[str, float]] | None:
    methods = ["gd", "cf_coord_whitened", "ntk"]
    common_facts: set[int] | None = None
    for method in methods:
        method_facts = set(
            int(v)
            for v in points.loc[points["method"] == method, "capacity_num_facts"].to_numpy()
        )
        common_facts = method_facts if common_facts is None else common_facts & method_facts
    if not common_facts:
        return None
    fact = float(max(common_facts))
    x_by_method: dict[str, float] = {}
    for method in methods:
        subset = points[
            (points["method"] == method)
            & (points["capacity_num_facts"].astype(int) == int(fact))
        ]
        if subset.empty:
            return None
        x_by_method[method] = float(subset.iloc[0]["plot_mlp_param_count"])
    return fact, x_by_method


def _capacity_savings_ranges(points: pd.DataFrame) -> dict[str, list[float]]:
    methods = ["gd", "cf_coord_whitened", "ntk"]
    ratios: dict[str, list[float]] = {"data_vs_gd": [], "ntk_vs_data": []}
    common_facts: set[int] | None = None
    for method in methods:
        method_facts = set(
            int(v)
            for v in points.loc[points["method"] == method, "capacity_num_facts"].to_numpy()
        )
        common_facts = method_facts if common_facts is None else common_facts & method_facts
    if not common_facts:
        return ratios

    for fact in sorted(common_facts):
        x_by_method: dict[str, float] = {}
        for method in methods:
            subset = points[
                (points["method"] == method)
                & (points["capacity_num_facts"].astype(int) == int(fact))
            ]
            if not subset.empty:
                x_by_method[method] = float(subset.iloc[0]["plot_mlp_param_count"])
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


def _annotate_capacity_savings(
    ax: plt.Axes,
    points: pd.DataFrame,
    *,
    x_right_multiplier: float,
    x_right_limit: float | None = None,
) -> None:
    x_values = points["plot_mlp_param_count"].to_numpy(dtype=float)
    y_values = points["capacity_num_facts"].to_numpy(dtype=float)
    x_values = x_values[np.isfinite(x_values) & (x_values > 0)]
    y_values = y_values[np.isfinite(y_values) & (y_values > 0)]
    if len(x_values) == 0 or len(y_values) == 0:
        return

    x_right = float(np.max(x_values)) * x_right_multiplier
    if x_right_limit is not None:
        x_right = max(float(x_right_limit), float(np.max(x_values)) * 1.05)
    ax.set_xlim(float(np.min(x_values)) / 1.25, x_right)

    reference = _capacity_savings_reference(points)
    if reference is None:
        return
    fact, x_by_method = reference
    gd_x = x_by_method["gd"]
    data_x = x_by_method["cf_coord_whitened"]
    ntk_x = x_by_method["ntk"]
    ranges = _capacity_savings_ranges(points)
    gd_label = _format_ratio_range(ranges["data_vs_gd"]) or _format_ratio(data_x / gd_x)
    ntk_label = _format_ratio_range(ranges["ntk_vs_data"]) or _format_ratio(ntk_x / data_x)
    ax.set_ylim(float(np.min(y_values)) / 1.2, max(float(np.max(y_values)) * 2.15, fact * 2.15))
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


def _set_attention_style() -> None:
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "font.family": "sans-serif",
            "font.size": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "lines.linewidth": 2.35,
            "lines.markeredgewidth": 1.0,
            "lines.markeredgecolor": "black",
            "lines.markersize": math.sqrt(110),
        }
    )


def _set_fact_style() -> None:
    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 17,
            "axes.labelweight": "bold",
            "font.family": "sans-serif",
            "font.size": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "lines.linewidth": 2.6,
            "lines.markeredgewidth": 1.2,
            "lines.markeredgecolor": "black",
            "lines.markersize": math.sqrt(125),
        }
    )


def plot_attention_noise(points_csv: Path, output_dir: Path) -> list[Path]:
    points = pd.read_csv(points_csv)
    points = points[points["junk_len"].isin([2, 4, 8, 16])].copy()
    points = points[np.isfinite(points["attn_noise_l2_floor"])]
    if points.empty:
        raise ValueError("No attention-noise rows available to plot.")

    _set_attention_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.9))

    for (d_model, num_facts), group in points.groupby(["d_model", "num_facts"], sort=True):
        group = group.sort_values("junk_len")
        d_model = int(d_model)
        num_facts = int(num_facts)
        ax.plot(
            group["junk_len"].to_numpy(dtype=float),
            group["attn_noise_l2_floor"].to_numpy(dtype=float),
            color=ATTENTION_COLOR_BY_D.get(d_model, "#444444"),
            linestyle="-",
            marker=ATTENTION_MARKER_BY_D.get(d_model, "o"),
            markeredgecolor="black",
            markeredgewidth=1.0,
            label=f"d={d_model}, F={num_facts}",
            alpha=0.92,
        )

    x_vals = sorted(int(v) for v in points["junk_len"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([_pow2_label(v) for v in x_vals])
    ax.set_xlabel(r"Junk Length $J$")
    ax.set_ylabel("Attention Noise Floor (L2)")
    ax.grid(True, which="major", alpha=0.3, linestyle="--")
    ax.legend(
        loc="upper left",
        fontsize=13,
        title="Model",
        title_fontsize=13,
        frameon=True,
        framealpha=0.92,
    )
    fig.tight_layout()
    return _save_panel(fig, output_dir, "fig3a_attention_noise_floor")


def _plot_attention_noise_on_axis(ax: plt.Axes, points_csv: Path) -> None:
    points = pd.read_csv(points_csv)
    points = points[points["junk_len"].isin([2, 4, 8, 16])].copy()
    points = points[np.isfinite(points["attn_noise_l2_floor"])]
    if points.empty:
        raise ValueError("No attention-noise rows available to plot.")

    for (d_model, num_facts), group in points.groupby(["d_model", "num_facts"], sort=True):
        group = group.sort_values("junk_len")
        d_model = int(d_model)
        num_facts = int(num_facts)
        ax.plot(
            group["junk_len"].to_numpy(dtype=float),
            group["attn_noise_l2_floor"].to_numpy(dtype=float),
            color=ATTENTION_COLOR_BY_D.get(d_model, "#444444"),
            linestyle="-",
            marker=ATTENTION_MARKER_BY_D.get(d_model, "o"),
            markersize=8.4,
            markeredgecolor="black",
            markeredgewidth=0.9,
            linewidth=2.35,
            label=f"d={d_model}, F={num_facts}",
            alpha=0.92,
        )

    x_vals = sorted(int(v) for v in points["junk_len"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([_pow2_label(v) for v in x_vals])
    ax.set_xlabel(r"Junk Length $J$")
    ax.set_ylabel("Attention Noise Floor (L2)")
    ax.grid(True, which="major", alpha=0.3, linestyle="--")
    ax.legend(
        loc="upper left",
        fontsize=9.7,
        title="Model",
        title_fontsize=9.7,
        frameon=True,
        framealpha=0.92,
        borderpad=0.36,
        labelspacing=0.22,
        handlelength=1.45,
    )


def plot_transformer_capacity(points_csv: Path, output_dir: Path) -> list[Path]:
    points = pd.read_csv(points_csv)
    points = points[
        np.isfinite(points["capacity_num_facts"])
        & np.isfinite(points["mlp_param_count"])
    ].copy()
    if points.empty:
        raise ValueError("No transformer-capacity rows available to plot.")
    points["plot_mlp_param_count"] = points.apply(_capacity_plot_param_count, axis=1)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    methods_present = [
        method for method in CAPACITY_METHOD_ORDER if (points["method"] == method).any()
    ]
    for method in methods_present:
        subset = points[points["method"] == method].sort_values("plot_mlp_param_count")
        x = subset["plot_mlp_param_count"].to_numpy(dtype=float)
        y = subset["capacity_num_facts"].to_numpy(dtype=float)
        color = CAPACITY_METHOD_COLOR.get(method, "#666666")
        linestyle = CAPACITY_METHOD_LINESTYLE.get(method, "-")
        if method in CAPACITY_FIT_METHODS and len(subset) >= 3:
            ax.scatter(
                x,
                y,
                marker="o",
                color=color,
                s=64,
                edgecolor="black",
                linewidth=1.0,
                alpha=0.95,
                zorder=3,
            )
            log_y = np.log2(y)
            fit_c, fit_b = _fit_capacity_f_log_f(x, y)
            fit_log_y = np.linspace(float(np.min(log_y)), float(np.max(log_y)), 256)
            fit_y = np.power(2.0, fit_log_y)
            fit_x = _capacity_f_log_f_curve(fit_y, fit_c, fit_b)
            ax.plot(
                fit_x,
                fit_y,
                linestyle="-",
                color=color,
                linewidth=2.4,
                alpha=0.86,
                zorder=2,
            )
        else:
            ax.plot(
                x,
                y,
                marker="o",
                linestyle=linestyle,
                color=color,
                linewidth=2.2,
                markersize=8,
                markeredgewidth=1.0,
                markeredgecolor="black",
                alpha=0.95,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of MLP Parameters")
    ax.set_ylabel("Number of Facts")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    _annotate_capacity_savings(ax, points, x_right_multiplier=6.0, x_right_limit=2.0**26)

    handles = [
        Line2D(
            [0],
            [0],
            color=CAPACITY_METHOD_COLOR.get(method, "#666666"),
            linestyle=CAPACITY_METHOD_LINESTYLE.get(method, "-"),
            marker="o",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=8,
        )
        for method in methods_present
    ]
    if handles:
        ax.legend(
            handles=handles,
            labels=[CAPACITY_METHOD_DISPLAY.get(method, method) for method in methods_present],
            title="Method",
            loc="lower right",
            bbox_to_anchor=(0.992, 0.02),
            fontsize=10,
            title_fontsize=10.7,
            framealpha=0.9,
            borderpad=0.45,
            labelspacing=0.3,
            handlelength=1.6,
            borderaxespad=0.18,
        )

    fig.tight_layout()
    return _save_panel(fig, output_dir, "fig3b_transformer_storage_capacity")


def _plot_transformer_capacity_on_axis(ax: plt.Axes, points_csv: Path) -> None:
    points = pd.read_csv(points_csv)
    points = points[
        np.isfinite(points["capacity_num_facts"])
        & np.isfinite(points["mlp_param_count"])
    ].copy()
    if points.empty:
        raise ValueError("No transformer-capacity rows available to plot.")
    points["plot_mlp_param_count"] = points.apply(_capacity_plot_param_count, axis=1)

    methods_present = [
        method for method in CAPACITY_METHOD_ORDER if (points["method"] == method).any()
    ]
    for method in methods_present:
        subset = points[points["method"] == method].sort_values("plot_mlp_param_count")
        x = subset["plot_mlp_param_count"].to_numpy(dtype=float)
        y = subset["capacity_num_facts"].to_numpy(dtype=float)
        color = CAPACITY_METHOD_COLOR.get(method, "#666666")
        linestyle = CAPACITY_METHOD_LINESTYLE.get(method, "-")
        if method in CAPACITY_FIT_METHODS and len(subset) >= 3:
            ax.scatter(
                x,
                y,
                marker="o",
                color=color,
                s=60,
                edgecolor="black",
                linewidth=0.9,
                alpha=0.95,
                zorder=3,
            )
            log_y = np.log2(y)
            fit_c, fit_b = _fit_capacity_f_log_f(x, y)
            fit_log_y = np.linspace(float(np.min(log_y)), float(np.max(log_y)), 256)
            fit_y = np.power(2.0, fit_log_y)
            fit_x = _capacity_f_log_f_curve(fit_y, fit_c, fit_b)
            ax.plot(
                fit_x,
                fit_y,
                linestyle="-",
                color=color,
                linewidth=2.3,
                alpha=0.86,
                zorder=2,
            )
        else:
            ax.plot(
                x,
                y,
                marker="o",
                linestyle=linestyle,
                color=color,
                linewidth=2.15,
                markersize=7.5,
                markeredgewidth=0.9,
                markeredgecolor="black",
                alpha=0.95,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of MLP Parameters")
    ax.set_ylabel("Number of Facts")
    ax.grid(True, which="both", linestyle="--", linewidth=0.55, alpha=0.48)
    _annotate_capacity_savings(ax, points, x_right_multiplier=8.0, x_right_limit=2.0**26)

    handles = [
        Line2D(
            [0],
            [0],
            color=CAPACITY_METHOD_COLOR.get(method, "#666666"),
            linestyle=CAPACITY_METHOD_LINESTYLE.get(method, "-"),
            marker="o",
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=7.3,
        )
        for method in methods_present
    ]
    if handles:
        ax.legend(
            handles=handles,
            labels=[CAPACITY_METHOD_DISPLAY.get(method, method) for method in methods_present],
            loc="lower right",
            bbox_to_anchor=(0.995, 0.015),
            fontsize=8.9,
            framealpha=0.86,
            borderpad=0.34,
            labelspacing=0.2,
            handlelength=1.25,
            handletextpad=0.4,
            borderaxespad=0.1,
        )


def plot_fact_editing_score(points_csv: Path, output_dir: Path) -> list[Path]:
    points = pd.read_csv(points_csv)
    points = points[np.isfinite(points["percent_edited"]) & np.isfinite(points["score"])].copy()
    if points.empty:
        raise ValueError("No fact-editing rows available to plot.")

    _set_fact_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    handles: list[Line2D] = []
    labels: list[str] = []
    for method in FACT_METHOD_ORDER:
        subset = points[points["method"] == method].sort_values("percent_edited")
        if subset.empty:
            continue
        ax.plot(
            subset["percent_edited"].to_numpy(dtype=float),
            subset["score"].to_numpy(dtype=float),
            color=FACT_METHOD_COLOR.get(method, "#666666"),
            linestyle="-",
            marker=FACT_METHOD_MARKER.get(method, "o"),
            linewidth=2.6,
            markersize=math.sqrt(125),
            markeredgewidth=1.2,
            markeredgecolor="black",
            alpha=0.92,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=FACT_METHOD_COLOR.get(method, "#666666"),
                linestyle="-",
                marker=FACT_METHOD_MARKER.get(method, "o"),
                markeredgecolor="black",
                markeredgewidth=1.2,
                markersize=9.5,
            )
        )
        labels.append(FACT_METHOD_DISPLAY.get(method, method))

    x_values = sorted(points["percent_edited"].unique())
    ax.set_xticks(x_values)
    ax.set_xticklabels([_format_percent(float(v)) for v in x_values])
    ax.set_xlabel("Edited Facts (%)")
    ax.set_ylabel("Edit Score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Edit Score vs. Edited Facts (%)")
    ax.grid(True, alpha=0.3, linestyle="--", which="major")
    if handles:
        ax.legend(
            handles=handles,
            labels=labels,
            loc="lower right",
            bbox_to_anchor=(0.985, 0.11),
            fontsize=11,
            framealpha=0.9,
            borderpad=0.5,
            labelspacing=0.35,
            handlelength=1.7,
        )

    return _save_panel(fig, output_dir, "fig3c_fact_editing_score")


def _plot_fact_editing_score_on_axis(ax: plt.Axes, points_csv: Path) -> None:
    points = pd.read_csv(points_csv)
    points = points[np.isfinite(points["percent_edited"]) & np.isfinite(points["score"])].copy()
    if points.empty:
        raise ValueError("No fact-editing rows available to plot.")

    handles: list[Line2D] = []
    labels: list[str] = []
    for method in FACT_METHOD_ORDER:
        subset = points[points["method"] == method].sort_values("percent_edited")
        if subset.empty:
            continue
        ax.plot(
            subset["percent_edited"].to_numpy(dtype=float),
            subset["score"].to_numpy(dtype=float),
            color=FACT_METHOD_COLOR.get(method, "#666666"),
            linestyle="-",
            marker=FACT_METHOD_MARKER.get(method, "o"),
            linewidth=2.35,
            markersize=8.4,
            markeredgewidth=0.9,
            markeredgecolor="black",
            alpha=0.92,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=FACT_METHOD_COLOR.get(method, "#666666"),
                linestyle="-",
                marker=FACT_METHOD_MARKER.get(method, "o"),
                markeredgecolor="black",
                markeredgewidth=0.9,
                markersize=7.6,
            )
        )
        labels.append(FACT_METHOD_DISPLAY.get(method, method))

    x_values = sorted(points["percent_edited"].unique())
    ax.set_xticks(x_values)
    ax.set_xticklabels([_format_percent(float(v)) for v in x_values])
    ax.set_xlabel("Edited Facts (%)")
    ax.set_ylabel("Edit Score")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--", which="major")
    if handles:
        ax.legend(
            handles=handles,
            labels=labels,
            loc="lower right",
            bbox_to_anchor=(0.985, 0.11),
            fontsize=9.0,
            framealpha=0.9,
            borderpad=0.36,
            labelspacing=0.22,
            handlelength=1.45,
        )


def plot_combined_1x3(
    attention_csv: Path,
    transformer_csv: Path,
    fact_editing_csv: Path,
    output_dir: Path,
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11.2,
            "axes.titlesize": 16.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 13.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.9, 5.65))
    _plot_attention_noise_on_axis(axes[0], attention_csv)
    _plot_transformer_capacity_on_axis(axes[1], transformer_csv)
    _plot_fact_editing_score_on_axis(axes[2], fact_editing_csv)

    for ax, title in zip(axes, COMBINED_PANEL_TITLES):
        ax.set_title(title, pad=14)

    fig.subplots_adjust(left=0.058, right=0.995, bottom=0.19, top=0.82, wspace=0.24)
    paths = [
        output_dir / "fig3_transformer_integration_1x3_labeled_titled.png",
        output_dir / "fig3_transformer_integration_1x3_labeled_titled.pdf",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def generate_panels(
    attention_csv: Path,
    transformer_csv: Path,
    fact_editing_csv: Path,
    output_dir: Path,
    panels: Iterable[str],
) -> list[Path]:
    panel_set = set(panels)
    written: list[Path] = []
    if "all" in panel_set or "attention" in panel_set:
        written.extend(plot_attention_noise(attention_csv, output_dir))
    if "all" in panel_set or "transformer" in panel_set:
        written.extend(plot_transformer_capacity(transformer_csv, output_dir))
    if "all" in panel_set or "fact-editing" in panel_set:
        written.extend(plot_fact_editing_score(fact_editing_csv, output_dir))
    if "all" in panel_set or "combined" in panel_set:
        written.extend(
            plot_combined_1x3(
                attention_csv,
                transformer_csv,
                fact_editing_csv,
                output_dir,
            )
        )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-csv", type=Path, default=_default_attention_csv())
    parser.add_argument("--transformer-csv", type=Path, default=_default_transformer_csv())
    parser.add_argument("--fact-editing-csv", type=Path, default=_default_fact_editing_csv())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument(
        "--panel",
        action="append",
        choices=["all", "attention", "transformer", "fact-editing", "combined"],
        default=None,
        help="Panel to render. Repeatable. Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panels = args.panel or ["all"]
    written = generate_panels(
        args.attention_csv,
        args.transformer_csv,
        args.fact_editing_csv,
        args.output_dir,
        panels,
    )
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
