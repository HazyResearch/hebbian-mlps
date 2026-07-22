"""
Plot hidden-dim sweep results with accuracy and gamma_min on dual y-axes.

This is the checkpoint-based hidden-dimension dual-axis plotter.

For plot-only reproduction it can also read a previously generated
``figure_points.csv`` via the ``points_csv`` option.

It reads checkpoints from:
  {base_dir}/m{hidden_dim}/seed_{seed}/checkpoints/last_model.pt

Usage:
    python -m hebbian.expts.hidden_dim.plot_dual_axis \
        'base_dir="./artifacts/hidden_dim/run_..."'
"""

from __future__ import annotations

import glob
import csv
import math
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from hebbian.config import main as pydra_main, pydraclass

from hebbian.expts.hidden_dim.plot_hidden_dim_sweep import (
    _aggregate,
    _load_sweep_results,
)

# Final "03_31" restyle palette (blue accuracy / red accuracy / orange margin),
# matching the paper hidden-dim usability figure.
_COLOR_MLP = "#2E86AB"
_COLOR_TFM = "#E63946"
_COLOR_GAMMA = "#F18F01"


def setup_plot_style() -> None:
    """Set plot styling to match the MLP capacity sweep plots."""
    cmap = plt.get_cmap("Set1")
    plt.rcParams.setdefault("axes.prop_cycle", plt.cycler(color=cmap.colors))
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markeredgewidth"] = 1.0
    plt.rcParams["lines.markeredgecolor"] = "black"
    plt.rcParams["lines.markersize"] = math.sqrt(80)


@pydraclass
class PlotConfig:
    """Configuration for dual-axis hidden-dim sweep plotting."""

    base_dir: Optional[str] = None
    search_root: str = "./artifacts/hidden_dim"
    output_dir: Optional[str] = None
    title_suffix: str = ""
    stem: str = "hidden_dim_sweep_dual_axis"
    points_csv: Optional[str] = None
    points_output_csv: Optional[str] = None


def _find_latest_sweep_dir(search_root: str) -> str:
    patterns = [
        os.path.join(search_root, "run_*"),
    ]
    dirs = sorted((d for p in patterns for d in glob.glob(p)), key=os.path.getmtime)
    if not dirs:
        raise FileNotFoundError(
            "No hidden-dim sweep directories found under: "
            f"{search_root}"
        )
    return dirs[-1]


def _normalize_base_dir(base_dir: str) -> str:
    base_dir = os.path.normpath(base_dir)
    if os.path.isfile(base_dir):
        # Normalize checkpoint-file inputs like
        # .../m{hidden_dim}/seed_{seed}/checkpoints/last_model.pt
        # to the checkpoint root (.../m{hidden_dim}).
        for _ in range(4):
            base_dir = os.path.dirname(base_dir)
    return base_dir


def _resolve_checkpoint_root(base_dir: str) -> str:
    pattern = os.path.join(base_dir, "m*", "seed_*", "checkpoints", "last_model.pt")
    if glob.glob(pattern):
        return base_dir
    raise FileNotFoundError(f"No checkpoints found under: {base_dir}")


def _load_points_csv(points_csv: str) -> list[dict]:
    """Load compact paper-source hidden-dim rows into the aggregate schema."""
    rows = []
    with open(points_csv, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            hidden_dim = int(float(raw["hidden_dim"]))
            param_count = int(float(raw["param_count"]))
            mlp_accuracy = float(raw["mlp_accuracy"])
            transformer_accuracy = float(
                raw.get("transformer_accuracy", raw.get("accuracy", "nan"))
            )
            gamma_min = float(raw["gamma_min"])
            transformer_train_accuracy = float(
                raw.get("attn_best_acc", raw.get("transformer_train_accuracy", "nan"))
            )
            rows.append(
                {
                    "hidden_dim": hidden_dim,
                    "param_count": param_count,
                    "mlp_accuracy_mean": mlp_accuracy,
                    "mlp_accuracy_std": 0.0,
                    "mlp_accuracy_max": mlp_accuracy,
                    "transformer_accuracy_mean": transformer_accuracy,
                    "transformer_accuracy_std": 0.0,
                    "transformer_accuracy_max": transformer_accuracy,
                    "transformer_train_accuracy_mean": transformer_train_accuracy,
                    "transformer_train_accuracy_std": 0.0,
                    "transformer_train_accuracy_max": transformer_train_accuracy,
                    "gamma_min_mean": gamma_min,
                    "gamma_min_std": 0.0,
                    "n_seeds": 1,
                    "mlp_str": None,
                    "gpt_str": None,
                }
            )
    rows.sort(key=lambda r: r["param_count"])
    if not rows:
        raise ValueError(f"No hidden-dim points found in CSV: {points_csv}")
    return rows


def _write_points_csv(rows: list[dict], path: str) -> None:
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "hidden_dim",
        "param_count",
        "mlp_accuracy",
        "transformer_accuracy",
        "transformer_train_accuracy",
        "gamma_min",
        "n_seeds",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "hidden_dim": row["hidden_dim"],
                    "param_count": row["param_count"],
                    "mlp_accuracy": row["mlp_accuracy_mean"],
                    "transformer_accuracy": row["transformer_accuracy_mean"],
                    "transformer_train_accuracy": row["transformer_train_accuracy_mean"],
                    "gamma_min": row["gamma_min_mean"],
                    "n_seeds": row["n_seeds"],
                }
            )
    print(f"Saved: {output_path}")


def _plot(rows: list[dict], output_dir: str, stem: str, title_suffix: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)

    x = np.array(
        [
            r["param_count"] if r["param_count"] is not None else r["hidden_dim"]
            for r in rows
        ],
        dtype=float,
    )
    mlp_acc_mean = np.array([r["mlp_accuracy_mean"] for r in rows])
    mlp_acc_std = np.array([r["mlp_accuracy_std"] for r in rows])
    tfm_acc_mean = np.array([r["transformer_accuracy_mean"] for r in rows])
    tfm_acc_std = np.array([r["transformer_accuracy_std"] for r in rows])
    gmin_mean = np.array([r["gamma_min_mean"] for r in rows])
    gmin_std = np.array([r["gamma_min_std"] for r in rows])

    mlp_acc_max = np.array([r["mlp_accuracy_max"] for r in rows])
    tfm_acc_max = np.array([r["transformer_accuracy_max"] for r in rows])
    mlp_thresh_x = next((xi for xi, acc in zip(x, mlp_acc_max) if acc >= 1.0), None)
    tfm_thresh_x = next((xi for xi, acc in zip(x, tfm_acc_max) if acc >= 1.0), None)

    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=(7.6, 6.3))
    ax2 = ax1.twinx()

    ms = math.sqrt(80)

    l1, = ax1.plot(
        x,
        mlp_acc_mean,
        "o-",
        color=_COLOR_MLP,
        markersize=ms,
        markeredgewidth=1.0,
        markeredgecolor="black",
        alpha=0.92,
        label="MLP accuracy",
    )
    ax1.fill_between(
        x,
        mlp_acc_mean - mlp_acc_std,
        mlp_acc_mean + mlp_acc_std,
        color=_COLOR_MLP,
        alpha=0.2,
    )

    l2, = ax1.plot(
        x,
        tfm_acc_mean,
        "s-",
        color=_COLOR_TFM,
        markersize=ms,
        markeredgewidth=1.0,
        markeredgecolor="black",
        alpha=0.92,
        label="Transformer accuracy",
    )
    ax1.fill_between(
        x,
        tfm_acc_mean - tfm_acc_std,
        tfm_acc_mean + tfm_acc_std,
        color=_COLOR_TFM,
        alpha=0.2,
    )

    l3, = ax2.plot(
        x,
        gmin_mean,
        "^--",
        color=_COLOR_GAMMA,
        markersize=ms,
        markeredgewidth=1.0,
        markeredgecolor="black",
        alpha=0.85,
        label="MLP margin",
    )
    ax2.fill_between(
        x,
        gmin_mean - gmin_std,
        gmin_mean + gmin_std,
        color=_COLOR_GAMMA,
        alpha=0.15,
    )

    vlines = []
    if mlp_thresh_x is not None:
        vlines.append(
            ax1.axvline(
                mlp_thresh_x,
                color=_COLOR_MLP,
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
            )
        )
        ax1.annotate(
            f"{int(mlp_thresh_x):,} params",
            xy=(mlp_thresh_x, 0.12),
            xycoords=ax1.get_xaxis_transform(),
            xytext=(6, 0),
            textcoords="offset points",
            color=_COLOR_MLP,
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="bottom",
            rotation=90,
            rotation_mode="anchor",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
    if tfm_thresh_x is not None:
        vlines.append(
            ax1.axvline(
                tfm_thresh_x,
                color=_COLOR_TFM,
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
            )
        )
        ax1.annotate(
            f"{int(tfm_thresh_x):,} params",
            xy=(tfm_thresh_x, 0.34),
            xycoords=ax1.get_xaxis_transform(),
            xytext=(6, 0),
            textcoords="offset points",
            color=_COLOR_TFM,
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="bottom",
            rotation=90,
            rotation_mode="anchor",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("# MLP parameters")
    ax1.set_ylabel("Accuracy", color="black")
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, alpha=0.3, linestyle="--", which="major")

    ax2.set_ylabel(r"$\gamma_{\min}$ (normalized margin)", color=_COLOR_GAMMA)
    ax2.tick_params(axis="y", labelcolor=_COLOR_GAMMA)

    if title_suffix:
        ax1.set_title(title_suffix)

    lines = [l1, l3, l2]
    ax1.legend(
        lines,
        [line.get_label() for line in lines],
        fontsize=9,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()
    for ext in (".pdf", ".png"):
        out_path = os.path.join(output_dir, stem + ext)
        kw = {"bbox_inches": "tight"}
        if ext == ".png":
            kw["dpi"] = 300
        plt.savefig(out_path, **kw)
        print(f"Saved: {out_path}")
    plt.close()

    print(f"\n{'hidden_dim':>12}  {'params':>10}  {'mlp_acc':>9}  {'tfm_eval':>9}  {'gamma_min':>10}")
    print("-" * 60)
    for row in rows:
        param_count = row["param_count"] if row["param_count"] is not None else "?"
        print(
            f"{row['hidden_dim']:>12}  {str(param_count):>10}  "
            f"{row['mlp_accuracy_mean']:>9.4f}  "
            f"{row['transformer_accuracy_mean']:>9.4f}  "
            f"{row['gamma_min_mean']:>10.4f}"
        )


@pydra_main(PlotConfig)
def main(config: PlotConfig):
    if config.points_csv:
        points_csv = os.path.abspath(config.points_csv)
        output_dir = config.output_dir or os.path.dirname(points_csv)
        print(f"Reading compact hidden-dim points from: {points_csv}")
        rows = _load_points_csv(points_csv)
    else:
        if config.base_dir is None:
            config.base_dir = _find_latest_sweep_dir(config.search_root)

        base_dir = _normalize_base_dir(config.base_dir)
        checkpoint_root = _resolve_checkpoint_root(base_dir)
        output_dir = config.output_dir or os.path.join(base_dir, "plots")

        print(f"Reading sweep from: {checkpoint_root}")
        by_dim = _load_sweep_results(checkpoint_root)
        rows = _aggregate(by_dim)
    if not rows:
        print("No valid results found — nothing to plot.")
        return

    points_output_csv = config.points_output_csv or os.path.join(output_dir, "figure_points.csv")
    _write_points_csv(rows, points_output_csv)
    _plot(rows, output_dir=output_dir, stem=config.stem, title_suffix=config.title_suffix)
    print(f"\nDone. Dual-axis plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
