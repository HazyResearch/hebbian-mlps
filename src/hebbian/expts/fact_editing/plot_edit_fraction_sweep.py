#!/usr/bin/env python3
"""Plot fact-editing performance versus percent of facts edited."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from hebbian.expts.fact_editing.summarize_results import (
    best_result_group_columns,
    create_results_frame,
    load_results,
)


METHOD_ORDER = ["memit", "alpha_edit", "rome", "gd_construction"]
METHOD_DISPLAY = {
    "memit": "MEMIT",
    "alpha_edit": "AlphaEdit",
    "rome": "ROME",
    "gd_construction": "MLP Swapping",
}
METHOD_COLOR = {
    "memit": "#E63946",
    "alpha_edit": "#2E86AB",
    "rome": "#F18F01",
    "gd_construction": "#16A085",
}
METHOD_MARKER = {
    "memit": "o",
    "alpha_edit": "s",
    "rome": "^",
    "gd_construction": "D",
}
VARIANT_ORDER = [
    "gd",
    "hebbian",
    "hebbian_whitened",
    "cf_coord_whitened",
    "ntk",
]
VARIANT_DISPLAY = {
    "gd": "GD",
    "hebbian": "Ours (no whitening)",
    "hebbian_whitened": "Ours (whitened)",
    "cf_coord_whitened": "Ours (data-dependent)",
    "ntk": "NTK",
}
VARIANT_COLOR = {
    "gd": "#E63946",
    "hebbian": "#2E86AB",
    "hebbian_whitened": "#16A085",
    "cf_coord_whitened": "#6D28D9",
    "ntk": "#F18F01",
}
METRIC_DISPLAY = {
    "score": "Edit Score",
    "efficacy": "Efficacy",
    "paraphrase": "Paraphrase",
    "specificity": "Specificity",
    "specificity_paraphrase": "Specificity (Paraphrase)",
    "non_fact_ppl_ratio": "Non-Fact PPL Ratio",
    "non_fact_pre_ppl": "Pre-Edit Non-Fact PPL",
    "non_fact_post_ppl": "Post-Edit Non-Fact PPL",
    "non_fact_post_nll": "Post-Edit Non-Fact NLL",
}
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*"]
LINESTYLES = ["-", "--", "-.", ":"]


def setup_plot_style() -> None:
    """Mirror the Section 4 plotting style."""
    cmap = plt.get_cmap("Set1")
    plt.rcParams.setdefault("axes.prop_cycle", plt.cycler(color=cmap.colors))
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 17
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 13
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["lines.linewidth"] = 2.6
    plt.rcParams["lines.markeredgewidth"] = 1.2
    plt.rcParams["lines.markeredgecolor"] = "black"
    plt.rcParams["lines.markersize"] = math.sqrt(125)


def _short_title(title: str, metric: str) -> str:
    default_titles = {
        "score": "Fact Editing Score vs. Percent of Facts Edited",
        "non_fact_ppl_ratio": "Non-Fact PPL Ratio vs. Percent of Facts Edited",
    }
    short_titles = {
        "score": "Edit Score vs. Edited Facts (%)",
        "non_fact_ppl_ratio": "Non-Fact PPL vs. Edited Facts (%)",
    }
    if title == default_titles.get(metric):
        return short_titles.get(metric, title)
    return title


def _load_plot_frame(input_path: Path) -> pd.DataFrame:
    if input_path.is_dir():
        results = load_results(str(input_path))
        if not results:
            raise ValueError(f"No fact-editing results found under {input_path}")
        frame = create_results_frame(results)
    else:
        frame = pd.read_csv(input_path)
    if frame.empty:
        raise ValueError(f"No rows available to plot from {input_path}")

    has_counts = {"num_preserve_facts", "num_alter_facts"}.issubset(frame.columns)
    if "percent_edited" not in frame.columns or frame["percent_edited"].isna().all():
        if not has_counts:
            raise ValueError(
                "Input must include percent_edited or both num_preserve_facts "
                "and num_alter_facts."
            )
        total = frame["num_preserve_facts"] + frame["num_alter_facts"]
        frame = frame.copy()
        frame["total_tested_facts"] = total
        frame["percent_edited"] = np.where(total > 0, 100.0 * frame["num_alter_facts"] / total, np.nan)
    elif "total_tested_facts" not in frame.columns:
        frame = frame.copy()
        if has_counts:
            frame["total_tested_facts"] = frame["num_preserve_facts"] + frame["num_alter_facts"]
        else:
            frame["total_tested_facts"] = np.nan

    group_frame = frame.copy()
    if has_counts:
        group_cols = best_result_group_columns(group_frame)
    else:
        group_cols = ["method", "percent_edited"]
        for optional_col in ("base_mlp_variant", "gd_replacement_variant"):
            if optional_col in group_frame.columns:
                group_cols.append(optional_col)
    if group_frame.duplicated(group_cols).any():
        frame = group_frame.loc[group_frame.groupby(group_cols)["score"].idxmax()].copy()
    else:
        frame = group_frame
    for helper in ("_group_base_variant", "_group_gd_replacement_variant"):
        if helper in frame.columns:
            frame = frame.drop(columns=[helper])
    return frame


def _non_null_unique(frame: pd.DataFrame, column: str) -> List[str]:
    if column not in frame.columns:
        return []
    values = []
    for value in frame[column].tolist():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        values.append(text)
    return sorted(set(values))


def _format_percent(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 0.15:
        return f"{rounded:.0f}%"
    return f"{value:.1f}%"


def _series_metadata(
    frame: pd.DataFrame,
    *,
    legend_style: str = "variant",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    frame = frame.copy()
    base_variants = _non_null_unique(frame, "base_mlp_variant")
    include_base_variant = len(base_variants) > 1

    series_rows: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        method = str(row["method"])
        base_label = METHOD_DISPLAY.get(method, method)
        base_variant = row.get("base_mlp_variant")
        replacement_variant = row.get("gd_replacement_variant")
        if pd.isna(replacement_variant):
            replacement_variant = row.get("gd_replacement_variant_label")
        if pd.isna(base_variant):
            base_variant = None
        if pd.isna(replacement_variant):
            replacement_variant = None

        use_variant_label = (
            legend_style == "variant"
            and method == "gd_construction"
            and replacement_variant
        )
        if use_variant_label:
            replacement_variant = str(replacement_variant)
            series_label = f"{base_label} ({VARIANT_DISPLAY.get(replacement_variant, replacement_variant)})"
            color = VARIANT_COLOR.get(replacement_variant, METHOD_COLOR["gd_construction"])
            variant_rank = (
                VARIANT_ORDER.index(replacement_variant)
                if replacement_variant in VARIANT_ORDER
                else len(VARIANT_ORDER)
            )
        else:
            if replacement_variant:
                replacement_variant = str(replacement_variant)
            else:
                replacement_variant = None
            series_label = base_label
            color = METHOD_COLOR.get(method, "#666666")
            variant_rank = (
                VARIANT_ORDER.index(replacement_variant)
                if replacement_variant in VARIANT_ORDER
                else -1
            )

        if include_base_variant and base_variant:
            series_label = f"{series_label} | base={base_variant}"

        method_rank = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
        base_rank = base_variants.index(str(base_variant)) if include_base_variant and base_variant in base_variants else -1
        series_key = "|".join(
            [
                method,
                str(base_variant) if base_variant is not None else "",
                str(replacement_variant) if replacement_variant is not None else "",
            ]
        )
        series_rows.append(
            {
                "series_key": series_key,
                "series_label": series_label,
                "series_rank": (method_rank, variant_rank, base_rank, series_label),
                "series_color": color,
                "base_variant": base_variant,
                "replacement_variant": replacement_variant,
            }
        )

    meta_frame = pd.DataFrame(series_rows).drop_duplicates(subset=["series_key"]).sort_values("series_rank")
    meta: Dict[str, Dict[str, Any]] = {}
    for index, (_, row) in enumerate(meta_frame.iterrows()):
        method = str(row["series_key"]).split("|", 1)[0]
        meta[str(row["series_key"])] = {
            "label": row["series_label"],
            "color": row["series_color"],
            "marker": METHOD_MARKER.get(method, MARKERS[index % len(MARKERS)]),
            "linestyle": LINESTYLES[(index // len(MARKERS)) % len(LINESTYLES)],
        }

    frame["series_key"] = [
        "|".join(
            [
                str(row["method"]),
                "" if pd.isna(row.get("base_mlp_variant")) else str(row.get("base_mlp_variant")),
                ""
                if (
                    pd.isna(row.get("gd_replacement_variant"))
                    and pd.isna(row.get("gd_replacement_variant_label"))
                )
                else str(
                    row.get("gd_replacement_variant")
                    if not pd.isna(row.get("gd_replacement_variant"))
                    else row.get("gd_replacement_variant_label")
                ),
            ]
        )
        for _, row in frame.iterrows()
    ]
    return frame, meta


def _write_points_csv(frame: pd.DataFrame, meta: Dict[str, Dict[str, Any]], output_csv: Path, metric: str) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "series_key",
        "series_label",
        "method",
        "base_mlp_variant",
        "gd_replacement_variant",
        "num_preserve_facts",
        "num_alter_facts",
        "total_tested_facts",
        "percent_edited",
        metric,
        "score",
        "efficacy",
        "paraphrase",
        "specificity",
        "specificity_paraphrase",
        "non_fact_ppl_ratio",
        "non_fact_pre_ppl",
        "non_fact_post_ppl",
        "non_fact_post_nll",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _, row in frame.sort_values(["series_key", "percent_edited"]).iterrows():
            writer.writerow(
                {
                    "series_key": row["series_key"],
                    "series_label": meta[row["series_key"]]["label"],
                    "method": row.get("method"),
                    "base_mlp_variant": row.get("base_mlp_variant"),
                    "gd_replacement_variant": row.get("gd_replacement_variant"),
                    "num_preserve_facts": row.get("num_preserve_facts"),
                    "num_alter_facts": row.get("num_alter_facts"),
                    "total_tested_facts": row.get("total_tested_facts"),
                    "percent_edited": row.get("percent_edited"),
                    metric: row.get(metric),
                    "score": row.get("score"),
                    "efficacy": row.get("efficacy"),
                    "paraphrase": row.get("paraphrase"),
                    "specificity": row.get("specificity"),
                    "specificity_paraphrase": row.get("specificity_paraphrase"),
                    "non_fact_ppl_ratio": row.get("non_fact_ppl_ratio"),
                    "non_fact_pre_ppl": row.get("non_fact_pre_ppl"),
                    "non_fact_post_ppl": row.get("non_fact_post_ppl"),
                    "non_fact_post_nll": row.get("non_fact_post_nll"),
                }
            )


def plot_metric_vs_percent(
    frame: pd.DataFrame,
    meta: Dict[str, Dict[str, Any]],
    output_path: Path,
    *,
    metric: str,
    title: str,
    show: bool,
) -> None:
    if metric not in frame.columns:
        raise ValueError(f"Metric {metric!r} not found in input data.")

    valid = frame[np.isfinite(frame["percent_edited"]) & np.isfinite(frame[metric])].copy()
    if valid.empty:
        raise ValueError("No valid rows with finite percent_edited and metric values to plot.")

    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.2))

    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []
    ordered_keys = list(meta.keys())
    for series_key in ordered_keys:
        series = valid[valid["series_key"] == series_key].sort_values("percent_edited")
        if series.empty:
            continue
        series_meta = meta[series_key]
        x = np.asarray(series["percent_edited"], dtype=np.float64)
        y = np.asarray(series[metric], dtype=np.float64)
        ax.plot(
            x,
            y,
            color=series_meta["color"],
            linestyle=series_meta["linestyle"],
            marker=series_meta["marker"],
            linewidth=2.6,
            markersize=math.sqrt(125),
            markeredgewidth=1.2,
            markeredgecolor="black",
            alpha=0.92,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=series_meta["color"],
                linestyle=series_meta["linestyle"],
                marker=series_meta["marker"],
                markeredgecolor="black",
                markeredgewidth=1.2,
                markersize=9.5,
            )
        )
        legend_labels.append(series_meta["label"])

    x_values = sorted(valid["percent_edited"].unique())
    ax.set_xticks(x_values)
    ax.set_xticklabels([_format_percent(value) for value in x_values])
    ax.set_xlabel("Edited Facts (%)")
    ax.set_ylabel(METRIC_DISPLAY.get(metric, metric))
    if metric in {"score", "efficacy", "paraphrase", "specificity", "specificity_paraphrase"}:
        ax.set_ylim(-0.02, 1.05)
    ax.set_title(_short_title(title, metric))
    ax.grid(True, alpha=0.3, linestyle="--", which="major")

    if legend_handles:
        legend_kwargs = {
            "handles": legend_handles,
            "labels": legend_labels,
            "fontsize": 11,
            "framealpha": 0.9,
            "borderpad": 0.5,
            "labelspacing": 0.35,
            "handlelength": 1.7,
        }
        if len(legend_labels) <= 4:
            if metric in {"score", "efficacy", "paraphrase", "specificity", "specificity_paraphrase"}:
                legend_loc = "lower right"
                ax.legend(loc=legend_loc, bbox_to_anchor=(0.985, 0.11), **legend_kwargs)
            else:
                legend_loc = "upper left"
                ax.legend(loc=legend_loc, **legend_kwargs)
        else:
            fig.legend(
                loc="upper right",
                bbox_to_anchor=(0.995, 0.98),
                borderaxespad=0.0,
                **legend_kwargs,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(legend_labels) > 4:
        fig.subplots_adjust(right=0.62)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot fact-editing metric versus percent of facts edited."
    )
    parser.add_argument(
        "input_path",
        help="Results directory or best_results.csv path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the plot and points CSV.",
    )
    parser.add_argument(
        "--metric",
        default="score",
        choices=sorted(METRIC_DISPLAY.keys()),
        help="Metric to plot on the y-axis.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output PNG filename (default: <metric>_vs_percent_edited.png).",
    )
    parser.add_argument(
        "--points-csv",
        default=None,
        help="Optional points CSV path (default: <output_dir>/figure_points.csv).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title override.",
    )
    parser.add_argument(
        "--legend-style",
        choices=["method", "variant"],
        default="variant",
        help=(
            "Legend labeling/color mode. 'method' uses concise paper labels; "
            "'variant' expands gd_construction replacement variants."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively instead of closing it after save.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    frame = _load_plot_frame(input_path)
    frame, meta = _series_metadata(frame, legend_style=args.legend_style)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif input_path.is_dir():
        output_dir = input_path / "plots"
    else:
        output_dir = input_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = args.output_name or f"{args.metric}_vs_percent_edited.png"
    output_path = output_dir / output_name
    points_csv = (
        Path(args.points_csv).expanduser().resolve()
        if args.points_csv
        else output_dir / "figure_points.csv"
    )
    title = args.title or f"{METRIC_DISPLAY.get(args.metric, args.metric)} vs. Percent of Facts Edited"

    _write_points_csv(frame, meta, points_csv, args.metric)
    plot_metric_vs_percent(
        frame,
        meta,
        output_path,
        metric=args.metric,
        title=title,
        show=args.show,
    )
    print(f"Saved plot to {output_path}")
    print(f"Saved figure points to {points_csv}")


if __name__ == "__main__":
    main()
