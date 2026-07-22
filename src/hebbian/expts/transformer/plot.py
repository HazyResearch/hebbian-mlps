"""Generic plotter for transformer-capacity summary CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from hebbian.expts.transformer.sweep import METHOD_DISPLAY, METHOD_ORDER


METHOD_COLOR = {
    "gd": "#E63946",
    "hebbian": "#2E86AB",
    "hebbian_whitened": "#16A085",
    "cf_coord_whitened": "#6D28D9",
    "ntk": "#F18F01",
    "unknown": "#666666",
}
D_MODEL_MARKER = {32: "o", 64: "o", 90: "s", 128: "^"}
JUNK_LINESTYLES = ["-", "--", "-.", ":"]
PAPER_METHOD_LINESTYLE = {
    "hebbian": ":",
    "hebbian_whitened": ":",
    "ntk": ":",
}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_rows(csv_path: Path | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row.get("method") or "").strip() or "unknown"
            capacity_num_facts = _to_float(row.get("capacity_num_facts"))
            capacity_hidden_dim = _to_float(row.get("capacity_hidden_dim"))
            mlp_param_count = _to_float(row.get("mlp_param_count"))
            rows.append(
                {
                    "orientation": row.get("orientation"),
                    "method": method,
                    "method_display": METHOD_DISPLAY.get(method, method),
                    "d_model": _to_float(row.get("d_model")),
                    "junk_len": _to_float(row.get("junk_len")),
                    "capacity_num_facts": capacity_num_facts,
                    "capacity_hidden_dim": capacity_hidden_dim,
                    "mlp_param_count": mlp_param_count,
                }
            )
    rows = [
        row
        for row in rows
        if np.isfinite(row["capacity_num_facts"]) and np.isfinite(row["mlp_param_count"])
    ]
    method_rank = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    rows.sort(
        key=lambda row: (
            method_rank.get(row["method"], len(METHOD_ORDER)),
            row["d_model"],
            row["mlp_param_count"],
        )
    )
    return rows


def plot_capacity_csv(
    csv_path: Path | str,
    output_dir: Path | str,
    *,
    show: bool = False,
    title: str = "Transformer Capacity Scaling",
    paper_style: bool = False,
    y_label: str | None = None,
) -> list[Path]:
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError("No valid rows found in capacity summary CSV.")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods_present = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]
    d_values = sorted({int(row["d_model"]) for row in rows if np.isfinite(row["d_model"])})
    junk_values = sorted({int(row["junk_len"]) for row in rows if np.isfinite(row["junk_len"])})
    junk_to_style = {junk: JUNK_LINESTYLES[idx % len(JUNK_LINESTYLES)] for idx, junk in enumerate(junk_values)}

    fig_size = (7.2, 6.0) if paper_style else (6.5, 5.0)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    for method in methods_present:
        method_rows = [row for row in rows if row["method"] == method]
        for d_model in d_values:
            for junk_len in junk_values or [0]:
                subset = [
                    row
                    for row in method_rows
                    if int(row["d_model"]) == d_model
                    and (not junk_values or int(row["junk_len"]) == junk_len)
                ]
                if not subset:
                    continue
                subset = sorted(subset, key=lambda row: row["mlp_param_count"])
                x = np.asarray([row["mlp_param_count"] for row in subset], dtype=np.float64)
                y = np.asarray([row["capacity_num_facts"] for row in subset], dtype=np.float64)
                ax.plot(
                    x,
                    y,
                    marker="o" if paper_style else D_MODEL_MARKER.get(d_model, "o"),
                    linestyle=(
                        PAPER_METHOD_LINESTYLE.get(method, "-")
                        if paper_style
                        else junk_to_style.get(junk_len, "-")
                    ),
                    color=METHOD_COLOR.get(method, "#666666"),
                    linewidth=2.2 if paper_style else 2.0,
                    markersize=8 if paper_style else 6.5,
                    markeredgewidth=1.0,
                    markeredgecolor="black",
                    alpha=0.95 if paper_style else 0.92,
                )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    if paper_style:
        ax.set_xlabel("Number of MLP Parameters")
        ax.set_ylabel(y_label or "Number of Facts")
        if title:
            ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    else:
        ax.set_xlabel("Number of Parameters (W)", fontweight="bold")
        ax.set_ylabel("Number of Facts (F)", fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)

    method_handles = [
        plt.Line2D(
            [0],
            [0],
            color=METHOD_COLOR.get(method, "#666666"),
            linestyle=PAPER_METHOD_LINESTYLE.get(method, "-") if paper_style else "-",
            marker="o",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=8 if paper_style else 7,
        )
        for method in methods_present
    ]
    if method_handles:
        method_legend = ax.legend(
            handles=method_handles,
            labels=[METHOD_DISPLAY.get(method, method) for method in methods_present],
            title="Method",
            loc="upper left" if paper_style else "lower right",
            bbox_to_anchor=None if paper_style else (0.98, 0.02),
            fontsize=9,
            title_fontsize=10 if not paper_style else None,
            framealpha=0.9,
        )
        ax.add_artist(method_legend)

    if d_values and not paper_style:
        d_handles = [
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                marker=D_MODEL_MARKER.get(d_model, "o"),
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=7,
            )
            for d_model in d_values
        ]
        d_legend = ax.legend(
            handles=d_handles,
            labels=[f"d={d_model}" for d_model in d_values],
            loc="upper left",
            bbox_to_anchor=(0.02, 0.71),
            fontsize=9,
            framealpha=0.9,
        )
        ax.add_artist(d_legend)

    if len(junk_values) > 1:
        junk_handles = [
            plt.Line2D([0], [0], color="black", linestyle=junk_to_style[junk], linewidth=2.0)
            for junk in junk_values
        ]
        ax.legend(
            handles=junk_handles,
            labels=[f"J={junk}" for junk in junk_values],
            title="Junk Len",
            loc="upper left",
            bbox_to_anchor=(0.02, 0.50),
            fontsize=9,
            title_fontsize=10,
            framealpha=0.9,
        )

    plt.tight_layout()
    output_paths = [
        output_dir / "transformer_capacity_scaling.png",
        output_dir / "transformer_capacity_scaling.pdf",
    ]
    fig.savefig(output_paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(output_paths[1], bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot transformer-capacity summary CSV.")
    parser.add_argument("csv_path", help="Path to capacity_points.csv.")
    parser.add_argument("--output-dir", default=None, help="Optional plot output directory.")
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title. Literal '\\n' sequences render as line breaks.",
    )
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="Use the fixed-d paper capacity panel style: no d legend and paper axis labels.",
    )
    parser.add_argument("--y-label", default=None, help="Optional y-axis label override.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    csv_path = Path(args.csv_path).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else csv_path.parent / "plots"
    )
    default_title = "" if args.paper_style else "Transformer Capacity Scaling"
    title = (args.title if args.title is not None else default_title).replace("\\n", "\n")
    for path in plot_capacity_csv(
        csv_path,
        output_dir,
        show=args.show,
        title=title,
        paper_style=args.paper_style,
        y_label=args.y_label,
    ):
        print(path)


if __name__ == "__main__":
    main()
