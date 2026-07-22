"""Pretty single-panel plot for Section 3.1 attention-noise sweeps.

This script focuses on:
- y-axis: attention noise floor (L2)
- x-axis: junk length
- optional junk-length subset filtering (default: 2,4,8,16)

Designed for cleaner presentation with larger text and stronger line styling.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from hebbian.expts.attention_noise.plot_sweep import load_rows


def load_points_csv(path: Path) -> List[Dict]:
    """Load the compact, release-facing attention-noise plot record."""
    rows: List[Dict] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "d_model": int(row["d_model"]),
                    "num_facts": int(row["num_facts"]),
                    "junk_len": int(row["junk_len"]),
                    "best_acc": float(row["best_acc"]),
                    "attn_noise_l2_floor": float(row["attn_noise_l2_floor"]),
                    "attn_noise_l2_mean": float(row["attn_noise_l2_mean"]),
                }
            )
    return rows


def write_points_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "d_model",
        "num_facts",
        "junk_len",
        "best_acc",
        "attn_noise_l2_floor",
        "attn_noise_l2_mean",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


_DEFAULT_JUNK_LENS = [2, 4, 8, 16]

_COLOR_BY_D = {
    64: "#1f77b4",   # blue
    96: "#2ca02c",   # green
    128: "#d62728",  # red
}
_MARKER_BY_D = {
    64: "o",
    96: "s",
    128: "D",
}


def _parse_junk_lens(raw: str) -> List[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    return sorted(set(vals))


def _filter_rows(rows: List[Dict], junk_lens: List[int]) -> List[Dict]:
    keep = set(int(j) for j in junk_lens)
    out: List[Dict] = []
    for r in rows:
        j = int(r["junk_len"])
        if j in keep and np.isfinite(float(r["attn_noise_l2_floor"])):
            out.append(r)
    return out


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


def _pow2_label(value: int) -> str:
    if _is_power_of_two(value):
        exp = int(np.log2(value))
        return rf"$2^{{{exp}}}$"
    return str(value)


def plot_pretty_noise_floor(
    rows: List[Dict],
    output_dir: Path,
    output_stem: str,
    show: bool,
) -> None:
    if not rows:
        raise ValueError("No rows available after filtering.")

    output_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 6.3))

    cfgs = sorted({(int(r["d_model"]), int(r["num_facts"])) for r in rows})
    for d_model, num_facts in cfgs:
        sub = [r for r in rows if int(r["d_model"]) == d_model and int(r["num_facts"]) == num_facts]
        sub = sorted(sub, key=lambda r: int(r["junk_len"]))
        if not sub:
            continue

        x = np.asarray([int(r["junk_len"]) for r in sub], dtype=np.float64)
        y = np.asarray([float(r["attn_noise_l2_floor"]) for r in sub], dtype=np.float64)

        color = _COLOR_BY_D.get(d_model, "#444444")
        marker = _MARKER_BY_D.get(d_model, "o")

        ax.plot(
            x,
            y,
            color=color,
            linestyle="-",
            marker=marker,
            markeredgecolor="black",
            markeredgewidth=1.0,
            label=f"d={d_model}, F={num_facts}",
            alpha=0.92,
        )

    x_vals = sorted({int(r["junk_len"]) for r in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([_pow2_label(v) for v in x_vals])
    ax.set_xlabel(r"Junk Length $J$")
    ax.set_ylabel("Attention Noise Floor (L2)")
    ax.set_title(r"Attention-Only: $\|\varepsilon_{\mathrm{attn}}\|_2$ vs. Junk Length")
    ax.grid(True, which="major", alpha=0.3, linestyle="--")

    ax.legend(
        loc="upper left",
        fontsize=12,
        title="Model",
        title_fontsize=12,
        frameon=True,
        framealpha=0.92,
    )

    fig.tight_layout()

    out_png = output_dir / f"{output_stem}.png"
    out_pdf = output_dir / f"{output_stem}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty attention-noise floor vs junk-length plot.")
    parser.add_argument("base_dir", type=str, help="Sweep directory to read (e.g., attention_only/coupled_len_vocab).")
    parser.add_argument(
        "--input-points-csv",
        type=str,
        default=None,
        help="Read compact plot points from CSV instead of sweep pickles.",
    )
    parser.add_argument(
        "--junk-lens",
        type=str,
        default="2,4,8,16",
        help="Comma-separated junk lengths to include (default: 2,4,8,16).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <base_dir>/plots_pretty).",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="section3_1_attention_only_noise_floor_pretty",
        help="Filename stem for output files.",
    )
    parser.add_argument(
        "--points-csv",
        type=str,
        default=None,
        help="Write the filtered plot points to CSV (default: <output-dir>/figure_points.csv).",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plot window.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    junk_lens = _parse_junk_lens(args.junk_lens)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else base_dir / "plots_pretty"

    if args.input_points_csv:
        points_csv = Path(args.input_points_csv).resolve()
        rows = load_points_csv(points_csv)
        source_description = str(points_csv)
    else:
        rows = load_rows(base_dir)
        source_description = str(base_dir)
    rows = _filter_rows(rows, junk_lens=junk_lens)
    print(f"Loaded {len(rows)} filtered cells from: {source_description}")
    points_csv = Path(args.points_csv).resolve() if args.points_csv else output_dir / "figure_points.csv"
    write_points_csv(rows, points_csv)
    print(f"Saved: {points_csv}")

    plot_pretty_noise_floor(
        rows=rows,
        output_dir=output_dir,
        output_stem=args.output_stem,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
