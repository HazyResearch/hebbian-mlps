"""Canonical paper plot for margin/theory validation sweeps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from hebbian.expts.margins.theory import (
        compute_fitted_bound,
        CASE_EQUATIONS,
        CASE_TITLES,
        detect_case,
        extract_quantities,
        load_results,
        r2_mse,
    )
except ModuleNotFoundError:
    import importlib.util

    sibling = Path(__file__).with_name("theory.py")
    spec = importlib.util.spec_from_file_location("_margin_theory", sibling)
    if spec is None or spec.loader is None:
        raise
    theory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theory)
    compute_fitted_bound = theory.compute_fitted_bound
    CASE_EQUATIONS = theory.CASE_EQUATIONS
    CASE_TITLES = theory.CASE_TITLES
    detect_case = theory.detect_case
    extract_quantities = theory.extract_quantities
    load_results = theory.load_results
    r2_mse = theory.r2_mse


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 10.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
            "lines.markeredgewidth": 1.0,
            "lines.markeredgecolor": "black",
            "savefig.dpi": 300,
        }
    )


def _subtitle(case: str, r2: float, mse: float, extra: str | None = None) -> str:
    equation = CASE_EQUATIONS.get(case, "")
    stats = rf"$R^2={r2:.3f}$, MSE={mse:.2e}"
    pieces = [equation, stats]
    if extra:
        pieces.insert(0, extra)
    return "\n".join(piece for piece in pieces if piece)


def _inset_text(title: str | None, subtitle: str | None) -> str:
    pieces = []
    if title:
        pieces.extend(line for line in title.splitlines() if line.strip())
    if subtitle:
        pieces.extend(line for line in subtitle.splitlines() if line.strip())
    return "\n".join(pieces)


def _legend_location(case: str, sweep_type: str) -> str:
    if sweep_type == "F" or case in {"rkav", "akav"}:
        return "lower left"
    return "upper right"


def plot_and_save(
    results: dict,
    output_dir: str,
    case: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    case_label: str | None = None,
) -> str:
    if case is None:
        case = detect_case(results)

    q = extract_quantities(results)
    gamma = q["gamma"]
    gamma_std = q["gamma_std"]
    x = q["x"]
    xlabel = q["xlabel"]

    fitted, _ = compute_fitted_bound(case, q)
    r2, mse = r2_mse(gamma, fitted)

    sweep_type = q["sweep_type"]
    log_x = sweep_type in ("F", "M") and np.all(x > 0)
    title_text = title
    subtitle_text = subtitle if subtitle is not None else _subtitle(
        case=case,
        r2=r2,
        mse=mse,
        extra=case_label if case_label is not None else CASE_TITLES.get(case, case),
    )

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.3))

    ax.plot(
        x,
        gamma,
        "o-",
        color="#202020",
        markersize=math.sqrt(70),
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
        markersize=math.sqrt(70),
        label="Fitted theory",
    )

    ax.axhline(0, color="0.55", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.grid(True, alpha=0.25, linestyle="--", which="major")
    if log_x:
        ax.set_xscale("log")
    ax.margins(y=0.18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Minimum margin $\gamma_{\min}$")
    ax.legend(loc=_legend_location(case, sweep_type), framealpha=0.92)

    info_text = _inset_text(title_text, subtitle_text)
    ax.text(
        0.5,
        0.965,
        info_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.8,
        linespacing=1.14,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
        },
        zorder=20,
    )
    fig.tight_layout(pad=0.35)

    stem = Path(results.get("_source_path", "results")).stem
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    png_path = out_path / f"{stem}_{case}_{sweep_type}.png"
    pdf_path = out_path / f"{stem}_{case}_{sweep_type}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    return str(png_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a paper margin/theory validation sweep."
    )
    parser.add_argument("results", help="Path to JSON or pickle results file")
    parser.add_argument("--output", default="./plots", help="Output directory")
    parser.add_argument(
        "--case",
        default=None,
        choices=["rkrv", "akrv", "rkav", "akav"],
        help="Force a 2x2 case; auto-detected if omitted.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional main title. Literal '\\n' sequences are rendered as line breaks.",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        help="Optional subtitle. Literal '\\n' sequences are rendered as line breaks.",
    )
    parser.add_argument(
        "--case-label",
        default=None,
        help=(
            "Optional case/setup line inserted before the fitted bound. "
            "Literal '\\n' sequences are rendered as line breaks."
        ),
    )
    args = parser.parse_args()

    results = load_results(args.results)
    results["_source_path"] = args.results

    title = args.title.replace("\\n", "\n") if args.title is not None else None
    subtitle = (
        args.subtitle.replace("\\n", "\n") if args.subtitle is not None else None
    )
    case_label = (
        args.case_label.replace("\\n", "\n") if args.case_label is not None else None
    )
    plot_and_save(
        results,
        args.output,
        case=args.case,
        title=title,
        subtitle=subtitle,
        case_label=case_label,
    )
    print("Done.")


if __name__ == "__main__":
    main()
