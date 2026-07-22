"""Fact-editing paper targets."""

from __future__ import annotations

from pathlib import Path

from common import PaperContext


def _best_results_csv(context: PaperContext) -> Path:
    run_dir = context.result_dir("fact_editing")
    result_root = context.path(
        context.env("FACT_EDITING_RESULT_ROOT", str(run_dir / "results"))
    )
    explicit = context.env("FACT_EDITING_BEST_CSV")
    csv_path = context.path(explicit) if explicit else result_root / "best_results.csv"

    if context.should_run and not (
        csv_path.is_file() and not context.env_bool("FACT_EDITING_FORCE_RERUN")
    ):
        context.run_module(
            "hebbian.expts.fact_editing.pipeline",
            "--base-dir",
            context.env("BASE_DIR", str(run_dir)),
            "--gpu-ids",
            context.env("EDIT_GPU_IDS", context.gpu_ids),
            "--python",
            context.python,
        )

    return context.require_file(
        csv_path,
        "Set FACT_EDITING_BEST_CSV or FACT_EDITING_RESULT_ROOT, or use "
        "--mode run-and-plot.",
    )


def _plot_metric(
    context: PaperContext,
    *,
    metric: str,
    output_name: str,
    paper_path: str,
    title: str,
) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _best_results_csv(context)
    context.run_module(
        "hebbian.expts.fact_editing.plot_edit_fraction_sweep",
        csv_path,
        "--metric",
        metric,
        "--output-dir",
        output_dir,
        "--output-name",
        output_name,
        "--points-csv",
        output_dir / "figure_points.csv",
        "--legend-style",
        context.env("FACT_EDITING_LEGEND_STYLE", "method"),
        "--title",
        title,
    )
    context.copy_asset(output_dir / output_name, paper_path)


def score(context: PaperContext) -> None:
    _plot_metric(
        context,
        metric="score",
        output_name="fact_editing_score_h512_normtok_mse_20260607.pdf",
        paper_path=(
            "sections/section_5_transformer_integration/figs/"
            "fact_editing_score_h512_normtok_mse_20260607.pdf"
        ),
        title="Fact Editing Score vs. Percent of Facts Edited",
    )


def nonfact_ppl_ratio(context: PaperContext) -> None:
    _plot_metric(
        context,
        metric="non_fact_ppl_ratio",
        output_name="fact_editing_nonfact_ppl_ratio_h512_normtok_mse_20260607.pdf",
        paper_path=(
            "sections/section_5_transformer_integration/figs/"
            "fact_editing_nonfact_ppl_ratio_h512_normtok_mse_20260607.pdf"
        ),
        title="Non-Fact PPL Ratio vs. Percent of Facts Edited",
    )


def table_h512(context: PaperContext) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tex = output_dir / "fact_editing_table_h512.tex"
    context.run_module(
        "hebbian.expts.fact_editing.export_paper_table",
        _best_results_csv(context),
        "--output-tex",
        output_tex,
    )
    context.copy_asset(
        output_tex, "appendix/experiments/section_5_transformer_integration.tex"
    )
