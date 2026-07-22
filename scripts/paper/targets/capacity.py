"""Hidden-dimension, MLP-capacity, attention, and Transformer targets."""

from __future__ import annotations

from pathlib import Path

from common import PaperContext, quoted_override


def plot_transformer_csv(
    context: PaperContext,
    csv_path: Path,
    output_dir: Path,
    *,
    title: str = "Transformer Capacity Scaling",
    paper_style: bool = False,
    y_label: str | None = None,
) -> None:
    args: list[str | Path] = [csv_path, "--output-dir", output_dir]
    if paper_style:
        args.append("--paper-style")
    args.extend(["--title", title])
    if y_label:
        args.extend(["--y-label", y_label])
    context.run_module("hebbian.expts.transformer.plot", *args)


def transformer_storage(context: PaperContext) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = context.result_dir("transformer_storage_capacity")
    csv_path = context.path(
        context.env("TRANSFORMER_CAPACITY_CSV", str(result_dir / "capacity_points.csv"))
    )
    if context.should_run:
        context.run_module(
            "hebbian.expts.transformer.run_num_facts",
            "--preset",
            context.env(
                "TRANSFORMER_CAPACITY_PRESET", "full_num_facts_attn_pretrain"
            ),
            "--output-root",
            result_dir,
            "--max-gpus",
            str(context.n_gpus),
            "--simultaneous-jobs-per-gpu",
            str(context.jobs_per_gpu),
        )
        csv_path = result_dir / "capacity_points.csv"
    csv_path = context.require_file(
        csv_path,
        "Set TRANSFORMER_CAPACITY_CSV or use --mode run-and-plot.",
    )
    plot_transformer_csv(
        context, csv_path, output_dir, title="", paper_style=True
    )
    context.copy_asset(
        output_dir / "transformer_capacity_scaling.png",
        "sections/section_5_transformer_integration/figs_0330/"
        "transformer_storage_capacity.png",
    )


def transformer_train99(context: PaperContext) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = context.result_dir("transformer_train99_capacity")
    csv_path = context.path(
        context.env("TRANSFORMER_TRAIN99_CSV", str(result_dir / "capacity_points.csv"))
    )
    if context.should_run:
        context.run_module(
            "hebbian.expts.transformer.run_num_facts",
            "--preset",
            context.env("TRANSFORMER_TRAIN99_PRESET", "paper_train99_num_facts"),
            "--output-root",
            result_dir,
            "--max-gpus",
            str(context.n_gpus),
            "--simultaneous-jobs-per-gpu",
            str(context.jobs_per_gpu),
        )
        csv_path = result_dir / "capacity_points.csv"
    csv_path = context.require_file(
        csv_path,
        "Set TRANSFORMER_TRAIN99_CSV or use --mode run-and-plot.",
    )
    plot_transformer_csv(
        context,
        csv_path,
        output_dir,
        title="",
        paper_style=True,
        y_label="Number of Facts (99% Train Acc)",
    )
    context.copy_asset(
        output_dir / "transformer_capacity_scaling.png",
        "sections/section_5_transformer_integration/figs/"
        "040126_transformer_capacity_scaling_train99.png",
    )


def _hidden_dim_source(context: PaperContext) -> Path:
    run_dir = context.result_dir("hidden_dim")
    if context.should_run:
        context.run_module(
            "hebbian.expts.hidden_dim.run",
            quoted_override("base_dir", run_dir),
            f'mlp_method={context.env("HIDDEN_DIM_MLP_METHOD", "gd")}',
            f'hidden_dim_max={context.env("HIDDEN_DIM_MAX", "256")}',
            f'n_hidden_dims={context.env("HIDDEN_DIM_N_POINTS", "10")}',
            f'attn_residual={context.env("HIDDEN_DIM_ATTN_RESIDUAL", "True")}',
            f"max_gpus={context.n_gpus}",
            f"simultaneous_jobs_per_gpu={context.jobs_per_gpu}",
        )
        return run_dir
    return context.require_dir(
        context.env("HIDDEN_DIM_SWEEP_DIR", str(run_dir)),
        "Set HIDDEN_DIM_SWEEP_DIR or use --mode run-and-plot.",
    )


def hidden_dim_margin(context: PaperContext) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        quoted_override("output_dir", output_dir),
        quoted_override("points_output_csv", output_dir / "figure_points.csv"),
        'stem="hidden_dim_sweep_overlayed"',
    ]
    points_csv = context.env("HIDDEN_DIM_POINTS_CSV")
    if points_csv:
        points_path = context.require_file(
            points_csv,
            "Set HIDDEN_DIM_POINTS_CSV to a compact CSV or set "
            "HIDDEN_DIM_SWEEP_DIR.",
        )
        args.insert(0, quoted_override("points_csv", points_path))
    else:
        args.insert(0, quoted_override("base_dir", _hidden_dim_source(context)))
    context.run_module("hebbian.expts.hidden_dim.plot_dual_axis", *args)
    context.copy_asset(
        output_dir / "hidden_dim_sweep_overlayed.png",
        "sections/section_3_basic_setting/figs/hidden_dim_sweep_overlayed_03_31.png",
    )


def margin_violin(context: PaperContext) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = [quoted_override("output_dir", output_dir)]
    points_npz = context.env("MARGIN_VIOLIN_POINTS_NPZ")
    if points_npz:
        points_path = context.require_file(
            points_npz,
            "Set MARGIN_VIOLIN_POINTS_NPZ to compact per-key data or set "
            "HIDDEN_DIM_SWEEP_DIR.",
        )
        args.insert(0, quoted_override("points_npz", points_path))
    else:
        args.insert(0, quoted_override("base_dir", _hidden_dim_source(context)))
    context.run_module("hebbian.expts.hidden_dim.plot_margin_distribution", *args)
    context.copy_asset(
        output_dir / "margin_violin.png",
        "sections/section_5_transformer_integration/figs/margin_violin_03_24.png",
    )


def _mlp_capacity_source(
    context: PaperContext, *, env_name: str, anisotropic: bool
) -> Path:
    kind = "anisotropic" if anisotropic else "isotropic"
    run_dir = context.result_dir(f"mlp_capacity/{kind}")
    if context.should_run:
        args = [
            quoted_override("base_dir", run_dir),
            quoted_override("device", context.device),
            f"max_gpus={context.n_gpus}",
            f"simultaneous_jobs_per_gpu={context.jobs_per_gpu}",
            "methods=('gd','hebbian','hebbian_whitened','cf_coord_whitened','ntk')",
        ]
        if anisotropic:
            args.extend(
                [
                    f'spike_beta={context.env("MLP_CAPACITY_SPIKE_BETA", "1.5")}',
                    quoted_override(
                        "spike_target",
                        context.env("MLP_CAPACITY_SPIKE_TARGET", "both"),
                    ),
                ]
            )
        context.run_module("hebbian.expts.mlp_capacity.run", *args)
        return run_dir
    source = context.env(
        env_name,
        context.env("MLP_CAPACITY_SWEEP_DIR", str(run_dir)),
    )
    return context.require_dir(
        source,
        f"Set {env_name} or MLP_CAPACITY_SWEEP_DIR, or use "
        "--mode run-and-plot.",
    )


def _plot_mlp_capacity(
    context: PaperContext,
    *,
    source_env: str,
    points_env: str,
    anisotropic: bool,
    title: str,
    paper_path: str,
) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    args: list[str | Path] = [
        "--output-dir",
        output_dir,
        "--title",
        title,
        "--method-display-override",
        "cf_coord_whitened=Ours (data-dependent)",
    ]
    for method in ("gd", "hebbian", "hebbian_whitened", "cf_coord_whitened", "ntk"):
        args.extend(["--method-order", method])
    args.append("--no-show")

    points_csv = context.env(points_env)
    if points_csv:
        args.extend(
            [
                "--input-points-csv",
                context.require_file(
                    points_csv,
                    f"Set {points_env} to a compact capacity CSV or set {source_env}.",
                ),
            ]
        )
    else:
        args.insert(
            0,
            _mlp_capacity_source(
                context, env_name=source_env, anisotropic=anisotropic
            ),
        )
    context.run_module("hebbian.expts.mlp_capacity.plot", *args)
    context.copy_asset(output_dir / "f_vs_w.png", paper_path)


def mlp_isotropic(context: PaperContext) -> None:
    _plot_mlp_capacity(
        context,
        source_env="MLP_CAPACITY_ISOTROPIC_DIR",
        points_env="MLP_CAPACITY_ISOTROPIC_POINTS_CSV",
        anisotropic=False,
        title="",
        paper_path=(
            "sections/section_4_hebbian_kernel_mlps/figs/"
            "040126_mlp_capacity_isotropic_final.png"
        ),
    )


def mlp_anisotropic(context: PaperContext) -> None:
    _plot_mlp_capacity(
        context,
        source_env="MLP_CAPACITY_ANISOTROPIC_DIR",
        points_env="MLP_CAPACITY_ANISOTROPIC_POINTS_CSV",
        anisotropic=True,
        title="MLP Capacity Scaling (Anisotropic Keys & Values)",
        paper_path=(
            "sections/section_4_hebbian_kernel_mlps/figs/"
            "040126_mlp_capacity_anisotropic_beta1p5_both_final.png"
        ),
    )


def _attention_source(context: PaperContext) -> Path:
    run_dir = context.result_dir("attention_noise_floor")
    if context.should_run:
        context.run_module(
            "hebbian.expts.attention_noise.run_attention_only",
            quoted_override("base_dir", run_dir),
            f"max_gpus={context.n_gpus}",
            f"simultaneous_jobs_per_gpu={context.jobs_per_gpu}",
        )
        return run_dir
    return context.require_dir(
        context.env("ATTENTION_NOISE_SWEEP_DIR", str(run_dir)),
        "Set ATTENTION_NOISE_SWEEP_DIR or use --mode run-and-plot.",
    )


def attention_noise_floor(context: PaperContext) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    args: list[str | Path] = []
    points_csv = context.env("ATTENTION_NOISE_POINTS_CSV")
    if points_csv:
        args.extend(
            [
                output_dir,
                "--input-points-csv",
                context.require_file(
                    points_csv,
                    "Set ATTENTION_NOISE_POINTS_CSV to a compact CSV or set "
                    "ATTENTION_NOISE_SWEEP_DIR.",
                ),
            ]
        )
    else:
        args.append(_attention_source(context))
    args.extend(
        [
            "--output-dir",
            output_dir,
            "--output-stem",
            "attention_only_noise_floor",
            "--points-csv",
            output_dir / "figure_points.csv",
            "--no-show",
        ]
    )
    context.run_module("hebbian.expts.attention_noise.plot_pretty", *args)
    context.copy_asset(
        output_dir / "attention_only_noise_floor.png",
        "sections/section_5_transformer_integration/figs_0330/"
        "attention_only_noise_floor.png",
    )
