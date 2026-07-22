"""Synthetic margin paper targets."""

from __future__ import annotations

from pathlib import Path

from common import PaperContext, PaperError, quoted_override


MARGIN_STEMS = {
    ("M", "both"): "rf_margin_m_d64_f256_30pts",
    ("F", "both"): "rf_margin_f_d64_m512_30pts",
    ("beta", "keys"): "beta_margin_keys_d64_f128_m512_30pts",
    ("beta", "values"): "beta_margin_values_d64_f128_m512_30pts",
    ("beta", "both"): "beta_margin_both_d64_f128_m512_30pts",
    ("epsilon", "both"): "noisy_query_margin_epsilon_d64_f256_m512_15pts",
}


def _data_dir(context: PaperContext) -> Path:
    return context.path(
        context.env("MARGIN_DATA_DIR", str(context.result_root / "margins"))
    )


def _stem(sweep: str, spike_target: str) -> str:
    return MARGIN_STEMS.get((sweep, spike_target), f"margin_{sweep}_{spike_target}")


def _default_json(context: PaperContext, sweep: str, spike_target: str) -> Path:
    return _data_dir(context) / f"{_stem(sweep, spike_target)}.json"


def _run_overrides(
    context: PaperContext, sweep: str, spike_target: str
) -> list[str]:
    configured_seeds = context.env("MARGIN_N_SEEDS")
    if configured_seeds is not None:
        n_seeds = configured_seeds
    elif sweep in {"M", "F"}:
        n_seeds = "5"
    elif sweep == "epsilon":
        n_seeds = context.env("NOISY_QUERY_N_SEEDS", "3")
    else:
        n_seeds = "1"

    args = [
        quoted_override("sweep", sweep),
        quoted_override("device", context.device),
        f"max_gpus={context.n_gpus}",
        f"simultaneous_jobs_per_gpu={context.jobs_per_gpu}",
        f'use_u_star_codes={context.env("MARGIN_USE_U_STAR_CODES", "False")}',
        f"n_seeds={n_seeds}",
    ]
    if (sweep, spike_target) == ("M", "both"):
        args.extend(
            [
                f'd={context.env("RF_MARGIN_M_D", "64")}',
                f'F={context.env("RF_MARGIN_M_F", "256")}',
                f'M_min={context.env("RF_MARGIN_M_MIN", "64")}',
                f'M_max={context.env("RF_MARGIN_M_MAX", "1024")}',
                f'n_M_points={context.env("RF_MARGIN_M_N_POINTS", "30")}',
            ]
        )
    elif (sweep, spike_target) == ("F", "both"):
        args.extend(
            [
                f'd={context.env("RF_MARGIN_F_D", "64")}',
                f'M={context.env("RF_MARGIN_F_M", "512")}',
                f'F_min={context.env("RF_MARGIN_F_MIN", "32")}',
                f'F_max={context.env("RF_MARGIN_F_MAX", "512")}',
                f'n_F_points={context.env("RF_MARGIN_F_N_POINTS", "30")}',
            ]
        )
    elif sweep == "beta":
        suffix = spike_target.upper()
        beta_max = {"keys": "5", "values": "10", "both": "3.35"}[spike_target]
        d = context.env(f"BETA_MARGIN_{suffix}_D", context.env("BETA_MARGIN_D", "64"))
        n_facts = context.env(
            f"BETA_MARGIN_{suffix}_F", context.env("BETA_MARGIN_F", "128")
        )
        width = context.env(
            f"BETA_MARGIN_{suffix}_M", context.env("BETA_MARGIN_M", "512")
        )
        beta_min = context.env(
            f"BETA_MARGIN_{suffix}_MIN", context.env("BETA_MARGIN_MIN", "0")
        )
        n_points = context.env(
            f"BETA_MARGIN_{suffix}_N_POINTS",
            context.env("BETA_MARGIN_N_POINTS", "30"),
        )
        args.extend(
            [
                f"d={d}",
                f"F={n_facts}",
                f"M={width}",
                quoted_override("spike_target", spike_target),
                f"beta_min={beta_min}",
                f'beta_max={context.env(f"BETA_MARGIN_{suffix}_MAX", beta_max)}',
                f"n_beta_points={n_points}",
            ]
        )
    elif (sweep, spike_target) == ("epsilon", "both"):
        args.extend(
            [
                f'd={context.env("NOISY_QUERY_MARGIN_D", "64")}',
                f'F={context.env("NOISY_QUERY_MARGIN_F", "256")}',
                f'M={context.env("NOISY_QUERY_MARGIN_M", "512")}',
                f'epsilon_min={context.env("NOISY_QUERY_EPSILON_MIN", "0.01")}',
                f'epsilon_max={context.env("NOISY_QUERY_EPSILON_MAX", "2.0")}',
                f'n_epsilon_points={context.env("NOISY_QUERY_N_EPSILON_POINTS", "15")}',
            ]
        )
    return args


def _results_json(
    context: PaperContext,
    *,
    sweep: str,
    spike_target: str = "both",
    env_name: str = "MARGIN_RESULTS_JSON",
) -> Path:
    default_path = _default_json(context, sweep, spike_target)
    if context.should_run:
        stem = _stem(sweep, spike_target)
        run_dir = _data_dir(context) / "raw" / stem
        default_path.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        if not (default_path.is_file() and context.env_bool("MARGIN_REUSE_JSON")):
            context.run_module(
                "hebbian.expts.margins.run",
                quoted_override("base_dir", run_dir),
                quoted_override("output_json", default_path),
                *_run_overrides(context, sweep, spike_target),
            )
        return default_path

    value = context.env(env_name)
    if value is None and sweep == "beta":
        value = context.env("BETA_MARGIN_RESULTS_JSON")
    if value is None:
        value = context.env("MARGIN_RESULTS_JSON", str(default_path))
    return context.require_file(
        value,
        f"Set {env_name} or MARGIN_RESULTS_JSON, use MARGIN_DATA_DIR, or use "
        "--mode run-and-plot.",
    )


def _plot_claim(
    context: PaperContext,
    *,
    sweep: str,
    case: str,
    paper_path: str,
    env_name: str,
    spike_target: str = "both",
) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    context.run_module(
        "hebbian.expts.margins.plot",
        _results_json(
            context,
            sweep=sweep,
            spike_target=spike_target,
            env_name=env_name,
        ),
        "--case",
        case,
        "--output",
        output_dir,
    )
    png = context.latest(output_dir, f"*_{case}_{sweep}*.png")
    pdf = context.latest(output_dir, f"*_{case}_{sweep}*.pdf")
    if context.paper_root and png is None:
        raise PaperError(f"plotter produced no *_{case}_{sweep}*.png in {output_dir}")
    if png:
        context.copy_asset(png, paper_path)
    if pdf:
        context.copy_asset(pdf, f"{paper_path.removesuffix('.png')}.pdf")


def rf_m(context: PaperContext) -> None:
    _plot_claim(
        context,
        sweep="M",
        case="rkrv",
        paper_path=(
            "sections/section_4_hebbian_kernel_mlps/figs_0330/"
            "rf_M_d64_30pts_rkrv_M.png"
        ),
        env_name="RF_MARGIN_M_RESULTS_JSON",
    )


def rf_f(context: PaperContext) -> None:
    _plot_claim(
        context,
        sweep="F",
        case="rkrv",
        paper_path="appendix/theory/rf_F_d64_30pts_rkrv_F.png",
        env_name="RF_MARGIN_F_RESULTS_JSON",
    )


def beta(context: PaperContext) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        ("keys", "akrv", "BETA_MARGIN_KEYS_RESULTS_JSON"),
        ("values", "rkav", "BETA_MARGIN_VALUES_RESULTS_JSON"),
        ("both", "akav", "BETA_MARGIN_BOTH_RESULTS_JSON"),
    )
    for spike_target, case, env_name in specs:
        target_dir = output_dir / spike_target
        context.run_module(
            "hebbian.expts.margins.plot",
            _results_json(
                context,
                sweep="beta",
                spike_target=spike_target,
                env_name=env_name,
            ),
            "--case",
            case,
            "--output",
            target_dir,
        )

    paper_outputs = (
        ("keys", "akrv", "beta_keys_F128_bmax35_akrv_beta"),
        ("values", "rkav", "beta_values_F128_bmax10_rkav_beta"),
        ("both", "akav", "beta_both_F128_bmax335_akav_beta"),
    )
    paper_dir = "sections/section_4_hebbian_kernel_mlps/figs_0330"
    for spike_target, case, stem in paper_outputs:
        for suffix in ("png", "pdf"):
            source = context.latest(
                output_dir / spike_target, f"*_{case}_beta*.{suffix}"
            )
            if source:
                context.copy_asset(source, f"{paper_dir}/{stem}.{suffix}")


def noisy_query(context: PaperContext) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    context.run_module(
        "hebbian.expts.margins.plot_noisy_query",
        _results_json(
            context,
            sweep="epsilon",
            env_name="NOISY_QUERY_MARGIN_JSON",
        ),
        "--output",
        output_dir,
    )
    for suffix in ("png", "pdf"):
        source = context.latest(output_dir, f"*_fit_only.{suffix}")
        if source:
            context.copy_asset(
                source,
                f"appendix/theory/margin_sweep_results_aux_epsilon_fit_only.{suffix}",
            )
