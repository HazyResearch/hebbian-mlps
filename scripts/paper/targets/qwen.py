"""Qwen3 layer-14 artifact and paper targets."""

from __future__ import annotations

from pathlib import Path

from common import PaperContext, PaperError
from targets.capacity import plot_transformer_csv


def _root(context: PaperContext) -> Path:
    return context.path(
        context.env(
            "QWEN3_ARTIFACT_ROOT",
            str(context.artifact_root / "qwen3/qwen3_06b_layer14"),
        )
    )


def _path(context: PaperContext, env_name: str, relative: str) -> Path:
    return context.path(context.env(env_name, str(_root(context) / relative)))


def _activation_dir(context: PaperContext) -> Path:
    return _path(context, "LLM_EMBEDDINGS_DIR", "activations")


def _require_activations(context: PaperContext) -> Path:
    activation_dir = context.require_dir(
        _activation_dir(context),
        "Set QWEN3_ARTIFACT_ROOT or LLM_EMBEDDINGS_DIR; see "
        "docs/paper_artifacts.md.",
    )
    context.require_file(activation_dir / "x.pt", "Activation directory needs x.pt.")
    context.require_file(activation_dir / "y.pt", "Activation directory needs y.pt.")
    return activation_dir


def _margin_sweep(context: PaperContext) -> str:
    sweep = context.env("LLM_MARGIN_SWEEP", "F")
    sweep = {"f": "F", "m": "M"}.get(sweep, sweep)
    if sweep not in {"F", "M"}:
        raise PaperError("LLM_MARGIN_SWEEP must be F or M")
    return sweep


def margin(context: PaperContext) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = _margin_sweep(context)
    stem = context.env("LLM_MARGIN_STEM", f"qwen3_06b_layer14_{sweep}")

    if context.should_run:
        activation_dir = _require_activations(context)
        result_dir = _path(context, "LLM_MARGIN_RESULTS_DIR", "results/margins")
        args: list[str | Path] = [
            "--embeddings-dir",
            activation_dir,
            "--output-dir",
            result_dir,
            "--stem",
            stem,
            "--sweep",
            sweep,
            "--m",
            context.env("LLM_MARGIN_M", "1024"),
            "--f-min",
            context.env("LLM_MARGIN_F_MIN", "4"),
            "--f-max",
            context.env("LLM_MARGIN_F_MAX", "128"),
            "--n-f-points",
            context.env("LLM_MARGIN_N_F_POINTS", "15"),
            "--num-facts",
            context.env("LLM_MARGIN_NUM_FACTS", "512"),
            "--m-min",
            context.env("LLM_MARGIN_M_MIN", "64"),
            "--m-max",
            context.env("LLM_MARGIN_M_MAX", "2048"),
            "--n-m-points",
            context.env("LLM_MARGIN_N_M_POINTS", "15"),
            "--n-seeds",
            context.env("LLM_MARGIN_N_SEEDS", "5"),
            "--seed-offset",
            context.env("LLM_MARGIN_SEED_OFFSET", "0"),
            "--device",
            context.env("LLM_MARGIN_DEVICE", context.device),
            "--max-gpus",
            str(context.n_gpus),
            "--simultaneous-jobs-per-gpu",
            str(context.jobs_per_gpu),
            "--admm-n-iters",
            context.env("LLM_MARGIN_ADMM_N_ITERS", "300"),
            "--admm-batch-size",
            context.env("LLM_MARGIN_ADMM_BATCH_SIZE", "256"),
        ]
        percentile = context.env("LLM_MARGIN_GAMMA_MIN_PERCENTILE")
        if percentile:
            args.extend(["--gamma-min-percentile", percentile])
        args.append(
            "--use-u-star-codes"
            if context.env_bool("LLM_MARGIN_USE_U_STAR_CODES", True)
            else "--no-u-star-codes"
        )
        context.run_module("hebbian.expts.margins.run_qwen3", *args)
        akav_json = context.path(
            context.env("LLM_MARGIN_AKAV_JSON", str(result_dir / f"{stem}_akav.json"))
        )
        rkrv_json = context.path(
            context.env("LLM_MARGIN_RKRV_JSON", str(result_dir / f"{stem}_rkrv.json"))
        )
    else:
        shared_json = context.env("LLM_MARGIN_RESULTS_JSON")
        akav_json = context.path(
            context.env(
                "LLM_MARGIN_AKAV_JSON",
                shared_json
                or str(_root(context) / f"results/margins/{stem}_akav.json"),
            )
        )
        rkrv_json = context.path(
            context.env(
                "LLM_MARGIN_RKRV_JSON",
                shared_json
                or str(_root(context) / f"results/margins/{stem}_rkrv.json"),
            )
        )
    akav_json = context.require_file(
        akav_json, "Set QWEN3_ARTIFACT_ROOT or LLM_MARGIN_AKAV_JSON."
    )
    rkrv_json = context.require_file(
        rkrv_json, "Set QWEN3_ARTIFACT_ROOT or LLM_MARGIN_RKRV_JSON."
    )

    title = context.env("LLM_MARGIN_TITLE", "Qwen3-0.6B layer 14")
    cases = (
        (
            "akav",
            akav_json,
            context.env("LLM_MARGIN_AKAV_TITLE", title),
            context.env(
                "LLM_MARGIN_AKAV_CASE_LABEL", "arbitrary keys, arbitrary values"
            ),
        ),
        (
            "rkrv",
            rkrv_json,
            context.env("LLM_MARGIN_RKRV_TITLE", title),
            context.env(
                "LLM_MARGIN_RKRV_CASE_LABEL", "isotropic keys, isotropic values"
            ),
        ),
    )
    for case, source, case_title, case_label in cases:
        context.run_module(
            "hebbian.expts.margins.plot",
            source,
            "--case",
            case,
            "--output",
            output_dir / case,
            "--title",
            case_title,
            "--case-label",
            case_label,
        )

    paper_dir = "sections/section_4_hebbian_kernel_mlps/figs"
    for case in ("akav", "rkrv"):
        for suffix in ("png", "pdf"):
            source = context.latest(output_dir / case, f"*_{case}_{sweep}*.{suffix}")
            if source:
                context.copy_asset(
                    source,
                    f"{paper_dir}/061426_margin_llm_qwen3_06b_layer14_{case}.{suffix}",
                )


def mlp_capacity(context: PaperContext) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    points_csv = context.env("LLM_MLP_CAPACITY_POINTS_CSV")
    result_dir = _path(context, "LLM_MLP_CAPACITY_SWEEP_DIR", "results/mlp_capacity")
    if context.should_run:
        context.run_module(
            "hebbian.expts.mlp_capacity.run_qwen3",
            "--embeddings-dir",
            _require_activations(context),
            "--output-dir",
            result_dir,
            "--num-facts",
            context.env(
                "LLM_MLP_CAPACITY_NUM_FACTS", "512,1024,2048,4096,8192,16384"
            ),
            "--methods",
            context.env("LLM_MLP_CAPACITY_METHODS", "hebbian_whitened"),
            "--binary-search-range",
            context.env("LLM_MLP_CAPACITY_M_RANGE", "1,65536"),
            "--binary-search-precision",
            context.env("LLM_MLP_CAPACITY_BINARY_SEARCH_PRECISION", "0.05"),
            "--success-acc-threshold",
            context.env("LLM_MLP_CAPACITY_SUCCESS_ACC_THRESHOLD", "0.98"),
            "--seed",
            context.env("LLM_MLP_CAPACITY_SEED", "42"),
            "--device",
            context.env("LLM_MLP_CAPACITY_DEVICE", context.device),
            "--max-gpus",
            str(context.n_gpus),
            "--simultaneous-jobs-per-gpu",
            str(context.jobs_per_gpu),
        )
    title = context.env(
        "LLM_MLP_CAPACITY_PLOT_TITLE",
        "MLP Storage Capacity Scaling w/ LLM Embeddings\\n"
        "(QWEN3-0.6B - Mid Layer)",
    )
    args: list[str | Path] = ["--output-dir", output_dir, "--title", title, "--no-show"]
    if points_csv:
        args[0:0] = [
            "--input-points-csv",
            context.require_file(
                points_csv, "Set LLM_MLP_CAPACITY_POINTS_CSV to a compact CSV."
            ),
        ]
    else:
        args.insert(
            0,
            context.require_dir(
                result_dir,
                "Set QWEN3_ARTIFACT_ROOT or LLM_MLP_CAPACITY_SWEEP_DIR.",
            ),
        )
    context.run_module("hebbian.expts.mlp_capacity.plot", *args)
    context.copy_asset(
        output_dir / "f_vs_w.png",
        "sections/section_4_hebbian_kernel_mlps/figs/"
        "061426_mlp_capacity_llm_qwen3_06b_layer14_whitened.png",
    )


def transformer_capacity(context: PaperContext) -> None:
    output_dir = context.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = _path(
        context, "LLM_TRANSFORMER_CAPACITY_DIR", "results/transformer_capacity"
    )
    csv_path = context.env("LLM_TRANSFORMER_CAPACITY_CSV")
    title = context.env(
        "LLM_TRANSFORMER_CAPACITY_PLOT_TITLE",
        "Transformer Block Storage Capacity Scaling w/ LLM Embeddings\\n"
        "(QWEN3 - 0.6B - Mid Layer)",
    )
    if context.should_run:
        args: list[str | Path] = [
            "--preset",
            context.env(
                "LLM_TRANSFORMER_CAPACITY_PRESET", "paper_trainacc100_hidden_dim"
            ),
            "--embeddings-dir",
            _require_activations(context),
            "--output-root",
            result_dir,
            "--max-gpus",
            str(context.n_gpus),
            "--simultaneous-jobs-per-gpu",
            str(context.jobs_per_gpu),
            "--mlp-methods",
            context.env("LLM_TRANSFORMER_CAPACITY_METHODS", "hebbian_whitened"),
            "--success-metric",
            context.env("LLM_TRANSFORMER_CAPACITY_SUCCESS_METRIC", "best_acc"),
            "--best-acc-success-threshold",
            context.env("LLM_TRANSFORMER_CAPACITY_SUCCESS_THRESHOLD", "0.98"),
        ]
        optional = (
            ("LLM_TRANSFORMER_CAPACITY_D_MODELS", "--d-models"),
            ("LLM_TRANSFORMER_CAPACITY_NUM_FACTS", "--num-facts-values"),
            ("LLM_TRANSFORMER_CAPACITY_NUM_EPOCHS", "--num-epochs"),
            ("LLM_TRANSFORMER_CAPACITY_STEPS_PER_DATASET", "--steps-per-dataset"),
            ("LLM_TRANSFORMER_CAPACITY_BATCH_SIZE", "--batch-size"),
            ("LLM_TRANSFORMER_CAPACITY_HIDDEN_DIM_MIN", "--hidden-dim-search-min"),
            ("LLM_TRANSFORMER_CAPACITY_HIDDEN_DIM_MAX", "--hidden-dim-search-max"),
            (
                "LLM_TRANSFORMER_CAPACITY_BINARY_SEARCH_PRECISION",
                "--binary-search-precision",
            ),
            ("LLM_TRANSFORMER_CAPACITY_N_SEEDS", "--n-seeds"),
            ("LLM_TRANSFORMER_CAPACITY_DEVICE", "--device"),
            ("LLM_TRANSFORMER_CAPACITY_DTYPE", "--dtype"),
            ("LLM_TRANSFORMER_CAPACITY_USE_LOCAL_RUNNER", "--use-local-runner"),
        )
        for env_name, flag in optional:
            value = context.env(env_name)
            if value:
                args.extend([flag, value])
        context.run_module(
            "hebbian.expts.transformer.run_hidden_dim",
            *args,
            extra_env={"HEBBIAN_TRANSFORMER_CAPACITY_PLOT_TITLE": title},
        )
        csv_path = str(result_dir / "capacity_points.csv")
    if csv_path is None:
        csv_path = str(result_dir / "capacity_points.csv")
    csv_file = context.require_file(
        csv_path,
        "Set QWEN3_ARTIFACT_ROOT or LLM_TRANSFORMER_CAPACITY_CSV, or use "
        "--mode run-and-plot.",
    )
    plot_transformer_csv(context, csv_file, output_dir, title=title)
    context.copy_asset(
        output_dir / "transformer_capacity_scaling.png",
        "sections/section_5_transformer_integration/figs/"
        "061426_transformer_capacity_llm_qwen3_06b_layer14.png",
    )


def prepare(context: PaperContext) -> None:
    artifact_root = _root(context)
    implementation = context.env("QWEN3_CAPTURE_IMPL", "packed_stream")
    common_args: list[str | Path] = [
        "--output-root",
        artifact_root,
        "--model-id",
        context.env("QWEN3_MODEL_ID", "Qwen/Qwen3-0.6B-Base"),
        "--layer-index",
        context.env("QWEN3_LAYER_INDEX", "14"),
        "--max-pairs",
        context.env("QWEN3_MAX_PAIRS", "500000"),
        "--batch-size",
        context.env("QWEN3_BATCH_SIZE", "4"),
        "--device",
        context.env("QWEN3_DEVICE", context.device),
        "--save-dtype",
        context.env("QWEN3_SAVE_DTYPE", "float32"),
        "--model-dtype",
        context.env("QWEN3_MODEL_DTYPE", "auto"),
    ]
    if implementation == "packed_stream":
        module = "hebbian.expts.llm_embeddings.capture_mlp_io"
        args = [
            *common_args,
            "--seq-length",
            context.env("QWEN3_SEQ_LENGTH", "1024"),
        ]
        if context.env_bool("QWEN3_STREAMING"):
            args.append("--streaming")
    elif implementation == "batch_nonpadding":
        module = "hebbian.expts.llm_embeddings.extract_qwen3"
        args = [
            *common_args,
            "--max-length",
            context.env("QWEN3_MAX_LENGTH", "512"),
        ]
        if context.env_bool("QWEN3_NO_STREAMING"):
            args.append("--no-streaming")
    else:
        raise PaperError(
            "QWEN3_CAPTURE_IMPL must be packed_stream or batch_nonpadding"
        )

    text_file = context.env("QWEN3_TEXT_FILE")
    if text_file:
        args.extend(["--text-file", text_file])
    else:
        args.extend(
            [
                "--dataset-name",
                context.env("QWEN3_DATASET_NAME", "wikitext"),
                "--dataset-config",
                context.env("QWEN3_DATASET_CONFIG", "wikitext-103-raw-v1"),
                "--dataset-split",
                context.env("QWEN3_DATASET_SPLIT", "train"),
            ]
        )
    if context.env_bool("QWEN3_TRUST_REMOTE_CODE"):
        args.append("--trust-remote-code")
    if context.env_bool("QWEN3_ALLOW_SHORT"):
        args.append("--allow-short")
    revision = context.env("QWEN3_REVISION")
    if revision:
        args.extend(["--revision", revision])
    context.run_module(module, *args)

    max_pairs = context.env("QWEN3_MAX_PAIRS", "500000")
    min_rows = (
        context.env("QWEN3_VERIFY_MIN_ROWS", "1")
        if context.env_bool("QWEN3_ALLOW_SHORT")
        else max_pairs
    )
    verify_args: list[str | Path] = [
        "--artifact-root",
        artifact_root,
        "--expected-layer-index",
        context.env("QWEN3_LAYER_INDEX", "14"),
        "--min-rows",
        min_rows,
        "--run-mlp-check",
    ]
    expected_d = context.env("QWEN3_EXPECTED_D_MODEL")
    if expected_d:
        verify_args.extend(["--expected-d-model", expected_d])
    if context.env_bool("QWEN3_RUN_TRANSFORMER_CHECK", True):
        verify_args.append("--run-transformer-check")
    context.run_module("hebbian.expts.llm_embeddings.verify_bundle", *verify_args)
    print(f"[qwen3] prepared activation bundle at {artifact_root}")


def check(context: PaperContext) -> None:
    scope = context.env("QWEN3_CHECK_SCOPE", "all")
    if scope not in {"all", "activations"}:
        raise PaperError("QWEN3_CHECK_SCOPE must be all or activations")
    activation_dir = _activation_dir(context)
    margin_sweep = _margin_sweep(context)
    margin_stem = context.env(
        "LLM_MARGIN_STEM", f"qwen3_06b_layer14_{margin_sweep}"
    )
    shared_margin = context.env("LLM_MARGIN_RESULTS_JSON")
    required = [
        (activation_dir, "activation directory", True),
        (activation_dir / "x.pt", "activation x.pt", False),
        (activation_dir / "y.pt", "activation y.pt", False),
    ]
    if scope == "all":
        required.extend(
            [
                (
                    context.path(
                        context.env(
                            "LLM_MARGIN_AKAV_JSON",
                            shared_margin
                            or str(
                                _root(context)
                                / f"results/margins/{margin_stem}_akav.json"
                            ),
                        )
                    ),
                    "AK-AV margin JSON",
                    False,
                ),
                (
                    context.path(
                        context.env(
                            "LLM_MARGIN_RKRV_JSON",
                            shared_margin
                            or str(
                                _root(context)
                                / f"results/margins/{margin_stem}_rkrv.json"
                            ),
                        )
                    ),
                    "RK-RV margin JSON",
                    False,
                ),
                (
                    _path(
                        context,
                        "LLM_MLP_CAPACITY_SWEEP_DIR",
                        "results/mlp_capacity",
                    ),
                    "MLP-capacity sweep directory",
                    True,
                ),
                (
                    _path(
                        context,
                        "LLM_TRANSFORMER_CAPACITY_CSV",
                        "results/transformer_capacity/capacity_points.csv",
                    ),
                    "transformer-capacity CSV",
                    False,
                ),
            ]
        )

    missing = []
    for path, label, is_dir in required:
        present = path.is_dir() if is_dir else path.is_file()
        print(f"[{'ok' if present else 'missing'}] {label}: {path}")
        if not present:
            missing.append(path)
    if missing:
        raise PaperError(
            "Qwen3 artifact check failed. Set QWEN3_ARTIFACT_ROOT or the "
            "individual LLM_* path overrides documented in docs/paper_artifacts.md."
        )

    verify_args: list[str | Path] = [
        "--embeddings-dir",
        activation_dir,
        "--expected-layer-index",
        context.env("QWEN3_EXPECTED_LAYER_INDEX", "14"),
        "--min-rows",
        context.env("QWEN3_MIN_ROWS", "1"),
    ]
    expected_d = context.env("QWEN3_EXPECTED_D_MODEL")
    if expected_d:
        verify_args.extend(["--expected-d-model", expected_d])
    if context.env_bool("QWEN3_RUN_MLP_CHECK"):
        verify_args.append("--run-mlp-check")
    context.run_module("hebbian.expts.llm_embeddings.verify_bundle", *verify_args)
    print(f"[qwen3] all required {scope} artifacts are present")
