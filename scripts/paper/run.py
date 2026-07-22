#!/usr/bin/env python3
"""Run or plot one paper reproduction target."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

from common import PaperContext, PaperError
from targets import capacity, fact_editing, margins, qwen


@dataclass(frozen=True)
class Target:
    description: str
    handler: Callable[[PaperContext], None]


TARGETS = {
    "fig_hidden_dim_margin": Target(
        "Hidden-dimension usability and margin sweep", capacity.hidden_dim_margin
    ),
    "fig_margin_violin": Target(
        "Per-key hidden-dimension margin distribution", capacity.margin_violin
    ),
    "fig_rf_margin_m": Target("Random-feature margin versus width", margins.rf_m),
    "fig_rf_margin_f": Target("Random-feature margin versus facts", margins.rf_f),
    "fig_beta_margin_sweeps": Target(
        "Anisotropic key/value margin sweeps", margins.beta
    ),
    "fig_noisy_query_margin": Target(
        "Noisy-query margin validation", margins.noisy_query
    ),
    "fig_mlp_capacity_isotropic": Target(
        "Isotropic standalone MLP capacity", capacity.mlp_isotropic
    ),
    "fig_mlp_capacity_anisotropic": Target(
        "Anisotropic standalone MLP capacity", capacity.mlp_anisotropic
    ),
    "fig_attention_noise_floor": Target(
        "Attention-only noise floor", capacity.attention_noise_floor
    ),
    "fig_transformer_storage_capacity": Target(
        "Transformer storage capacity", capacity.transformer_storage
    ),
    "fig_transformer_capacity_train99": Target(
        "Transformer capacity at 99% train accuracy", capacity.transformer_train99
    ),
    "fig_fact_editing_score": Target(
        "Fact-editing score", fact_editing.score
    ),
    "fig_fact_editing_nonfact_ppl_ratio": Target(
        "Fact-editing non-fact perplexity ratio", fact_editing.nonfact_ppl_ratio
    ),
    "tab_fact_editing_h512": Target(
        "Fact-editing appendix table", fact_editing.table_h512
    ),
    "fig_llm_margin_qwen3_layer14": Target(
        "Qwen3 layer-14 margin sweeps", qwen.margin
    ),
    "fig_llm_mlp_capacity_qwen3_layer14": Target(
        "Qwen3 layer-14 standalone MLP capacity", qwen.mlp_capacity
    ),
    "fig_llm_transformer_capacity_qwen3_layer14": Target(
        "Qwen3 layer-14 Transformer capacity", qwen.transformer_capacity
    ),
    "prepare_qwen3_layer14_artifacts": Target(
        "Capture and verify the Qwen3 layer-14 activation bundle", qwen.prepare
    ),
    "check_qwen3_artifacts": Target(
        "Validate a Qwen3 activation/result bundle", qwen.check
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=sorted(TARGETS),
        metavar="TARGET",
    )
    parser.add_argument("--list", action="store_true", help="List available targets")
    parser.add_argument("--mode", choices=("plot-only", "run-and-plot"))
    parser.add_argument("--repo-root")
    parser.add_argument("--artifact-root")
    parser.add_argument("--result-root")
    parser.add_argument("--figure-root")
    parser.add_argument(
        "--paper-root", help="Optional paper checkout to receive assets"
    )
    parser.add_argument(
        "--python", help="Python interpreter used for experiment modules"
    )
    parser.add_argument("--device", help="Default experiment device")
    parser.add_argument("--n-gpus", type=int)
    parser.add_argument("--jobs-per-gpu", type=int)
    return parser


def print_targets() -> None:
    width = max(len(name) for name in TARGETS)
    for name, target in TARGETS.items():
        print(f"{name:<{width}}  {target.description}")


def main() -> int:
    args = build_parser().parse_args()
    if args.list:
        print_targets()
        return 0
    if args.target is None:
        build_parser().error("target is required unless --list is used")

    try:
        context = PaperContext.from_args(args.target, args)
        TARGETS[args.target].handler(context)
    except (PaperError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
