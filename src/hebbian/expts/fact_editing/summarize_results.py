"""Summarize fact-editing result JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from hebbian.config import main as main_decorator

from hebbian.expts.fact_editing.config import SummaryConfig


def harmonic_mean(values: List[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def load_results(directory: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for json_path in Path(directory).rglob("*.json"):
        if json_path.name == "fact_editing_metadata.json":
            continue
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if {"efficacy", "paraphrase", "specificity", "config"}.issubset(payload.keys()):
            results.append(payload)
    return results


def create_results_frame(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        config = result["config"]
        num_preserve_facts = config.get("num_preserve_facts")
        num_alter_facts = config.get("num_alter_facts")
        base_num_facts = config.get("base_num_facts")
        total_tested_facts = None
        percent_edited = None
        if num_preserve_facts is not None and num_alter_facts is not None:
            total_tested_facts = num_preserve_facts + num_alter_facts
            if base_num_facts is not None and base_num_facts > 0:
                percent_edited = 100.0 * num_alter_facts / base_num_facts
            elif total_tested_facts > 0:
                percent_edited = 100.0 * num_alter_facts / total_tested_facts
        row = {
            "method": config["type"],
            "score": harmonic_mean(
                [result["efficacy"], result["paraphrase"], result["specificity"]]
            ),
            "efficacy": result["efficacy"],
            "paraphrase": result["paraphrase"],
            "specificity": result["specificity"],
            "specificity_paraphrase": result["specificity_paraphrase"],
            "non_fact_pre_nll": result.get("non_fact_pre_nll"),
            "non_fact_post_nll": result.get("non_fact_post_nll"),
            "non_fact_pre_ppl": result.get("non_fact_pre_ppl"),
            "non_fact_post_ppl": result.get("non_fact_post_ppl"),
            "non_fact_ppl_ratio": result.get("non_fact_ppl_ratio"),
            "non_fact_num_tokens": result.get("non_fact_num_tokens"),
            "total_tested_facts": total_tested_facts,
            "percent_edited": percent_edited,
            **config,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def best_result_group_columns(frame: pd.DataFrame) -> List[str]:
    group_cols = ["method", "num_preserve_facts", "num_alter_facts"]
    if "base_mlp_variant" in frame.columns:
        frame["_group_base_variant"] = frame["base_mlp_variant"].fillna(frame.get("experiment_dir"))
        group_cols.append("_group_base_variant")
    elif "experiment_dir" in frame.columns:
        group_cols.append("experiment_dir")
    if "gd_replacement_variant" in frame.columns:
        frame["_group_gd_replacement_variant"] = frame["gd_replacement_variant"].fillna("__none__")
        group_cols.append("_group_gd_replacement_variant")
    return group_cols


def run(config: SummaryConfig) -> Dict[str, Any]:
    config.finalize()
    results = load_results(config.directory)
    if not results:
        raise ValueError(f"No fact-editing results found under {config.directory}")
    frame = create_results_frame(results)
    group_cols = best_result_group_columns(frame)
    best_rows = frame.loc[frame.groupby(group_cols)["score"].idxmax()]
    sort_cols = [col for col in group_cols if col != "_group_base_variant"]
    if "_group_base_variant" in best_rows.columns:
        sort_cols.append("_group_base_variant")
    if "_group_gd_replacement_variant" in best_rows.columns:
        sort_cols.append("_group_gd_replacement_variant")
    best_rows = best_rows.sort_values(sort_cols).reset_index(drop=True)
    if "_group_base_variant" in best_rows.columns:
        best_rows = best_rows.drop(columns=["_group_base_variant"])
    if "_group_gd_replacement_variant" in best_rows.columns:
        best_rows = best_rows.drop(columns=["_group_gd_replacement_variant"])
    Path(config.output_csv).parent.mkdir(parents=True, exist_ok=True)
    best_rows.to_csv(config.output_csv, index=False)
    print(best_rows.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved best-result summary to {config.output_csv}")
    return {
        "num_results": len(frame),
        "output_csv": config.output_csv,
    }


@main_decorator(SummaryConfig)
def main(config: SummaryConfig):
    run(config)


if __name__ == "__main__":
    main()
