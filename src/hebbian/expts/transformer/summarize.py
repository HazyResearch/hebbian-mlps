"""Summarize transformer-capacity binary-search results into one CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from hebbian.expts.transformer.sweep import (
    METHOD_DISPLAY,
    METHOD_ORDER,
    load_pickle_compat,
)


KNOWN_METHODS = set(METHOD_DISPLAY) - {"unknown"}
_NUM_FACTS_D_RE = re.compile(r"^d(?P<d>\d+)_fref(?P<fref>\d+)$")
_HIDDEN_DIM_D_RE = re.compile(r"^d(?P<d>\d+)_F(?P<f>\d+)$")
_JUNK_RE = re.compile(r"^junk_len_(?P<j>\d+)$")
_M_RE = re.compile(r"^m(?P<m>\d+)$")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _find_binary_pickles(raw_dir: Path) -> List[Path]:
    pickles = sorted(raw_dir.rglob("binary_search_results_*.pkl"))
    latest_by_dir: Dict[Path, Path] = {}
    for path in pickles:
        previous = latest_by_dir.get(path.parent)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            latest_by_dir[path.parent] = path
    return sorted(latest_by_dir.values())


def _unwrap_payload(obj: Any) -> Dict[str, Any] | None:
    if obj is None:
        return None
    if hasattr(obj, "result"):
        obj = obj.result
    return obj if isinstance(obj, dict) else None


def _extract_achieved(binary_result: Dict[str, Any]) -> tuple[float | None, Dict[str, Any] | None]:
    achieved = binary_result.get("achieved_results")
    if achieved is None:
        return None, None
    if isinstance(achieved, tuple) and len(achieved) >= 2:
        return _safe_float(achieved[0]), _unwrap_payload(achieved[1])
    payload = _unwrap_payload(achieved)
    return None, payload


def _infer_orientation(rel_path: Path) -> str:
    parts = rel_path.parts
    if any(_NUM_FACTS_D_RE.match(part) for part in parts):
        return "num_facts"
    if any(_HIDDEN_DIM_D_RE.match(part) for part in parts):
        return "hidden_dim"
    raise ValueError(f"Could not infer orientation from {rel_path}")


def _parse_metadata(rel_path: Path, orientation: str) -> Dict[str, Any]:
    method = rel_path.parts[0] if rel_path.parts and rel_path.parts[0] in KNOWN_METHODS else "unknown"
    meta: Dict[str, Any] = {
        "method": method,
        "method_display": METHOD_DISPLAY.get(method, method),
        "d_model": "",
        "reference_num_facts": "",
        "fixed_num_facts": "",
        "junk_len": "",
        "fixed_hidden_dim": "",
    }

    for part in rel_path.parts:
        if orientation == "num_facts":
            match = _NUM_FACTS_D_RE.match(part)
            if match:
                meta["d_model"] = int(match.group("d"))
                meta["reference_num_facts"] = int(match.group("fref"))
                continue
        else:
            match = _HIDDEN_DIM_D_RE.match(part)
            if match:
                meta["d_model"] = int(match.group("d"))
                meta["fixed_num_facts"] = int(match.group("f"))
                continue
        match = _JUNK_RE.match(part)
        if match:
            meta["junk_len"] = int(match.group("j"))
            continue
        match = _M_RE.match(part)
        if match:
            meta["fixed_hidden_dim"] = int(match.group("m"))
    return meta


def _row_from_pickle(path: Path, raw_dir: Path) -> Dict[str, Any]:
    data = load_pickle_compat(path)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected pickle payload at {path}")

    rel_path = path.relative_to(raw_dir)
    orientation = _infer_orientation(rel_path)
    meta = _parse_metadata(rel_path, orientation)
    capacity_value, payload = _extract_achieved(data)

    capacity_num_facts = float("nan")
    capacity_hidden_dim = float("nan")
    capacity_value_f = _safe_float(capacity_value)
    if orientation == "num_facts":
        capacity_num_facts = capacity_value_f
        capacity_hidden_dim = _safe_float(meta["fixed_hidden_dim"])
    else:
        capacity_hidden_dim = capacity_value_f
        capacity_num_facts = _safe_float(meta["fixed_num_facts"])

    if payload is not None:
        if not np.isfinite(capacity_num_facts):
            capacity_num_facts = _safe_float(payload.get("num_facts"))
        if not np.isfinite(capacity_hidden_dim):
            capacity_hidden_dim = _safe_float(
                payload.get("mlp_hidden_dim", payload.get("m", float("nan")))
            )
        gamma = _safe_float(payload.get("mlp_gamma_min"))
        best_acc = _safe_float(payload.get("best_acc"))
        best_train_acc = _safe_float(payload.get("best_train_acc"))
        mlp_param_count = _safe_float(payload.get("mlp_param_count"))
    else:
        gamma = float("nan")
        best_acc = float("nan")
        best_train_acc = float("nan")
        mlp_param_count = float("nan")

    d_model = _safe_float(meta["d_model"])
    if (not np.isfinite(mlp_param_count) or mlp_param_count <= 0) and np.isfinite(d_model) and np.isfinite(capacity_hidden_dim):
        mlp_param_count = float(3 * int(d_model) * int(capacity_hidden_dim) + 2 * int(capacity_hidden_dim) + int(d_model))

    return {
        "orientation": orientation,
        "method": meta["method"],
        "method_display": meta["method_display"],
        "d_model": meta["d_model"],
        "reference_num_facts": meta["reference_num_facts"],
        "fixed_num_facts": meta["fixed_num_facts"],
        "junk_len": meta["junk_len"],
        "fixed_hidden_dim": meta["fixed_hidden_dim"],
        "capacity_num_facts": "" if not np.isfinite(capacity_num_facts) else int(round(capacity_num_facts)),
        "capacity_hidden_dim": "" if not np.isfinite(capacity_hidden_dim) else int(round(capacity_hidden_dim)),
        "mlp_param_count": mlp_param_count,
        "mlp_gamma_min_at_capacity": gamma,
        "best_acc_at_capacity": best_acc,
        "best_train_acc_at_capacity": best_train_acc,
        "binary_result_path": str(rel_path),
    }


def summarize_results_dir(results_dir: Path | str, output_csv: Path | str | None = None) -> Path:
    results_dir = Path(results_dir).resolve()
    raw_dir = results_dir / "raw" if (results_dir / "raw").exists() else results_dir
    pickles = _find_binary_pickles(raw_dir)
    rows = [_row_from_pickle(path, raw_dir) for path in pickles]
    method_rank = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    rows.sort(
        key=lambda row: (
            row["orientation"],
            method_rank.get(str(row["method"]), len(METHOD_ORDER)),
            _safe_float(row["d_model"]),
            _safe_float(row["fixed_hidden_dim"] or 0),
            _safe_float(row["fixed_num_facts"] or 0),
        )
    )

    output_csv = Path(output_csv).resolve() if output_csv is not None else results_dir / "capacity_points.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "orientation",
        "method",
        "method_display",
        "d_model",
        "reference_num_facts",
        "fixed_num_facts",
        "junk_len",
        "fixed_hidden_dim",
        "capacity_num_facts",
        "capacity_hidden_dim",
        "mlp_param_count",
        "mlp_gamma_min_at_capacity",
        "best_acc_at_capacity",
        "best_train_acc_at_capacity",
        "binary_result_path",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize transformer-capacity results.")
    parser.add_argument("results_dir", help="Run output root or raw results directory.")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_csv = summarize_results_dir(args.results_dir, args.output_csv)
    print(output_csv)


if __name__ == "__main__":
    main()
