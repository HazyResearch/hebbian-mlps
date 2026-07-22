#!/usr/bin/env python3
"""Plot Section 4 MLP capacity sweep results (F vs W) from binary-search pickles.

This script mirrors the Fig2 post-processing flow in `expts/synthetics/sweeps/fig2`
for Section 4 outputs under:

  <base_dir>/<method>/d{d_model}_F{num_facts}/binary_search_results_*.pkl

It writes:
  - summary CSV (default: <output_dir>/summary.csv)
  - F-vs-W plot (default: <output_dir>/f_vs_w.png)
  - F-vs-W plot with heuristic method fits (default: <output_dir>/f_vs_w_with_fits.png)
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


METHOD_ORDER = [
    "gd",
    "hebbian",
    "hebbian_whitened",
    "cf_coord_whitened",
    "ntk",
    "unknown",
]
METHOD_DISPLAY = {
    "gd": "GD",
    "hebbian": "Ours (no whitening)",
    "hebbian_whitened": "Ours (whitened)",
    "cf_coord_whitened": "Ours (data-dependent)",
    "ntk": "NTK",
    "unknown": "Unknown",
}
METHOD_COLOR = {
    "gd": "#E63946",       # red
    "hebbian": "#2E86AB",  # blue
    "hebbian_whitened": "#16A085",  # teal
    "cf_coord_whitened": "#6D28D9",  # deeper purple
    "ntk": "#F18F01",      # orange
    "unknown": "#666666",
}
D_MODEL_MARKER = {64: "o", 90: "s", 128: "^"}
D_MODEL_LINESTYLE = {64: "-", 90: "--", 128: "-."}

KNOWN_METHODS = {
    "gd",
    "hebbian",
    "hebbian_whitened",
    "cf_coord_whitened",
    "ntk",
}
_D_F_RE = re.compile(r"^d(?P<d>\d+)_F(?P<f>\d+)$")


class _CompatUnpickler(pickle.Unpickler):
    """Unpickler resilient to `__main__` class refs and CUDA tensor storages."""

    _dummy_cls_cache: dict[tuple[str, str], type] = {}

    @classmethod
    def _dummy_cls(cls, module: str, name: str) -> type:
        def _noop(*_args, **_kwargs):
            return None

        def _fallback_getattr(self, _attr):
            return _noop

        def _setstate(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

        key = (module, name)
        if key not in cls._dummy_cls_cache:
            cls._dummy_cls_cache[key] = type(
                name,
                (),
                {
                    "__module__": module,
                    "__getattr__": _fallback_getattr,
                    "__setstate__": _setstate,
                    "run_experiment_config": _noop,
                    "get_experiment_config_and_base_dir": _noop,
                    "agg_results": _noop,
                },
            )
        return cls._dummy_cls_cache[key]

    def find_class(self, module: str, name: str):
        if module == "__main__":
            return self._dummy_cls(module, name)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _load_pickle_compat(path: Path) -> Any:
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _coerce_pairs(raw_value: Any) -> List[Tuple[float, Any]]:
    pairs: List[Tuple[float, Any]] = []
    if isinstance(raw_value, tuple) and len(raw_value) >= 2:
        pairs.append((raw_value[0], raw_value[1]))
    elif isinstance(raw_value, list):
        for item in raw_value:
            if isinstance(item, tuple) and len(item) >= 2:
                pairs.append((item[0], item[1]))
    return pairs


def setup_plot_style() -> None:
    """Set the paper's Fig. 2 plot styling."""
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


def _find_binary_pickles(base_dir: Path) -> List[Path]:
    pickles = sorted(base_dir.rglob("binary_search_results_*.pkl"))
    latest_by_dir: Dict[Path, Path] = {}
    for p in pickles:
        prev = latest_by_dir.get(p.parent)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            latest_by_dir[p.parent] = p
    return sorted(latest_by_dir.values())


def _parse_path_metadata(rel_path: Path) -> Dict[str, Any]:
    method = "unknown"
    d_model = None
    num_facts = None

    if len(rel_path.parts) > 0 and rel_path.parts[0] in KNOWN_METHODS:
        method = rel_path.parts[0]

    for part in rel_path.parts:
        m = _D_F_RE.match(part)
        if m:
            d_model = int(m.group("d"))
            num_facts = int(m.group("f"))
            break

    return {"method": method, "d_model": d_model, "num_facts": num_facts}


def _extract_result_dict(job_result: Any) -> Dict[str, Any]:
    if hasattr(job_result, "result") and isinstance(job_result.result, dict):
        return dict(job_result.result)
    if isinstance(job_result, dict):
        return dict(job_result)
    return {}


def collect_rows(base_dir: Path, verbose: bool = False) -> List[Dict[str, Any]]:
    pickles = _find_binary_pickles(base_dir)
    if not pickles:
        raise FileNotFoundError(f"No binary_search_results_*.pkl files found under: {base_dir}")

    rows: List[Dict[str, Any]] = []
    skipped_no_achieved = 0
    skipped_bad = 0

    for pkl_path in pickles:
        try:
            data = _load_pickle_compat(pkl_path)
            if not isinstance(data, dict):
                skipped_bad += 1
                continue

            achieved_pairs = _coerce_pairs(data.get("achieved_results", []))
            if not achieved_pairs:
                skipped_no_achieved += 1
                continue

            search_value, job_result = achieved_pairs[-1]
            result_dict = _extract_result_dict(job_result)
            rel_path = pkl_path.relative_to(base_dir)
            meta = _parse_path_metadata(rel_path)

            method = str(result_dict.get("method") or meta["method"] or "unknown")
            if method not in METHOD_DISPLAY:
                method = "unknown"

            d_model = meta["d_model"]
            num_facts = meta["num_facts"]

            m_star: Optional[int]
            m_raw = result_dict.get("m", None)
            if m_raw is not None:
                m_star = int(round(float(m_raw)))
            else:
                m_star = int(round(float(search_value))) if search_value is not None else None

            best_acc = _to_float(result_dict.get("best_acc"))
            param_count = _to_float(result_dict.get("param_count"))
            out_file = getattr(job_result, "out_file", None)

            rows.append(
                {
                    "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    "d_model": int(d_model) if d_model is not None else "",
                    "num_facts": int(num_facts) if num_facts is not None else "",
                    "m_star": int(m_star) if m_star is not None else "",
                    "param_count": param_count,
                    "best_acc": best_acc,
                    "binary_result_path": str(rel_path),
                    "out_file": str(out_file) if out_file is not None else "",
                    "data_source_method": method,
                    "data_source_display": METHOD_DISPLAY[method],
                    "is_replaced_point": False,
                }
            )
        except Exception:
            skipped_bad += 1
            continue

    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    rows.sort(
        key=lambda r: (
            method_rank.get(str(r["method"]), len(METHOD_ORDER)),
            int(r["d_model"]) if r["d_model"] != "" else 10**9,
            int(r["num_facts"]) if r["num_facts"] != "" else 10**18,
            _to_float(r["param_count"]),
        )
    )

    if verbose:
        print(f"Found {len(pickles)} binary-search result files.")
        print(f"Loaded {len(rows)} achieved-capacity rows.")
        if skipped_no_achieved:
            print(f"Skipped {skipped_no_achieved} file(s) with no achieved result.")
        if skipped_bad:
            print(f"Skipped {skipped_bad} unreadable/invalid file(s).")

    return rows


def load_rows_from_points_csv(points_csv: Path) -> List[Dict[str, Any]]:
    """Load a compact paper `figure_points.csv`/`summary.csv` as plot rows."""
    rows: List[Dict[str, Any]] = []
    with open(points_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row.get("method") or "unknown")
            method_display = str(row.get("method_display") or METHOD_DISPLAY.get(method, method))
            param_count = row.get("param_count", row.get("num_parameters", ""))
            rows.append(
                {
                    "method": method,
                    "method_display": method_display,
                    "d_model": row.get("d_model", ""),
                    "num_facts": row.get("num_facts", ""),
                    "m_star": row.get("m_star", ""),
                    "param_count": param_count,
                    "best_acc": row.get("best_acc", ""),
                    "binary_result_path": row.get("binary_result_path", ""),
                    "out_file": row.get("out_file", ""),
                    "data_source_method": row.get("data_source_method", method),
                    "data_source_display": row.get("data_source_display", method_display),
                    "is_replaced_point": row.get("is_replaced_point", False),
                }
            )
    return _sort_rows(rows)


def write_summary_csv(rows: List[Dict[str, Any]], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "method_display",
        "d_model",
        "num_facts",
        "m_star",
        "param_count",
        "best_acc",
        "binary_result_path",
        "out_file",
        "data_source_method",
        "data_source_display",
        "is_replaced_point",
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_points_csv(rows: List[Dict[str, Any]], points_csv: Path) -> None:
    points_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "method_display",
        "d_model",
        "num_facts",
        "num_parameters",
        "m_star",
        "best_acc",
        "data_source_method",
        "data_source_display",
        "is_replaced_point",
        "binary_result_path",
        "out_file",
    ]
    with open(points_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.get("method", ""),
                    "method_display": row.get("method_display", ""),
                    "d_model": row.get("d_model", ""),
                    "num_facts": row.get("num_facts", ""),
                    "num_parameters": row.get("param_count", ""),
                    "m_star": row.get("m_star", ""),
                    "best_acc": row.get("best_acc", ""),
                    "data_source_method": row.get("data_source_method", row.get("method", "")),
                    "data_source_display": row.get(
                        "data_source_display",
                        row.get("method_display", ""),
                    ),
                    "is_replaced_point": row.get("is_replaced_point", False),
                    "binary_result_path": row.get("binary_result_path", ""),
                    "out_file": row.get("out_file", ""),
                }
            )


def combine_rows(*row_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine multiple row lists, deduplicating by (method, d_model, num_facts)."""
    combined: Dict[tuple[str, Any, Any], Dict[str, Any]] = {}
    for rows in row_lists:
        for row in rows:
            key = (str(row.get("method")), row.get("d_model"), row.get("num_facts"))
            combined[key] = row
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(
        combined.values(),
        key=lambda r: (
            method_rank.get(str(r["method"]), len(METHOD_ORDER)),
            int(r["d_model"]) if r["d_model"] != "" else 10**9,
            int(r["num_facts"]) if r["num_facts"] != "" else 10**18,
            _to_float(r["param_count"]),
        ),
    )


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(
        rows,
        key=lambda r: (
            method_rank.get(str(r["method"]), len(METHOD_ORDER)),
            int(r["d_model"]) if r["d_model"] != "" else 10**9,
            int(r["num_facts"]) if r["num_facts"] != "" else 10**18,
            _to_float(r["param_count"]),
        ),
    )


def filter_rows_by_method(rows: List[Dict[str, Any]], include_methods: List[str]) -> List[Dict[str, Any]]:
    include_set = set(include_methods)
    return _sort_rows([row for row in rows if str(row.get("method")) in include_set])


def parse_method_display_overrides(
    override_specs: List[str],
) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for spec in override_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid method display override {spec!r}; expected METHOD=LABEL."
            )
        method, label = spec.split("=", 1)
        method = method.strip()
        label = label.strip()
        if not method or not label:
            raise ValueError(
                f"Invalid method display override {spec!r}; expected METHOD=LABEL."
            )
        overrides[method] = label
    return overrides


def apply_method_display_overrides(
    rows: List[Dict[str, Any]],
    display_overrides: Dict[str, str],
) -> List[Dict[str, Any]]:
    if not display_overrides:
        return rows
    updated: List[Dict[str, Any]] = []
    for row in rows:
        row_i = dict(row)
        method = str(row_i.get("method", ""))
        if method in display_overrides:
            row_i["method_display"] = display_overrides[method]
        updated.append(row_i)
    return updated


def _parse_point_replacement_spec(spec: str) -> tuple[str, int, int, str]:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Invalid point replacement {spec!r}; expected "
            "'target_method,d_model,num_facts,source_method'."
        )
    target_method, d_model_str, num_facts_str, source_method = parts
    try:
        d_model = int(d_model_str)
        num_facts = int(num_facts_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid point replacement {spec!r}; d_model and num_facts must be ints."
        ) from exc
    return target_method, d_model, num_facts, source_method


def apply_point_replacements(rows: List[Dict[str, Any]], replacement_specs: List[str]) -> List[Dict[str, Any]]:
    if not replacement_specs:
        return rows

    by_key: Dict[tuple[str, int, int], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["d_model"]), int(row["num_facts"]))
        by_key[key] = dict(row)

    for spec in replacement_specs:
        target_method, d_model, num_facts, source_method = _parse_point_replacement_spec(spec)
        source_key = (source_method, d_model, num_facts)
        if source_key not in by_key:
            raise KeyError(
                f"Replacement source point not found for spec {spec!r}: {source_key}"
            )
        source_row = dict(by_key[source_key])
        source_row["method"] = target_method
        source_row["method_display"] = METHOD_DISPLAY.get(target_method, target_method)
        source_row["data_source_method"] = source_method
        source_row["data_source_display"] = source_row.get(
            "data_source_display",
            METHOD_DISPLAY.get(source_method, source_method),
        )
        source_row["is_replaced_point"] = True
        by_key[(target_method, d_model, num_facts)] = source_row

    return _sort_rows(list(by_key.values()))


def _valid_rows_for_plot(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = []
    for r in rows:
        w = _to_float(r.get("param_count"))
        f = _to_float(r.get("num_facts"))
        d = r.get("d_model")
        if np.isfinite(w) and np.isfinite(f) and w > 0 and f > 0 and d != "":
            valid.append(r)
    return valid


def _resolve_methods_present(
    valid_rows: List[Dict[str, Any]],
    method_order: Optional[List[str]] = None,
) -> List[str]:
    present_set = {str(r["method"]) for r in valid_rows}
    order = list(method_order) if method_order else list(METHOD_ORDER)
    methods_present = [m for m in order if m in present_set]
    for method in METHOD_ORDER:
        if method in present_set and method not in methods_present:
            methods_present.append(method)
    for method in sorted(present_set):
        if method not in methods_present:
            methods_present.append(method)
    return methods_present


def _present_methods_and_dims(
    valid_rows: List[Dict[str, Any]],
    method_order: Optional[List[str]] = None,
) -> tuple[List[str], List[int]]:
    methods_present = _resolve_methods_present(valid_rows, method_order=method_order)
    d_values = sorted({int(r["d_model"]) for r in valid_rows})
    return methods_present, d_values


def _method_display_map(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    display_map: Dict[str, str] = {}
    for row in rows:
        method = str(row.get("method", ""))
        label = str(row.get("method_display") or METHOD_DISPLAY.get(method, method))
        display_map.setdefault(method, label)
    return display_map


def _plot_capacity_series(
    ax: plt.Axes,
    valid_rows: List[Dict[str, Any]],
    *,
    alpha: float = 0.9,
    method_order: Optional[List[str]] = None,
) -> tuple[List[str], List[int]]:
    methods_present, d_values = _present_methods_and_dims(valid_rows, method_order=method_order)

    for method in methods_present:
        color = METHOD_COLOR.get(method, "#666666")
        method_rows = [r for r in valid_rows if r["method"] == method]
        for d_model in d_values:
            sub = [r for r in method_rows if int(r["d_model"]) == d_model]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: _to_float(r["param_count"]))
            x = np.asarray([_to_float(r["param_count"]) for r in sub], dtype=np.float64)
            y = np.asarray([_to_float(r["num_facts"]) for r in sub], dtype=np.float64)
            ax.plot(
                x,
                y,
                marker=D_MODEL_MARKER.get(d_model, "o"),
                linestyle=D_MODEL_LINESTYLE.get(d_model, "-"),
                color=color,
                linewidth=2.0,
                markersize=math.sqrt(80),
                markeredgewidth=1.0,
                markeredgecolor="black",
                alpha=alpha,
            )

    return methods_present, d_values


def _set_capacity_axes(ax: plt.Axes, *, title: str) -> None:
    try:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
    except TypeError:
        ax.set_xscale("log", basex=2)
        ax.set_yscale("log", basey=2)
    ax.set_xlabel("Number of Parameters (W)")
    ax.set_ylabel("Number of Facts (F)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--", which="major")


def _add_method_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    methods_present: List[str],
    *,
    method_display_map: Optional[Dict[str, str]] = None,
    labels_override: Optional[Dict[str, str]] = None,
    outside_right: bool = False,
) -> None:
    handles = []
    labels = []
    for method in methods_present:
        handles.append(
            Line2D(
                [0],
                [0],
                color=METHOD_COLOR.get(method, "#666666"),
                linestyle="-",
                marker="o",
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=8,
            )
        )
        labels.append((labels_override or {}).get(method, METHOD_DISPLAY.get(method, method)))
        if method_display_map is not None and method not in (labels_override or {}):
            labels[-1] = method_display_map.get(method, labels[-1])
    if not handles:
        return

    if outside_right:
        fig.legend(
            handles=handles,
            labels=labels,
            title="Method",
            loc="upper right",
            bbox_to_anchor=(0.995, 0.98),
            fontsize=9,
            title_fontsize=10,
            framealpha=0.9,
            borderaxespad=0.0,
        )
    else:
        legend = ax.legend(
            handles=handles,
            labels=labels,
            title="Method",
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02),
            fontsize=9,
            title_fontsize=10,
            framealpha=0.9,
        )
        ax.add_artist(legend)


def _add_d_model_legend(ax: plt.Axes, d_values: List[int]) -> None:
    handles = []
    labels = []
    for d_model in d_values:
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=D_MODEL_LINESTYLE.get(d_model, "-"),
                marker=D_MODEL_MARKER.get(d_model, "o"),
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=8,
            )
        )
        labels.append(f"d={d_model}")
    if handles:
        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            fontsize=9,
            framealpha=0.9,
        )


def _fit_loglog_power_law(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    if len(x) < 2 or len(y) < 2:
        return None
    if np.min(x) <= 0 or np.min(y) <= 0:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    try:
        slope, intercept = np.polyfit(np.log2(x), np.log2(y), 1)
    except Exception:
        return None
    return float(slope), float(2.0 ** intercept)


def _format_power_law_formula(slope: float, coeff: float) -> str:
    return f"F ~ {coeff:.2g} * W^{slope:.2f}"


def plot_f_vs_w(
    rows: List[Dict[str, Any]],
    output_path: Path,
    show: bool,
    *,
    method_order: Optional[List[str]] = None,
    title: str = "MLP Capacity Scaling",
) -> None:
    valid = _valid_rows_for_plot(rows)

    if not valid:
        raise ValueError("No valid rows with positive F and W to plot.")

    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 6.3))

    methods_present, d_values = _plot_capacity_series(
        ax,
        valid,
        alpha=0.92,
        method_order=method_order,
    )
    method_display_map = _method_display_map(valid)
    _set_capacity_axes(ax, title=title)
    _add_method_legend(
        fig,
        ax,
        methods_present,
        method_display_map=method_display_map,
        outside_right=False,
    )
    _add_d_model_legend(ax, d_values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_f_vs_w_with_method_fits(
    rows: List[Dict[str, Any]],
    output_path: Path,
    show: bool,
    *,
    method_order: Optional[List[str]] = None,
    title: str = "MLP Capacity Scaling (with Heuristic Fits)",
) -> None:
    """Save a companion plot with heuristic log-log fits overlaid by method."""
    valid = _valid_rows_for_plot(rows)

    if not valid:
        raise ValueError("No valid rows with positive F and W to plot.")

    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(10.6, 6.4))

    methods_present, d_values = _plot_capacity_series(
        ax,
        valid,
        alpha=0.75,
        method_order=method_order,
    )
    method_display_map = _method_display_map(valid)
    _set_capacity_axes(ax, title=title)
    _add_d_model_legend(ax, d_values)

    method_fit_labels: Dict[str, str] = {}
    print("[plot_f_vs_w_with_method_fits] Heuristic fits:")
    for method in methods_present:
        method_rows = [r for r in valid if r["method"] == method]
        x = np.asarray([_to_float(r["param_count"]) for r in method_rows], dtype=np.float64)
        y = np.asarray([_to_float(r["num_facts"]) for r in method_rows], dtype=np.float64)
        fit = _fit_loglog_power_law(x, y)
        base_label = method_display_map.get(method, METHOD_DISPLAY.get(method, method))
        if fit is None:
            print(f"  {base_label}: n/a")
            method_fit_labels[method] = f"{base_label} (fit: n/a)"
            continue

        slope, coeff = fit
        fit_x = np.geomspace(np.min(x), np.max(x), num=256)
        fit_y = coeff * np.power(fit_x, slope)
        ax.plot(
            fit_x,
            fit_y,
            linestyle=":",
            linewidth=2.6,
            color=METHOD_COLOR.get(method, "#666666"),
            alpha=0.95,
        )
        fit_formula = _format_power_law_formula(slope, coeff)
        method_fit_labels[method] = f"{base_label} ({fit_formula})"
        print(f"  {base_label}: {fit_formula}")

    _add_method_legend(
        fig,
        ax,
        methods_present,
        method_display_map=method_display_map,
        labels_override=method_fit_labels,
        outside_right=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(right=0.63)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Section 4 MLP capacity sweep (F vs W) from binary-search pickles, "
            "including a companion fit-overlay plot."
        )
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        help="Base run directory (e.g. ./artifacts/mlp_capacity).",
    )
    parser.add_argument(
        "--input-points-csv",
        default=None,
        help=(
            "Load compact paper points directly instead of collecting "
            "binary-search pickles. Accepts either figure_points.csv "
            "(num_parameters column) or summary.csv (param_count column)."
        ),
    )
    parser.add_argument(
        "--extra-base-dir",
        action="append",
        default=[],
        help=(
            "Additional run directory to merge into the plot/summary. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots/summary (default: <base_dir>/plots).",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Summary CSV path (default: <output_dir>/summary.csv).",
    )
    parser.add_argument(
        "--points-csv",
        default=None,
        help="Per-point CSV path for downstream figure tooling (default: <output_dir>/figure_points.csv).",
    )
    parser.add_argument(
        "--include-method",
        action="append",
        default=[],
        help="Method key to include in the plot/export. Can be passed multiple times.",
    )
    parser.add_argument(
        "--replace-point",
        action="append",
        default=[],
        help=(
            "Replace a target method's point with another method's data at the same "
            "(d_model, num_facts). Format: target_method,d_model,num_facts,source_method."
        ),
    )
    parser.add_argument(
        "--method-display-override",
        action="append",
        default=[],
        help="Override plotted/exported label for a method. Format: METHOD=LABEL.",
    )
    parser.add_argument(
        "--method-order",
        action="append",
        default=[],
        help="Method key order to use for plotting/legend. Can be passed multiple times.",
    )
    parser.add_argument(
        "--title",
        default="MLP Capacity Scaling",
        help="Title for the main F-vs-W plot. Literal '\\n' sequences render as line breaks.",
    )
    parser.add_argument(
        "--fit-title",
        default=None,
        help=(
            "Title for the fit-overlay plot "
            "(default: '<title> (with Heuristic Fits)')."
        ),
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plot window.")
    parser.add_argument("--verbose", action="store_true", help="Print extra processing details.")
    args = parser.parse_args()

    if args.base_dir is None and args.input_points_csv is None:
        parser.error("base_dir is required unless --input-points-csv is provided.")

    base_dir = Path(args.base_dir).resolve() if args.base_dir is not None else None
    input_points_csv = Path(args.input_points_csv).resolve() if args.input_points_csv else None
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (base_dir / "plots" if base_dir is not None else input_points_csv.parent / "plots")
    )
    summary_csv = (
        Path(args.summary_csv).resolve() if args.summary_csv else output_dir / "summary.csv"
    )
    points_csv = (
        Path(args.points_csv).resolve() if args.points_csv else output_dir / "figure_points.csv"
    )
    plot_path = output_dir / "f_vs_w.png"
    fit_plot_path = output_dir / "f_vs_w_with_fits.png"

    base_rows = (
        load_rows_from_points_csv(input_points_csv)
        if input_points_csv is not None
        else collect_rows(base_dir, verbose=args.verbose)
    )
    extra_rows = []
    for extra_base_dir_str in args.extra_base_dir:
        extra_base_dir = Path(extra_base_dir_str).resolve()
        rows_i = collect_rows(extra_base_dir, verbose=args.verbose)
        if args.verbose:
            print(f"Merging {len(rows_i)} rows from extra base dir: {extra_base_dir}")
        extra_rows.append(rows_i)
    rows = combine_rows(base_rows, *extra_rows)
    rows = apply_point_replacements(rows, args.replace_point)
    if args.include_method:
        rows = filter_rows_by_method(rows, args.include_method)
    display_overrides = parse_method_display_overrides(args.method_display_override)
    rows = apply_method_display_overrides(rows, display_overrides)
    write_summary_csv(rows, summary_csv)
    print(f"Wrote summary CSV: {summary_csv}")
    write_points_csv(rows, points_csv)
    print(f"Wrote points CSV: {points_csv}")

    if not rows:
        print("No achieved capacity rows found; skipping plot.")
        return

    method_order = args.method_order if args.method_order else None
    title = args.title.replace("\\n", "\n")
    default_fit_title = (
        f"{args.title} (with Heuristic Fits)"
        if args.title
        else "MLP Capacity Scaling (with Heuristic Fits)"
    )
    fit_title = (args.fit_title or default_fit_title).replace("\\n", "\n")
    plot_f_vs_w(
        rows,
        plot_path,
        show=not args.no_show,
        method_order=method_order,
        title=title,
    )
    print(f"Wrote plot: {plot_path}")
    plot_f_vs_w_with_method_fits(
        rows,
        fit_plot_path,
        show=not args.no_show,
        method_order=method_order,
        title=fit_title,
    )
    print(f"Wrote fit plot: {fit_plot_path}")


if __name__ == "__main__":
    main()
