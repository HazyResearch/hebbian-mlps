"""Plot Section 3.1 sweep outputs (attention noise vs junk length).

Loads latest `grid_search_results_*.pkl` per `(d_model, num_facts, junk_len)` cell and
generates a compact validation figure.
"""

from __future__ import annotations

import argparse
import io
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


_D_N_RE = re.compile(r"^d(?P<d>\d+)_n(?P<n>\d+)$")
_JUNK_RE = re.compile(r"^junk_len_(?P<j>\d+)$")


class _CompatUnpickler(pickle.Unpickler):
    """Unpickler resilient to `__main__` class references and CUDA tensors."""

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
        # Sweep results produced via script entrypoints often pickle local config
        # classes as "__main__.ClassName"; they are not needed for plotting.
        if module == "__main__":
            return self._dummy_cls(module, name)
        # Map CUDA storages to CPU on load.
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _load_pickle_compat(path: Path) -> Any:
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


def _unwrap_result_object(obj: Any) -> Dict[str, Any] | None:
    """Best-effort unwrapping for sweep pickle payloads."""
    current = obj
    for _ in range(6):
        if current is None:
            return None
        if hasattr(current, "result"):
            current = current.result
            continue
        if isinstance(current, dict) and "results" in current:
            current = current["results"]
            continue
        if isinstance(current, dict) and "result" in current and not (
            "best_acc" in current or "attn_noise_l2_floor" in current
        ):
            current = current["result"]
            continue
        break
    return current if isinstance(current, dict) else None


def _parse_meta(path: Path) -> Dict[str, int]:
    d_model = -1
    num_facts = -1
    junk_len = -1
    for part in path.parts:
        m = _D_N_RE.match(part)
        if m:
            d_model = int(m.group("d"))
            num_facts = int(m.group("n"))
            continue
        m = _JUNK_RE.match(part)
        if m:
            junk_len = int(m.group("j"))
    return {"d_model": d_model, "num_facts": num_facts, "junk_len": junk_len}


def _latest_pickles_by_cell(base_dir: Path) -> List[Path]:
    latest: Dict[Path, Path] = {}
    for p in sorted(base_dir.rglob("grid_search_results_*.pkl")):
        prev = latest.get(p.parent)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            latest[p.parent] = p
    return sorted(latest.values())


def load_rows(base_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pkl_path in _latest_pickles_by_cell(base_dir):
        data = _load_pickle_compat(pkl_path)
        payload = _unwrap_result_object(data)
        if payload is None:
            continue
        # Preferred source: all per-key max-L2 values pooled across seeds.
        floor_key_values_all_seeds = payload.get("attn_noise_l2_per_key_max_values_all_seeds", [])
        if not isinstance(floor_key_values_all_seeds, list):
            floor_key_values_all_seeds = []
        floor_key_values_all_seeds = [
            float(v)
            for v in floor_key_values_all_seeds
            if isinstance(v, (int, float)) and np.isfinite(float(v))
        ]

        meta = _parse_meta(pkl_path.relative_to(base_dir))
        rows.append(
            {
                **meta,
                "best_acc": float(payload.get("best_acc", float("nan"))),
                "attn_noise_l2_floor": float(payload.get("attn_noise_l2_floor", float("nan"))),
                "attn_noise_l2_mean": float(payload.get("attn_noise_l2_mean", float("nan"))),
                "attn_noise_l2_floor_key_values_all_seeds": floor_key_values_all_seeds,
                "path": str(pkl_path.relative_to(base_dir)),
            }
        )
    rows.sort(key=lambda r: (r["d_model"], r["num_facts"], r["junk_len"]))
    return rows


def plot_rows(rows: List[Dict[str, Any]], output_dir: Path, show: bool) -> None:
    if not rows:
        raise ValueError("No rows found to plot.")

    output_dir.mkdir(parents=True, exist_ok=True)
    cfgs = sorted({(r["d_model"], r["num_facts"]) for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_noise, ax_acc = axes

    for d_model, num_facts in cfgs:
        sub = [r for r in rows if r["d_model"] == d_model and r["num_facts"] == num_facts]
        sub = sorted(sub, key=lambda r: r["junk_len"])
        x = np.asarray([r["junk_len"] for r in sub], dtype=np.float64)
        y_floor = np.asarray([r["attn_noise_l2_floor"] for r in sub], dtype=np.float64)
        y_acc = np.asarray([r["best_acc"] for r in sub], dtype=np.float64)
        label = f"d={d_model}, F={num_facts}"

        ax_noise.plot(x, y_floor, marker="o", linewidth=1.8, label=label)
        ax_acc.plot(x, y_acc, marker="o", linewidth=1.8, label=label)

    for ax in (ax_noise, ax_acc):
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("junk_len")

    ax_noise.set_ylabel("attention noise floor (L2)")
    ax_noise.set_title("Noise Floor vs Junk Length")
    ax_noise.legend(fontsize=8)

    ax_acc.set_ylabel("best transformer accuracy")
    ax_acc.set_title("Accuracy vs Junk Length")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.legend(fontsize=8)

    plt.tight_layout()

    out_png = output_dir / "section3_1_attention_noise_validation.png"
    out_pdf = output_dir / "section3_1_attention_noise_validation.pdf"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


def plot_l2_floor_boxplots(rows: List[Dict[str, Any]], output_dir: Path, show: bool) -> None:
    if not rows:
        raise ValueError("No rows found to plot.")

    output_dir.mkdir(parents=True, exist_ok=True)
    cfgs = sorted({(r["d_model"], r["num_facts"]) for r in rows})
    ncols = max(1, len(cfgs))
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 4.5), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, (d_model, num_facts) in zip(axes, cfgs):
        sub = [r for r in rows if r["d_model"] == d_model and r["num_facts"] == num_facts]
        sub = sorted(sub, key=lambda r: r["junk_len"])

        box_values: List[List[float]] = []
        labels: List[str] = []
        for r in sub:
            vals = list(r.get("attn_noise_l2_floor_key_values_all_seeds", []))
            vals = [float(v) for v in vals if np.isfinite(v)]
            if not vals:
                continue
            box_values.append(vals)
            labels.append(str(int(r["junk_len"])))

        if not box_values:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"d={d_model}, F={num_facts}")
            ax.set_xlabel("junk_len")
            ax.grid(True, axis="y", alpha=0.3)
            continue

        bp = ax.boxplot(
            box_values,
            patch_artist=True,
            widths=0.6,
            whis=1.5,
            showfliers=True,
        )
        for box in bp["boxes"]:
            box.set(facecolor="#95c8d8", edgecolor="#1f3b4d", alpha=0.8)
        for med in bp["medians"]:
            med.set(color="#b03a2e", linewidth=1.8)
        for whisk in bp["whiskers"]:
            whisk.set(color="#1f3b4d", linewidth=1.2)
        for cap in bp["caps"]:
            cap.set(color="#1f3b4d", linewidth=1.2)

        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("junk_len")
        ax.set_title(f"d={d_model}, F={num_facts}")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("attention noise floor (L2)")
    fig.suptitle("Per-Key Max Attention Noise (All Keys Across Seeds)", y=1.02)
    plt.tight_layout()

    out_png = output_dir / "section3_1_attention_noise_floor_boxplot.png"
    out_pdf = output_dir / "section3_1_attention_noise_floor_boxplot.pdf"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Section 3.1 attention-noise sweep outputs.")
    parser.add_argument("base_dir", type=str, help="Sweep base directory.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <base_dir>/plots).",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plot window.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else base_dir / "plots"

    rows = load_rows(base_dir)
    print(f"Loaded {len(rows)} sweep cells from: {base_dir}")
    plot_rows(rows, output_dir=output_dir, show=not args.no_show)
    plot_l2_floor_boxplots(rows, output_dir=output_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
