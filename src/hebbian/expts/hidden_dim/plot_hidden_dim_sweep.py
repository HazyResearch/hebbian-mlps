"""
Plot results from hidden_dim_sweep.py.

Reads checkpoints from {base_dir}/m{hidden_dim}/seed_{seed}/checkpoints/last_model.pt
and produces a 1×2 figure:

  Left  — MLP accuracy and transformer accuracy vs # parameters
  Right — Minimum margin (γ_min) vs # parameters

Both panels show mean ± std shading over seeds.

Usage:
    # Auto-find the latest generated sweep directory
    python -m hebbian.expts.hidden_dim.plot_hidden_dim_sweep

    # Explicit base dir
    python -m hebbian.expts.hidden_dim.plot_hidden_dim_sweep 'base_dir="./artifacts/hidden_dim/run_..."'
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from hebbian.config import main as pydra_main, pydraclass


@pydraclass
class PlotConfig:
    """Configuration for hidden-dim sweep plotting."""

    base_dir: Optional[str] = None   # explicit sweep root; if None, auto-find latest
    search_root: str = "./artifacts/hidden_dim"
    output_dir: str = "./artifacts/hidden_dim/plots"
    title_suffix: str = ""


# ---------------------------------------------------------------------------
# Auto-find latest sweep directory
# ---------------------------------------------------------------------------

def _find_latest_sweep_dir(search_root: str) -> str:
    patterns = [
        os.path.join(search_root, "run_*"),
    ]
    dirs = sorted((d for p in patterns for d in glob.glob(p)), key=os.path.getmtime)
    if not dirs:
        raise FileNotFoundError(
            f"No generated hidden-dimension runs found under: {search_root}"
        )
    return dirs[-1]


# ---------------------------------------------------------------------------
# Load checkpoints
# ---------------------------------------------------------------------------

def _load_sweep_results(base_dir: str) -> dict:
    """
    Walk base_dir/m*/seed_*/checkpoints/last_model.pt and collect metrics.

    Returns a dict keyed by hidden_dim (int), each value is a list of per-seed dicts:
        {
            "param_count": int,
            "hidden_dim": int,
            "mlp_accuracy": float,
            "gamma_min": float,
            "transformer_accuracy": float,   # best_acc during training
        }
    """
    pattern = os.path.join(base_dir, "m*", "seed_*", "checkpoints", "last_model.pt")
    ckpt_paths = sorted(glob.glob(pattern))
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints found matching: {pattern}"
        )
    print(f"Found {len(ckpt_paths)} checkpoints under {base_dir}")

    by_dim = defaultdict(list)

    for path in ckpt_paths:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [WARN] Could not load {path}: {e}")
            continue

        mlp_metrics    = ckpt.get("mlp_metrics", {})
        best_acc       = ckpt.get("best_acc", float("nan"))
        best_train_acc = ckpt.get("best_train_acc", float("nan"))
        mlp_str        = ckpt.get("mlp_str", None)
        gpt_str        = ckpt.get("gpt_str", None)

        hidden_dim   = mlp_metrics.get("hidden_dim", None)
        param_count  = mlp_metrics.get("param_count", None)
        mlp_accuracy = mlp_metrics.get("final_accuracy", mlp_metrics.get("accuracy", float("nan")))
        gamma_min    = mlp_metrics.get("gamma_min", float("nan"))

        # Infer hidden_dim from directory name if not stored in metrics
        if hidden_dim is None:
            m = re.search(r"[/\\]m(\d+)[/\\]", path)
            hidden_dim = int(m.group(1)) if m else -1

        by_dim[hidden_dim].append({
            "param_count":               param_count,
            "hidden_dim":                hidden_dim,
            "mlp_accuracy":              mlp_accuracy,
            "gamma_min":                 gamma_min,
            "transformer_accuracy":      best_acc,
            "transformer_train_accuracy": best_train_acc,
            "mlp_str":                   mlp_str,
            "gpt_str":                   gpt_str,
        })

    return dict(by_dim)


# ---------------------------------------------------------------------------
# Aggregate over seeds
# ---------------------------------------------------------------------------

def _aggregate(by_dim: dict) -> list:
    """
    Aggregate per-seed entries.

    Returns list of dicts sorted by param_count:
        {
            "hidden_dim", "param_count",
            "mlp_accuracy_mean", "mlp_accuracy_std",
            "transformer_accuracy_mean", "transformer_accuracy_std",
            "gamma_min_mean", "gamma_min_std",
            "n_seeds",
        }
    """
    rows = []
    for hidden_dim, entries in sorted(by_dim.items()):
        def _stat(key):
            vals = [e[key] for e in entries
                    if not (isinstance(e[key], float) and np.isnan(e[key]))
                    and e[key] is not None]
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals))

        param_count = next(
            (e["param_count"] for e in entries if e["param_count"] is not None),
            None
        )

        def _max(key):
            vals = [e[key] for e in entries
                    if not (isinstance(e[key], float) and np.isnan(e[key]))
                    and e[key] is not None]
            return float(np.max(vals)) if vals else float("nan")

        mlp_mean, mlp_std           = _stat("mlp_accuracy")
        tfm_mean, tfm_std           = _stat("transformer_accuracy")
        tfm_train_mean, tfm_train_std = _stat("transformer_train_accuracy")
        gmin_mean, gmin_std         = _stat("gamma_min")

        # Collect architecture strings from first entry that has them
        mlp_str = next((e["mlp_str"] for e in entries if e.get("mlp_str") is not None), None)
        gpt_str = next((e["gpt_str"] for e in entries if e.get("gpt_str") is not None), None)

        rows.append({
            "hidden_dim":                        hidden_dim,
            "param_count":                       param_count,
            "mlp_accuracy_mean":                 mlp_mean,
            "mlp_accuracy_std":                  mlp_std,
            "mlp_accuracy_max":                  _max("mlp_accuracy"),
            "transformer_accuracy_mean":         tfm_mean,
            "transformer_accuracy_std":          tfm_std,
            "transformer_accuracy_max":          _max("transformer_accuracy"),
            "transformer_train_accuracy_mean":   tfm_train_mean,
            "transformer_train_accuracy_std":    tfm_train_std,
            "transformer_train_accuracy_max":    _max("transformer_train_accuracy"),
            "gamma_min_mean":                    gmin_mean,
            "gamma_min_std":                     gmin_std,
            "n_seeds":                           len(entries),
            "mlp_str":                           mlp_str,
            "gpt_str":                           gpt_str,
        })

    # Sort by param_count (or hidden_dim as fallback)
    rows.sort(key=lambda r: r["param_count"] if r["param_count"] is not None else r["hidden_dim"])
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(rows: list, output_dir: str, title_suffix: str = "") -> None:
    param_counts = [r["param_count"] for r in rows]
    hidden_dims  = [r["hidden_dim"]  for r in rows]

    # Use param_count on x-axis; fall back to hidden_dim
    x      = np.array([p if p is not None else h for p, h in zip(param_counts, hidden_dims)], dtype=float)
    x_label = "W"

    mlp_acc_mean      = np.array([r["mlp_accuracy_mean"]               for r in rows])
    mlp_acc_std       = np.array([r["mlp_accuracy_std"]                for r in rows])
    tfm_acc_mean      = np.array([r["transformer_accuracy_mean"]       for r in rows])
    tfm_acc_std       = np.array([r["transformer_accuracy_std"]        for r in rows])
    tfm_train_mean    = np.array([r["transformer_train_accuracy_mean"] for r in rows])
    tfm_train_std     = np.array([r["transformer_train_accuracy_std"]  for r in rows])
    gmin_mean         = np.array([r["gamma_min_mean"]                  for r in rows])
    gmin_std          = np.array([r["gamma_min_std"]                   for r in rows])
    n_seeds           = rows[0]["n_seeds"] if rows else "?"

    os.makedirs(output_dir, exist_ok=True)

    # Find first x where any seed first reached 100% accuracy (best case over seeds)
    mlp_acc_max       = np.array([r["mlp_accuracy_max"]               for r in rows])
    tfm_acc_max       = np.array([r["transformer_accuracy_max"]       for r in rows])
    tfm_train_acc_max = np.array([r["transformer_train_accuracy_max"] for r in rows])
    mlp_100_idx       = next((i for i, v in enumerate(mlp_acc_max)       if v >= 1.0), None)
    tfm_100_idx       = next((i for i, v in enumerate(tfm_acc_max)       if v >= 1.0), None)
    tfm_train_100_idx = next((i for i, v in enumerate(tfm_train_acc_max) if v >= 1.0), None)
    mlp_100_x         = x[mlp_100_idx]       if mlp_100_idx       is not None else None
    tfm_100_x         = x[tfm_100_idx]       if tfm_100_idx       is not None else None
    tfm_train_100_x   = x[tfm_train_100_idx] if tfm_train_100_idx is not None else None

    def _add_vlines(ax):
        if mlp_100_x is not None:
            ax.axvline(mlp_100_x, color="tab:blue",   linestyle="--", alpha=0.7,
                       linewidth=1.2, label="MLP hits 100% accuracy")
        if tfm_100_x is not None:
            ax.axvline(tfm_100_x, color="tab:orange", linestyle="--", alpha=0.7,
                       linewidth=1.2, label="Transformer eval hits 100% accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: Accuracy vs #params ----
    ax = axes[0]
    ax.plot(x, mlp_acc_mean, "o-", color="tab:blue",   label="MLP standalone accuracy")
    ax.fill_between(x,
                    mlp_acc_mean - mlp_acc_std,
                    mlp_acc_mean + mlp_acc_std,
                    color="tab:blue", alpha=0.2)
    ax.plot(x, tfm_acc_mean, "s--", color="tab:orange", label="Transformer best eval accuracy")
    ax.fill_between(x,
                    tfm_acc_mean - tfm_acc_std,
                    tfm_acc_mean + tfm_acc_std,
                    color="tab:orange", alpha=0.2)
    _add_vlines(ax)
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs # MLP parameters")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Right: γ_min vs #params ----
    ax = axes[1]
    ax.plot(x, gmin_mean, "o-", color="tab:green", label=r"$\gamma_{\min}$ (MLP, normalised)")
    ax.fill_between(x,
                    gmin_mean - gmin_std,
                    gmin_mean + gmin_std,
                    color="tab:green", alpha=0.2)
    ax.axhline(0, color="k", linestyle="--", alpha=0.4, linewidth=0.8)
    _add_vlines(ax)
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\gamma_{\min}$")
    ax.set_title(r"$\gamma_{\min}$ vs # MLP parameters")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    stem = "hidden_dim_sweep"
    for ext in (".pdf", ".png"):
        out_path = os.path.join(output_dir, stem + ext)
        kw = {"bbox_inches": "tight"}
        if ext == ".png":
            kw["dpi"] = 150
        plt.savefig(out_path, **kw)
        print(f"Saved: {out_path}")

    plt.close()

    # Print summary table
    print(f"\n{'hidden_dim':>12}  {'params':>10}  {'mlp_acc':>9}  {'tfm_eval':>9}  {'tfm_train':>10}  {'gamma_min':>10}")
    print("-" * 72)
    for r in rows:
        pc = r["param_count"] if r["param_count"] is not None else "?"
        print(
            f"{r['hidden_dim']:>12}  {str(pc):>10}  "
            f"{r['mlp_accuracy_mean']:>9.4f}  "
            f"{r['transformer_accuracy_mean']:>9.4f}  "
            f"{r['transformer_train_accuracy_mean']:>10.4f}  "
            f"{r['gamma_min_mean']:>10.4f}"
        )

    # Print architecture strings (from the first available row)
    mlp_str = next((r["mlp_str"] for r in rows if r.get("mlp_str") is not None), None)
    gpt_str = next((r["gpt_str"] for r in rows if r.get("gpt_str") is not None), None)
    if mlp_str is not None:
        print("\n" + "=" * 60)
        print("MLP Architecture:")
        print("=" * 60)
        print(mlp_str)
    if gpt_str is not None:
        print("\n" + "=" * 60)
        print("Transformer Architecture:")
        print("=" * 60)
        print(gpt_str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@pydra_main(PlotConfig)
def main(config: PlotConfig):
    if config.base_dir is None:
        config.base_dir = _find_latest_sweep_dir(config.search_root)
    print(f"Reading sweep from: {config.base_dir}")

    by_dim = _load_sweep_results(config.base_dir)
    rows   = _aggregate(by_dim)

    if not rows:
        print("No valid results found — nothing to plot.")
        return

    _plot(rows, config.output_dir, config.title_suffix)
    print(f"\nDone. Plots saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
