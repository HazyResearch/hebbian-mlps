"""
Plot per-key margin distributions vs combined accuracy.

Loads checkpoints from an attn_pretrain_hidden_dim_sweep, or an NPZ exported
from those checkpoints, and produces:

  Panel 1 — Violin plot: x = combined accuracy per hidden_dim,
             y = per-key margin γ_i distribution.
             Overlaid: γ_min line, γ=0 reference, and
             fraction(γ_i > 0) annotated per violin.

  Panel 2 — Fraction of keys with γ_i > τ vs combined accuracy,
             for several τ values (τ = 0, 0.1, 0.2, 0.3).
             Tests the claim: frac(γ_i > τ) ≈ accuracy.

  Panel 3 — MLP standalone accuracy per hidden_dim (bar chart).

Usage:
    python -m hebbian.expts.hidden_dim.plot_margin_distribution \\
        'base_dir="./artifacts/hidden_dim/run_..."'
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hebbian.config import main as pydra_main, pydraclass


@pydraclass
class PlotConfig:
    base_dir: Optional[str] = None
    search_root: str = "./artifacts/hidden_dim"
    output_dir: str = "./artifacts/hidden_dim/plots"
    tau_values: list = None   # τ thresholds for fraction plot; default [0, 0.1, 0.2, 0.3]
    points_npz: Optional[str] = None


# ---------------------------------------------------------------------------
# Load sweep checkpoints
# ---------------------------------------------------------------------------

def _load_sweep(base_dir: str) -> list[dict]:
    """Load all seed checkpoints; return list sorted by hidden_dim."""
    pattern = os.path.join(base_dir, "m*/seed_*/checkpoints/last_model.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found: {pattern}")

    print(f"Found {len(paths)} checkpoints under {base_dir}")

    rows = []
    for p in paths:
        m = re.search(r"/m(\d+)/seed_(\d+)/", p)
        if m is None:
            continue
        hidden_dim = int(m.group(1))
        seed = int(m.group(2))
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [WARN] Could not load {p}: {e}")
            continue

        mlp_metrics = ckpt.get("mlp_metrics", {})
        per_key_margins = mlp_metrics.get("per_key_margins", None)
        if per_key_margins is None:
            print(f"  [WARN] No per_key_margins in {p} — re-run sweep with updated script")
            continue

        rows.append({
            "hidden_dim":    hidden_dim,
            "seed":          seed,
            "combined_acc":  ckpt.get("best_acc", float("nan")),
            "attn_acc":      ckpt.get("best_train_acc", float("nan")),
            "mlp_accuracy":  mlp_metrics.get("accuracy",
                             mlp_metrics.get("final_accuracy", float("nan"))),
            "gamma_min":     mlp_metrics.get("gamma_min", float("nan")),
            "param_count":   mlp_metrics.get("param_count", None),
            "per_key_margins": np.asarray(per_key_margins, dtype=float),
        })

    rows.sort(key=lambda r: r["hidden_dim"])
    return rows


def _load_points_npz(points_npz: str) -> list[dict]:
    """Load compact extracted checkpoint fields for plot-only reproduction."""
    payload = np.load(points_npz, allow_pickle=True)
    required = [
        "hidden_dim",
        "seed",
        "combined_acc",
        "attn_acc",
        "mlp_accuracy",
        "gamma_min",
        "param_count",
        "per_key_margins",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing required NPZ arrays in {points_npz}: {missing}")

    rows = []
    n_rows = len(payload["hidden_dim"])
    for idx in range(n_rows):
        rows.append(
            {
                "hidden_dim": int(payload["hidden_dim"][idx]),
                "seed": int(payload["seed"][idx]),
                "combined_acc": float(payload["combined_acc"][idx]),
                "attn_acc": float(payload["attn_acc"][idx]),
                "mlp_accuracy": float(payload["mlp_accuracy"][idx]),
                "gamma_min": float(payload["gamma_min"][idx]),
                "param_count": float(payload["param_count"][idx]),
                "per_key_margins": np.asarray(payload["per_key_margins"][idx], dtype=float),
            }
        )
    rows.sort(key=lambda r: (r["hidden_dim"], r["seed"]))
    if not rows:
        raise ValueError(f"No margin rows found in NPZ: {points_npz}")
    print(f"Loaded {len(rows)} compact margin records from {points_npz}")
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    """Average over seeds for each hidden_dim."""
    from collections import defaultdict
    by_dim = defaultdict(list)
    for r in rows:
        by_dim[r["hidden_dim"]].append(r)

    agg = []
    for hd in sorted(by_dim):
        group = by_dim[hd]
        margins_all = np.concatenate([g["per_key_margins"] for g in group])
        agg.append({
            "hidden_dim":    hd,
            "param_count":   group[0]["param_count"],
            "combined_acc":  np.mean([g["combined_acc"] for g in group]),
            "attn_acc":      np.mean([g["attn_acc"]     for g in group]),
            "mlp_accuracy":  np.mean([g["mlp_accuracy"] for g in group]),
            "gamma_min":     np.min([g["gamma_min"]     for g in group]),
            "per_key_margins": margins_all,   # pooled across seeds
            "n_seeds":       len(group),
        })
    return agg


# ---------------------------------------------------------------------------
# Fit τ*
# ---------------------------------------------------------------------------

def _fit_tau(agg: list[dict]) -> tuple[float, float]:
    """Find τ* minimising MSE between frac(γ_i > τ) and combined_acc."""
    combined_accs = np.array([r["combined_acc"] for r in agg])
    margins_list  = [r["per_key_margins"] for r in agg]

    taus = np.linspace(-1.0, 1.0, 2001)
    mses = np.array([
        np.mean((np.array([(m > t).mean() for m in margins_list]) - combined_accs) ** 2)
        for t in taus
    ])
    best_idx = int(np.argmin(mses))
    return float(taus[best_idx]), float(mses[best_idx])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(agg: list[dict], output_dir: str, tau_values: list[float],
          tau_star: float) -> None:
    os.makedirs(output_dir, exist_ok=True)

    n = len(agg)
    combined_accs = [r["combined_acc"] for r in agg]
    hidden_dims   = [r["hidden_dim"]   for r in agg]
    mlp_accs      = [r["mlp_accuracy"] for r in agg]
    gamma_mins    = [r["gamma_min"]    for r in agg]

    x_pos = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ----------------------------------------------------------------
    # Panel 1: Violin of per-key margin distributions
    # ----------------------------------------------------------------
    ax = axes[0]
    data = [r["per_key_margins"] for r in agg]

    parts = ax.violinplot(data, positions=x_pos, showmedians=True,
                          showextrema=False, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("tab:blue")
        pc.set_alpha(0.5)
    parts["cmedians"].set_color("tab:blue")
    parts["cmedians"].set_linewidth(2)

    # γ_min markers
    ax.scatter(x_pos, gamma_mins, color="tab:red", zorder=5,
               s=40, label=r"$\gamma_{\min}$")

    # γ = 0 reference
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label=r"$\gamma = 0$")

    # Fitted τ* line
    ax.axhline(tau_star, color="tab:orange", linestyle="-", linewidth=2, alpha=0.85,
               label=rf"fitted $\tau^*={tau_star:.3f}$")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{a:.2f}" for a in combined_accs], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Transformer Block Accuracy")
    ax.set_ylabel(r"Per-key margin $\gamma_i$")
    ax.set_title(r"Per-key margin distribution vs combined accuracy")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ----------------------------------------------------------------
    # Panel 2: frac(γ_i > τ) vs combined accuracy
    # ----------------------------------------------------------------
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tau_values)))

    for tau, color in zip(tau_values, colors):
        fracs = [(r["per_key_margins"] > tau).mean() for r in agg]
        ax.plot(combined_accs, fracs, "o-", color=color,
                label=rf"$\tau={tau:.2f}$", linewidth=1.5, markersize=5)

    # Fitted τ* highlighted
    fracs_star_list = [(r["per_key_margins"] > tau_star).mean() for r in agg]
    ax.plot(combined_accs, fracs_star_list, "D-", color="tab:orange", linewidth=2.5,
            markersize=7, zorder=5, label=rf"fitted $\tau^*={tau_star:.3f}$")

    # Identity line
    lim = [min(combined_accs + [0]) - 0.02, max(combined_accs + [1]) + 0.02]
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5, label="identity")
    ax.set_xlim(lim)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Combined accuracy (attn + MLP)")
    ax.set_ylabel(r"Fraction of keys with $\gamma_i > \tau$")
    ax.set_title(r"Frac($\gamma_i > \tau$) vs accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ----------------------------------------------------------------
    # Panel 3: MLP standalone accuracy (bar) + combined accuracy (line)
    # ----------------------------------------------------------------
    ax = axes[2]
    bars = ax.bar(x_pos, mlp_accs, color="tab:blue", alpha=0.6, label="MLP standalone acc")
    ax.plot(x_pos, combined_accs, "s--", color="tab:orange", linewidth=1.5,
            markersize=6, label="Combined acc")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"m={hd}" for hd in hidden_dims], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("MLP vs combined accuracy per hidden_dim")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Per-key margin analysis  |  n_dims={n}  |  "
        f"hidden_dims={hidden_dims}",
        fontsize=10,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"margin_distribution.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

    # Save the violin panel alone
    fig_v, ax_v = plt.subplots(figsize=(7, 5))
    parts = ax_v.violinplot(data, positions=x_pos, showmedians=True,
                            showextrema=False, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("tab:blue")
        pc.set_alpha(0.5)
    parts["cmedians"].set_color("tab:blue")
    parts["cmedians"].set_linewidth(2)
    ax_v.scatter(x_pos, gamma_mins, color="tab:red", zorder=5,
                 s=40, label=r"$\gamma_{\min}$")
    ax_v.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5,
                 label=r"$\gamma = 0$")
    ax_v.axhline(tau_star, color="tab:orange", linestyle="-", linewidth=2, alpha=0.85,
                 label=rf"fitted $\tau^*={tau_star:.3f}$")
    ax_v.set_xticks(x_pos)
    ax_v.set_xticklabels([f"{a:.2f}" for a in combined_accs], rotation=45, ha="right", fontsize=8)
    ax_v.set_xlabel("Transformer Block Accuracy")
    ax_v.set_ylabel(r"Per-key margin $\gamma_i$")
    ax_v.set_title(r"Per-key margin distribution vs combined accuracy")
    ax_v.legend(fontsize=8, loc="lower right")
    ax_v.grid(True, alpha=0.3)
    fig_v.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"margin_violin.{ext}")
        fig_v.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig_v)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_table(agg: list[dict], tau_values: list[float]) -> None:
    header = (
        f"{'hidden_dim':>12}  {'params':>10}  {'mlp_acc':>8}  "
        f"{'combined':>9}  {'gamma_min':>10}  "
        + "  ".join(f"frac(>{t:.2f})" for t in tau_values)
    )
    print("\n" + header)
    print("-" * len(header))
    for r in agg:
        fracs = "  ".join(
            f"{(r['per_key_margins'] > t).mean():>10.4f}" for t in tau_values
        )
        pc = r["param_count"] if r["param_count"] is not None else float("nan")
        print(
            f"{r['hidden_dim']:>12}  {pc:>10.0f}  "
            f"{r['mlp_accuracy']:>8.4f}  "
            f"{r['combined_acc']:>9.4f}  "
            f"{r['gamma_min']:>10.4f}  {fracs}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@pydra_main(PlotConfig)
def main(config: PlotConfig) -> None:
    if config.tau_values is None:
        tau_values = [0.0, 0.1, 0.2, 0.3]
    else:
        tau_values = list(config.tau_values)

    if config.points_npz:
        rows = _load_points_npz(config.points_npz)
    elif config.base_dir is None:
        # Auto-find the latest generated sweep.
        import glob as _glob
        dirs = sorted(_glob.glob(os.path.join(config.search_root,
                                               "run_*")))
        if not dirs:
            raise FileNotFoundError("No generated hidden-dimension runs found in " + config.search_root)
        base_dir = dirs[-1]
        print(f"Auto-selected: {base_dir}")
    else:
        base_dir = config.base_dir

    if not config.points_npz:
        print(f"Reading sweep from: {base_dir}")
        rows = _load_sweep(base_dir)
    if not rows:
        print("No valid results — nothing to plot.")
        return

    agg = _aggregate(rows)

    tau_star, mse_star = _fit_tau(agg)
    print(f"\nFitted τ* = {tau_star:.4f}  (MSE = {mse_star:.6f})")

    _print_table(agg, tau_values)
    _plot(agg, config.output_dir, tau_values, tau_star)
    print(f"\nDone. Plots saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
