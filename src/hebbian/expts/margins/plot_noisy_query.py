"""
Auxiliary side-by-side plots for Sec 3.2 (noisy-query margin experiments).

Layout per figure (1 × 3):
  Left:   Clean γ_min (dashed gray) + Noisy γ_min (solid black ± std)
  Middle: Theory bound (red, left axis) + L_bil (blue, right axis)
  Right:  Noisy γ_min (black) + fitted noisy-query bound (red dashed), with
          the fitted equation and fit statistics moved into an in-axes text
          block for the paper-facing fit-only panel.

Produces one figure per results JSON (epsilon-sweep and M-sweep).

Usage:
    python -m hebbian.expts.margins.plot_noisy_query
    python -m hebbian.expts.margins.plot_noisy_query --output ./aux_plots
    python -m hebbian.expts.margins.plot_noisy_query path/to/results.json [path/to/results2.json] --output ./aux_plots
"""

import os
import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit


def setup_plot_style() -> None:
    """Set plot styling to match the paper margin plots."""
    cmap = plt.get_cmap("Set1")
    plt.rcParams.setdefault("axes.prop_cycle", plt.cycler(color=cmap.colors))
    plt.rcParams["axes.titlesize"] = 10.5
    plt.rcParams["axes.titleweight"] = "normal"
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markeredgewidth"] = 1.0
    plt.rcParams["lines.markeredgecolor"] = "black"
    plt.rcParams["lines.markersize"] = math.sqrt(80)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(path: str) -> dict:
    with open(path) as f:
        results = json.load(f)
    # For M-sweeps, epsilon may not be stored at top level — recover from sibling pkl
    if results.get("sweep_type") == "M" and not results.get("epsilon"):
        import glob, pickle
        parent = os.path.dirname(path)
        pkls = glob.glob(os.path.join(parent, "**", "grid_search_results*.pkl"), recursive=True)
        for pkl_path in sorted(pkls)[:1]:
            with open(pkl_path, "rb") as f:
                r = pickle.load(f)
            eps = (r.get("results") or {}).get("epsilon", 0.0)
            if eps:
                results["epsilon"] = eps
                break
    return results


def _sort(results: dict):
    sweep_type = results.get("sweep_type", "epsilon")
    if sweep_type == "epsilon":
        x = np.array(results["epsilon_values"])
        xlabel = r"Query noise level $\varepsilon$"
        log_x = True
    elif sweep_type == "M":
        x = np.array(results["M_values"])
        xlabel = "Feature dimension $m$"
        log_x = True
    else:
        x = np.arange(len(results.get("gamma_min_best", [])))
        xlabel = sweep_type
        log_x = False

    idx = np.argsort(x)
    x = x[idx]

    def s(key, default=float("nan")):
        raw = results.get(key, [])
        if not raw:
            return np.full(len(x), float("nan"))
        return np.array([v if v is not None else float("nan") for v in raw],
                        dtype=float)[idx]

    n_values = s("n_values")

    return dict(
        x=x, xlabel=xlabel, log_x=log_x,
        gamma_clean=s("gamma_min_best"),
        gamma_std=s("gamma_min_std", 0.0),
        gamma_noisy=s("gamma_min_noisy_best"),
        noisy_bound=s("noisy_bound_best"),
        L_bil=s("L_bil_best"),
        n_values=n_values,
        d=results.get("d", "?"),
        F=results.get("F", "?"),
        M=results.get("M", "?"),
        epsilon=results.get("epsilon", 0.0) or 0.0,
        sweep_type=sweep_type,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_aux(data: dict, save_path: str):
    x = data["x"]
    log_x = data["log_x"]
    xlabel = data["xlabel"]
    d, F, M, eps = data["d"], data["F"], data["M"], data["epsilon"]

    if data["sweep_type"] == "epsilon":
        subtitle = f"$d={d}$,  $F={F}$,  $m={M}$"
    else:
        subtitle = f"$d={d}$,  $F={F}$,  $\\varepsilon={eps}$"

    fig, (ax_emp, ax_bound, ax_fit) = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(subtitle, fontsize=13)

    # ---- Left: empirical ----
    gamma_clean = data["gamma_clean"]
    gamma_noisy = data["gamma_noisy"]
    gamma_std   = data["gamma_std"]

    ax_emp.plot(x, gamma_clean, "o--", color="gray", linewidth=1.8,
                markersize=4, alpha=0.8,
                label=r"Clean $\gamma_{\min}$ (stored queries)")
    ax_emp.fill_between(
        x,
        np.where(np.isfinite(gamma_noisy - gamma_std), gamma_noisy - gamma_std, np.nan),
        np.where(np.isfinite(gamma_noisy + gamma_std), gamma_noisy + gamma_std, np.nan),
        color="black", alpha=0.15,
    )
    ax_emp.plot(x, gamma_noisy, "o-", color="black", linewidth=2.2,
                markersize=5, label=r"Noisy $\gamma_{\min}(z)$", zorder=10)
    ax_emp.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    if log_x and np.all(x > 0):
        ax_emp.set_xscale("log")
    ax_emp.set_xlabel(xlabel, fontsize=12)
    ax_emp.set_ylabel(r"$\gamma_{\min}$", fontsize=12)
    ax_emp.set_title("Empirical margins", fontsize=12)
    ax_emp.legend(fontsize=10, loc="best", frameon=True, framealpha=0.85)
    ax_emp.grid(True, alpha=0.3)

    # ---- Right: theory bound (left axis) + L_bil (right axis) ----
    noisy_bound = data["noisy_bound"]
    L_bil       = data["L_bil"]

    color_bound = "#d62728"  # red
    color_lbil  = "#1f77b4"  # blue

    ax_bound.plot(x, noisy_bound, "s-", color=color_bound, linewidth=1.8,
                  markersize=4, label="Theory bound")
    ax_bound.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_bound.set_xlabel(xlabel, fontsize=12)
    ax_bound.set_ylabel("Theory bound", color=color_bound, fontsize=12)
    ax_bound.tick_params(axis="y", labelcolor=color_bound)
    if log_x and np.all(x > 0):
        ax_bound.set_xscale("log")
    ax_bound.set_title("Theory bound  +  $L_{\\mathrm{bil}}$", fontsize=12)

    # Annotate the bound formula (from noisy_query_margin_bound)
    formula = (
        r"$\gamma_{\min}(z) \geq$"
        "\n"
        r"$1 - 2\sqrt{6}\sqrt{\frac{nL}{d^3}} - 8\sqrt{\frac{nL}{md}}$"
        "\n"
        r"$- \sqrt{2}\sqrt{\frac{L}{d}} - \sqrt{18}\sqrt{\frac{L_n}{m}} - \frac{8L_n}{m} - \frac{4L}{d^2}$"
        "\n"
        r"$- L_{\mathrm{bil}}\,\varepsilon\,(1 + 2\sqrt{2}\sqrt{nL/d})$"
        "\n"
        r"$L=\log(4n^2/\delta),\ L_n=\log(4n/\delta)$"
    )
    ax_bound.text(0.03, 0.03, formula, transform=ax_bound.transAxes,
                  fontsize=7, verticalalignment="bottom",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax_bound.grid(True, alpha=0.3)

    ax_lbil = ax_bound.twinx()
    valid = np.isfinite(L_bil)
    if valid.any():
        ax_lbil.plot(x[valid], L_bil[valid], "^--", color=color_lbil,
                     linewidth=1.6, markersize=4, alpha=0.85,
                     label=r"$L_{\mathrm{bil}}$")
    ax_lbil.set_ylabel(r"$L_{\mathrm{bil}}$", color=color_lbil, fontsize=12)
    ax_lbil.tick_params(axis="y", labelcolor=color_lbil)

    # Combined legend
    lines_b, labs_b = ax_bound.get_legend_handles_labels()
    lines_l, labs_l = ax_lbil.get_legend_handles_labels()
    ax_bound.legend(lines_b + lines_l, labs_b + labs_l,
                    fontsize=10, loc="best", frameon=True, framealpha=0.85)

    # ---- Right: fitted theory bound (fit constants via curve_fit) ----
    # Model: Ĉ_sig·1 - Ĉ_xtalk·√(n·L̂/(m·d)) - C_penalty·L_bil·ε·√(n·L̂/d)
    # where L̂ = log(Ĉ_L·n²/δ), fitting Ĉ_sig, Ĉ_L, Ĉ_xtalk, C_penalty
    n_arr = data["n_values"]
    L_bil = data["L_bil"]
    eps_arr = x  # epsilon values are the x-axis for epsilon sweep
    m_val = float(M) if M != "?" else 1.0
    d_val = float(d) if d != "?" else 1.0
    delta = 0.5

    valid = np.isfinite(gamma_noisy) & np.isfinite(L_bil) & np.isfinite(n_arr)

    def _model(_, C_sig, C_L, C_xtalk, C_eps):
        Lhat = np.log(np.maximum(C_L * n_arr ** 2 / delta, 1e-30))
        clean = C_sig - C_xtalk * np.sqrt(np.maximum(n_arr * Lhat / (m_val * d_val), 0.0))
        penalty = C_eps * L_bil * eps_arr * np.sqrt(np.maximum(n_arr * Lhat / d_val, 0.0))
        return clean - penalty

    fitted = np.full_like(gamma_noisy, float("nan"))
    fit_label = "Fitted theory"
    fit_stats = "fit unavailable"
    fit_equation = (
        r"Fitted bound: $\gamma_{\min}(z)=C_s"
        r"-C_x\sqrt{n\hat L/(md)}$"
        "\n"
        r"$-C_\varepsilon L_{\rm bil}\varepsilon\sqrt{n\hat L/d}$"
    )
    if valid.sum() >= 5:
        try:
            popt, _ = curve_fit(_model, np.zeros(valid.sum()), gamma_noisy[valid],
                                p0=[1.0, 4.0, 1.0, 1.0],
                                bounds=([0, 1e-6, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
                                maxfev=20000)
            C_sig, C_L, C_xtalk, C_eps = popt
            fitted = _model(None, C_sig, C_L, C_xtalk, C_eps)
            ss_tot = float(np.sum((gamma_noisy[valid] - gamma_noisy[valid].mean()) ** 2))
            ss_res = float(np.sum((gamma_noisy[valid] - fitted[valid]) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")
            r2_str = f"{r2:.3f}" if np.isfinite(r2) else "nan"
            mse = float(np.mean((gamma_noisy[valid] - fitted[valid]) ** 2))
            fit_stats = rf"$R^2={r2_str}$, MSE={mse:.2e}"
        except Exception:
            pass

    def _draw_right_panel(ax, add_inset: bool = False):
        ax.fill_between(
            x,
            np.where(np.isfinite(gamma_noisy - gamma_std), gamma_noisy - gamma_std, np.nan),
            np.where(np.isfinite(gamma_noisy + gamma_std), gamma_noisy + gamma_std, np.nan),
            color="#2E86AB", alpha=0.15,
        )
        ax.plot(x, gamma_noisy, "o-", color="#2E86AB", linewidth=2.0,
                markersize=6.5, markeredgewidth=1.0, markeredgecolor="black",
                alpha=0.92, label=r"Noisy $\gamma_{\min}(z)$", zorder=10)
        ax.plot(x, fitted, "s--", color="#E63946", linewidth=2.0,
                markersize=6.5, markeredgewidth=1.0, markeredgecolor="black",
                alpha=0.92, label=fit_label)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        if log_x and np.all(x > 0):
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Minimum margin $\gamma_{\min}$")
        ax.set_title("")
        ax.legend(fontsize=10, loc="lower left", frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--", which="major")
        if add_inset:
            info_text = "\n".join(
                [
                    "noisy queries (isotropic keys, isotropic values)",
                    f"$d={d}$, $F={F}$, $m={M}$",
                    fit_equation,
                    fit_stats,
                ]
            )
            ax.margins(y=0.18)
            ax.text(
                0.5,
                0.055,
                info_text,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=11.4,
                linespacing=1.13,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                },
                zorder=20,
            )

    _draw_right_panel(ax_fit)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    full_base, _ = os.path.splitext(save_path)
    plt.savefig(f"{full_base}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    print(f"Saved: {full_base}.pdf")

    # Isolated right-panel plot
    setup_plot_style()
    fig_solo, ax_solo = plt.subplots(figsize=(7.2, 6.3))
    _draw_right_panel(ax_solo, add_inset=True)
    fig_solo.tight_layout(pad=0.35)
    base, ext = os.path.splitext(save_path)
    solo_path = f"{base}_fit_only{ext}"
    solo_pdf_path = f"{base}_fit_only.pdf"
    fig_solo.savefig(solo_path, dpi=300, bbox_inches="tight")
    fig_solo.savefig(solo_pdf_path, bbox_inches="tight")
    plt.close(fig_solo)
    print(f"Saved: {solo_path}")
    print(f"Saved: {solo_pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DEFAULT_RESULTS = [
    "./results/margin_sweep_epsilon_02262026_110445/margin_sweep_results.json",
    "./results/margin_sweep_M_02262026_110447/margin_sweep_results.json",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", default=_DEFAULT_RESULTS,
                        help="Path(s) to margin_sweep_results.json files "
                             f"(default: {_DEFAULT_RESULTS})")
    parser.add_argument("--output", default="./aux_plots",
                        help="Output directory (default: ./aux_plots)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for path in args.inputs:
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        results = load(path)
        if not results or "d" not in results:
            print(f"Warning: {path} looks empty or malformed, skipping.")
            continue
        data = _sort(results)
        sweep_type = data["sweep_type"]
        stem = os.path.splitext(os.path.basename(path))[0]
        fname = f"{stem}_aux_{sweep_type}.png"
        plot_aux(data, os.path.join(args.output, fname))


if __name__ == "__main__":
    main()
