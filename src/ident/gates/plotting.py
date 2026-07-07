"""Shared mu(I) plotting harness for all gates.

The core figure of the project is the four-curve plot per aspect ratio
(truth, true-p recovery, P1 recovery, P0 recovery). G0 and G1 use the same
function with fewer curves so the harness stays stable from day one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVE_STYLE = {
    "truth": dict(color="#222222", lw=2.2, ls="-"),
    "true_p": dict(color="#d62728", lw=1.8, ls="-"),
    "P1": dict(color="#1f77b4", lw=1.8, ls="--"),
    "P0": dict(color="#2ca02c", lw=1.8, ls=":"),
}


def plot_mu_curves(
    I_grid: np.ndarray,
    curves: dict[str, np.ndarray],
    path: str | Path,
    bands: dict[str, np.ndarray] | None = None,
    observed_I: tuple[float, float] | None = None,
    title: str = "",
) -> Path:
    """curves: name -> mu values on I_grid. bands: name -> std values."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    for name, mu in curves.items():
        style = CURVE_STYLE.get(name, dict(lw=1.6))
        ax.semilogx(I_grid, mu, label=name, **style)
        if bands and name in bands:
            ax.fill_between(
                I_grid,
                mu - 2.0 * bands[name],
                mu + 2.0 * bands[name],
                alpha=0.18,
                color=style.get("color", "#888888"),
                linewidth=0,
            )
    if observed_I is not None:
        ax.axvspan(observed_I[0], observed_I[1], color="#cccc66", alpha=0.12, zorder=0)
        ax.text(
            observed_I[0], ax.get_ylim()[0], " observed I band",
            fontsize=8, color="#777733", va="bottom",
        )
    ax.set_xlabel("inertial number I")
    ax.set_ylabel("mu(I)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_convergence(
    h_values: np.ndarray,
    errors: dict[str, np.ndarray],
    path: str | Path,
    title: str = "",
) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=150)
    for name, err in errors.items():
        ax.loglog(h_values, err, "o-", label=name)
    # order-2 guide through the last point of the first series
    first = next(iter(errors.values()))
    guide = first[-1] * (np.asarray(h_values) / h_values[-1]) ** 2
    ax.loglog(h_values, guide, "k--", lw=0.9, alpha=0.6, label="order 2 guide")
    ax.set_xlabel("quadrature spacing h")
    ax.set_ylabel("relative error")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
