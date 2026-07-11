"""Compare observed dynamics with a rollout of the identified model.

The comparison uses the observables from MATH_REFERENCE.md Section 10: front
trajectory x_f(t), final deposit profile h(x),
runout distance, and deposit area. This is forward validation of an already
identified model, not rollout-matching training: no simulator differentiation
(invariant 1).

This module has no Warp or Torch dependency and operates on two reference dumps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ident.io.schema import Dump, load_dump


@dataclass
class Observables:
    t: np.ndarray              # (F,)
    front_x: np.ndarray        # (F,) leading-edge x of the flow
    bin_centers: np.ndarray    # (nb,) deposit profile abscissae
    deposit_h: np.ndarray      # (nb,) final surface height per x bin
    runout: float              # final front advance from initial right edge
    deposit_area: float        # cross-sectional area of the final deposit


def compute_observables(dump: Dump, front_pct: float = 99.5,
                        bin_cells: float = 1.0) -> Observables:
    ax, az = dump.meta.in_plane_axes
    cfg = dump.meta.extra.get("config", {})
    dx = cfg["grid_lim"] / cfg["n_grid"] if "grid_lim" in cfg else 4 * dump.meta.grain_diameter
    bin_w = bin_cells * dx

    x = dump.x[..., ax]
    z = dump.x[..., az]
    front = np.full(dump.meta.n_frames, np.nan)
    for f in range(dump.meta.n_frames):
        m = dump.active[f]
        if np.any(m):
            front[f] = np.percentile(x[f, m], front_pct)
    x0_right = np.percentile(x[0, dump.active[0]], front_pct)  # initial front

    # final deposit profile (last frame, settled)
    mf = dump.active[-1]
    xf, zf = x[-1, mf], z[-1, mf]
    x_lo = xf.min()
    idx = np.floor((xf - x_lo) / bin_w).astype(int)
    nb = idx.max() + 1
    h = np.full(nb, np.nan)
    # surface height per bin (high percentile of z, robust to fly-out)
    for b in range(nb):
        sel = idx == b
        if sel.sum() >= 3:
            h[b] = np.percentile(zf[sel], 98)
    centers = x_lo + (np.arange(nb) + 0.5) * bin_w
    good = np.isfinite(h)
    centers, h = centers[good], h[good]
    z_base = np.percentile(zf, 1)
    area = float(np.nansum((h - z_base)) * bin_w)

    return Observables(
        t=dump.times, front_x=front, bin_centers=centers, deposit_h=h,
        runout=float(front[-1] - x0_right), deposit_area=area,
    )


def _hausdorff(a_xy, b_xy):
    """Symmetric Hausdorff distance between two profile point sets."""
    from scipy.spatial.distance import cdist
    d = cdist(a_xy, b_xy)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def compare(observed_path, resim_path, out_dir="out/roundtrip", label="constant"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = compute_observables(load_dump(observed_path))
    sim = compute_observables(load_dump(resim_path))

    # front trajectory relative L2 over the common time span
    n = min(len(obs.t), len(sim.t))
    fo, fs = obs.front_x[:n], sim.front_x[:n]
    valid = np.isfinite(fo) & np.isfinite(fs)
    front_relL2 = float(np.sqrt(np.mean((fo[valid] - fs[valid]) ** 2))
                        / np.sqrt(np.mean(fo[valid] ** 2)))
    runout_rel = abs(sim.runout - obs.runout) / max(abs(obs.runout), 1e-9)
    area_rel = abs(sim.deposit_area - obs.deposit_area) / max(abs(obs.deposit_area), 1e-9)
    haus = _hausdorff(
        np.column_stack([obs.bin_centers, obs.deposit_h]),
        np.column_stack([sim.bin_centers, sim.deposit_h]),
    )
    # normalize Hausdorff by initial column height for interpretability
    haus_rel = haus / max(obs.deposit_h.max() - np.percentile(obs.deposit_h, 1), 1e-9)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140)
    a1.plot(obs.t[:n], fo, "k-", lw=2, label="observed")
    a1.plot(sim.t[:n], fs, "r--", lw=2, label="re-sim (identified mu)")
    a1.set_xlabel("time (s)"); a1.set_ylabel("front position x_f (m)")
    a1.set_title("front trajectory"); a1.grid(alpha=0.3); a1.legend(fontsize=9)
    a2.plot(obs.bin_centers, obs.deposit_h, "k-", lw=2, label="observed deposit")
    a2.plot(sim.bin_centers, sim.deposit_h, "r--", lw=2, label="re-sim deposit")
    a2.set_xlabel("x (m)"); a2.set_ylabel("surface height (m)")
    a2.set_title("final deposit profile"); a2.set_aspect("equal")
    a2.grid(alpha=0.3); a2.legend(fontsize=9)
    fig.suptitle(f"Round-trip replication ({label}): identified model vs observed dynamics")
    fig.tight_layout()
    figpath = out_dir / f"roundtrip_{label}.png"
    fig.savefig(figpath); plt.close(fig)

    result = {
        "observed": str(observed_path), "resim": str(resim_path), "label": label,
        "front_trajectory_relL2": front_relL2,
        "runout_observed": obs.runout, "runout_resim": sim.runout,
        "runout_relative_error": runout_rel,
        "deposit_area_relative_error": area_rel,
        "deposit_hausdorff_rel": haus_rel,
        "replicates": bool(front_relL2 < 0.05 and runout_rel < 0.10 and haus_rel < 0.15),
        "figure": str(figpath),
    }
    with open(out_dir / f"roundtrip_{label}.json", "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    return result


if __name__ == "__main__":
    import sys
    obs_p = sys.argv[1]
    sim_p = sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else "constant"
    res = compare(obs_p, sim_p, label=label)
    print(json.dumps(res, indent=2, default=float))
