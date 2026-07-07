"""Patch-scale and grid-convergence study for the oracle recovery.

Diagnostic for the weak-form discretization bias: the recovered constant mu
depends on the test-function (patch) length scale r relative to the grid
spacing dx, because the continuum weak form sampled at particles does not
equal the simulator's discrete momentum balance. This sweeps physical patch
radius for one or more dumps (different grid resolutions) and reports mu vs
physical r and vs r/dx, so grid refinement can be shown to drive mu toward
the truth (the G1 analog of G0's quadrature-refinement convergence).

Uses the raw (unscaled) least-squares estimate mu = (A.b)/(A.A): empirically
the inertia-gravity row scaling up-weights low-signal rows and worsens the
estimate for constant mu, whereas the A-weighted raw LS up-weights
high-stress-power rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ident.features.constant import ConstantDict
from ident.gates.g1_oracle import stratify_patches
from ident.io.schema import load_dump
from ident.weakform.assembly import assemble_system
from ident.weakform.from_dump import build_inplane_frames


def _dx_of(dump) -> float:
    cfg = dump.meta.extra.get("config", {})
    if "grid_lim" in cfg and "n_grid" in cfg:
        return cfg["grid_lim"] / cfg["n_grid"]
    return 4.0 * dump.meta.grain_diameter


def sweep_dump(dump_path, radii_cells=(3, 4, 6, 8, 10, 12), r_t_frames=30):
    dump = load_dump(dump_path)
    dx = _dx_of(dump)
    bundle = build_inplane_frames(dump)
    rows_out = []
    for rc in radii_cells:
        r = rc * dx
        rows, pm = stratify_patches(
            bundle, r, r, r_t_frames * dump.meta.frame_dt,
            min_particles=int(20 * (rc / 4) ** 2),
        )
        if not rows:
            rows_out.append(dict(rc=rc, r=r, r_over_dx=rc, n_patches=0,
                                 mu_acc=None, mu_tw=None))
            continue
        sysm = assemble_system(bundle.frames, rows, ConstantDict(), bundle.eps_gamma)
        A = sysm.A[:, 0]
        keep = np.abs(A) > 1e-12 * np.abs(A).max()
        mu_acc = float((A[keep] @ sysm.b_acc[keep]) / (A[keep] @ A[keep]))
        mu_tw = float((A[keep] @ sysm.b_tw[keep]) / (A[keep] @ A[keep]))
        rows_out.append(dict(rc=rc, r=float(r), r_over_dx=float(rc),
                             n_patches=len(pm), n_rows=int(keep.sum()),
                             mu_acc=mu_acc, mu_tw=mu_tw))
    mu_true = dump.meta.law_params.get("mu_s") if dump.meta.law == "constant" else None
    return dict(dump=str(dump_path), dx=dx, n_grid=dump.meta.extra.get("config", {}).get("n_grid"),
                mu_true=mu_true, sweep=rows_out)


def study(dump_paths, out_dir="out/scale_study"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [sweep_dump(p) for p in dump_paths]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axr, axn) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140)
    for res in results:
        sw = [s for s in res["sweep"] if s["mu_acc"] is not None]
        r_mm = [s["r"] * 1e3 for s in sw]
        rod = [s["r_over_dx"] for s in sw]
        mu_acc = [s["mu_acc"] for s in sw]
        mu_tw = [s["mu_tw"] for s in sw]
        lbl = f"n_grid={res['n_grid']} (dx={res['dx']*1e3:.1f}mm)"
        axr.plot(r_mm, mu_acc, "o-", label=lbl + " acc")
        axr.plot(r_mm, mu_tw, "s--", alpha=0.6, label=lbl + " tw")
        axn.plot(rod, mu_acc, "o-", label=lbl + " acc")
        axn.plot(rod, mu_tw, "s--", alpha=0.6, label=lbl + " tw")
    mu_true = next((r["mu_true"] for r in results if r["mu_true"]), None)
    for ax in (axr, axn):
        if mu_true:
            ax.axhline(mu_true, color="k", lw=1.5, label=f"truth {mu_true}")
        ax.set_ylabel("recovered mu"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axr.set_xlabel("patch radius r (mm)")
    axn.set_xlabel("r / dx (test scale in grid cells)")
    fig.suptitle("G1 patch-scale and grid-convergence study (constant mu)")
    fig.tight_layout()
    p = out_dir / "scale_study.png"
    fig.savefig(p)
    plt.close(fig)
    with open(out_dir / "scale_study.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    return results, p


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] or ["out/dumps/column_constant_a2.npz"]
    res, fig = study(paths)
    for r in res:
        print(f"\n{r['dump']}  dx={r['dx']*1e3:.2f}mm  truth={r['mu_true']}")
        for s in r["sweep"]:
            if s["mu_acc"] is not None:
                print(f"  r/dx={s['r_over_dx']:5.1f}  r={s['r']*1e3:5.1f}mm  "
                      f"npatch={s['n_patches']:3d}  mu_acc={s['mu_acc']:.3f}  mu_tw={s['mu_tw']:.3f}")
    print("figure:", fig)
