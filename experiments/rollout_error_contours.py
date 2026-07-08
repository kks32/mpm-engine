"""Spatial error-field contours of the held-out 3D shear rollout.

The deformation snapshots (rollout_snapshots.py) all look identical, so this shades where each
model departs from truth. All four sims start from the same seeded block, so particles are in
1:1 correspondence; the per-particle deviation |x_model - x_truth| (mm) is binned over the x-z
plane and contour-shaded. Rows = FE / HB-ridge / Bingham, columns = time points. Reveals where
the biggest deformation errors concentrate (and confirms they stay small, the kinematics being
displacement-controlled, while the force carries the model difference).

Run:  python experiments/rollout_error_contours.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binned_statistic_2d

import shear_cell_3d as S
from warpmpm import newtonian

OUT = S.OUT
V_HOLDOUT = 0.16
HB_RIDGE = dict(eta=18.6, tau_y=68.6, pk=71.9, pn=0.20)
FRACS = (0.18, 0.4, 0.7, 1.0)


def run():
    d = np.load(OUT / "rollout_3d.npz")
    traj_t = d["traj_t"]
    floor = float(d["floor"])

    mat = (newtonian(eta=HB_RIDGE["eta"], density=S.RHO, bulk_modulus=9.0e5)
           .with_yield(HB_RIDGE["tau_y"]).with_powerlaw(K=HB_RIDGE["pk"], n=HB_RIDGE["pn"]))
    seg = S.shear_segment(V_HOLDOUT, mat, n_frames=160, record_traj=True, traj_stride=2)

    Xt = d["X_truth"]
    models = {"FE": d["X_FE"], "HB-ridge": seg["X"], "Bingham": d["X_Bingham"]}
    fr = {"FE": float(d["fr_FE"]), "HB-ridge": 0.25, "Bingham": float(d["fr_Bingham"])}
    rows = ["FE", "HB-ridge", "Bingham"]
    nft = min([len(Xt)] + [len(m) for m in models.values()])
    cols = [min(int(f * (nft - 1)), nft - 1) for f in FRACS]

    # common spatial frame (truth positions over the whole run) and bin grid
    allx = Xt[:nft, :, 0]
    allz = Xt[:nft, :, 2]
    xe = np.linspace(allx.min(), allx.max(), 46)
    ze = np.linspace(floor, allz.max() + 0.002, 18)
    # error magnitude (mm) per particle per frame, per model; shared colour scale
    err = {r: np.linalg.norm(models[r][:nft] - Xt[:nft], axis=2) * 1e3 for r in rows}
    vmax = max(np.percentile(err[r][cols[-1]], 98) for r in rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, nc = len(rows), len(cols)
    fig, ax = plt.subplots(nr, nc, figsize=(2.9 * nc, 1.9 * nr), squeeze=False)
    im = None
    for i, r in enumerate(rows):
        for j, cf in enumerate(cols):
            a = ax[i][j]
            x = Xt[cf][:, 0]
            z = Xt[cf][:, 2]
            stat, _, _, _ = binned_statistic_2d(x, z, err[r][cf], statistic="mean",
                                                bins=[xe, ze])
            XX, ZZ = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ze[:-1] + ze[1:]))
            field = np.ma.masked_invalid(stat.T)
            im = a.contourf(XX, ZZ, field, levels=np.linspace(0, vmax, 13),
                            cmap="inferno", extend="max")
            a.axhline(floor, color="#5b4636", lw=2)
            a.set_xlim(xe[0], xe[-1])
            a.set_ylim(ze[0], ze[-1])
            a.set_aspect("equal")
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(f"t = {traj_t[cf]:.2f} s", fontsize=10)
            if j == 0:
                a.set_ylabel(f"{r}\n({fr[r]*100:.0f}% force err)", fontsize=10)
    fig.suptitle("Held-out 3D shear rollout: deformation error vs truth, |x_model - x_truth| (mm)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label="|x_model - x_truth|  (mm)")
    p = OUT / "rollout_error_contours.png"
    fig.savefig(p, dpi=135)
    plt.close(fig)
    print("wrote", p, f"(deformation error up to ~{vmax:.1f} mm)")
    return p


if __name__ == "__main__":
    run()
