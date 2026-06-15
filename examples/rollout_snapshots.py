"""Snapshot grid of the held-out 3D shear rollout: rows = truth / FE / HB-ridge / Bingham,
columns = time points. The dough (x-z projection) coloured by speed. The shapes track across
all four laws (displacement-controlled motion), which is exactly why the FORCE, not the
deformation, is the discriminator (FE 11%, HB-ridge 25%, Bingham 34% on the wall force).

Uses the cached truth/FE/Bingham trajectories (out/shear_cell_3d/rollout_3d.npz) and re-sims
the HB-ridge law (recovered in correct_model_check.py). Run:
  PYTHONPATH=src ../.venv/bin/python examples/rollout_snapshots.py
"""
from __future__ import annotations

import numpy as np

import shear_cell_3d as S
from warpmpm import newtonian

OUT = S.OUT
V_HOLDOUT = 0.16
# HB-ridge law recovered in correct_model_check.py (true form + generic dough prior)
HB_RIDGE = dict(eta=18.6, tau_y=68.6, pk=71.9, pn=0.20)
FRACS = (0.04, 0.18, 0.5, 1.0)            # fraction of the rollout at which to snapshot


def run():
    d = np.load(OUT / "rollout_3d.npz")
    traj_t = d["traj_t"]
    floor = float(d["floor"])
    nft = len(d["X_truth"])

    # re-sim HB-ridge with the SAME stride/length so its frames align with the cache
    mat = (newtonian(eta=HB_RIDGE["eta"], density=S.RHO, bulk_modulus=9.0e5)
           .with_yield(HB_RIDGE["tau_y"]).with_powerlaw(K=HB_RIDGE["pk"], n=HB_RIDGE["pn"]))
    seg = S.shear_segment(V_HOLDOUT, mat, n_frames=160, record_traj=True, traj_stride=2)
    X = {"truth": d["X_truth"], "FE": d["X_FE"], "HB-ridge": seg["X"], "Bingham": d["X_Bingham"]}
    V = {"truth": d["V_truth"], "FE": d["V_FE"], "HB-ridge": seg["V"], "Bingham": d["V_Bingham"]}
    rows = ["truth", "FE", "HB-ridge", "Bingham"]
    fr = {"FE": float(d["fr_FE"]), "HB-ridge": 0.25, "Bingham": float(d["fr_Bingham"])}
    nft = min(nft, len(seg["X"]))
    cols = [min(int(f * (nft - 1)), nft - 1) for f in FRACS]

    allx = np.concatenate([X[r][:nft, :, 0].ravel() for r in rows])
    allz = np.concatenate([X[r][:nft, :, 2].ravel() for r in rows])
    xlim = (allx.min(), allx.max())
    zlim = (floor - 0.004, allz.max() + 0.004)
    vmax = float(np.percentile(np.linalg.norm(V["truth"][cols[-1]], axis=1), 98)) or V_HOLDOUT

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, nc = len(rows), len(cols)
    fig, ax = plt.subplots(nr, nc, figsize=(2.7 * nc, 1.7 * nr), squeeze=False)
    for i, r in enumerate(rows):
        for j, cf in enumerate(cols):
            a = ax[i][j]
            spd = np.linalg.norm(V[r][cf], axis=1)
            a.scatter(X[r][cf][:, 0], X[r][cf][:, 2], s=2.0, c=spd, cmap="viridis",
                      vmin=0, vmax=vmax, edgecolors="none")
            a.axhline(floor, color="#5b4636", lw=2)
            a.set_xlim(*xlim)
            a.set_ylim(*zlim)
            a.set_aspect("equal")
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(f"t = {traj_t[cf]:.2f} s", fontsize=10)
            if j == 0:
                lbl = r if r == "truth" else f"{r}  ({fr[r]*100:.0f}% force)"
                a.set_ylabel(lbl, fontsize=10)
    fig.suptitle("Held-out 3D shear rollout snapshots (v=0.16 m/s): shapes track, "
                 "force separates", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = OUT / "rollout_snapshots.png"
    fig.savefig(p, dpi=135)
    plt.close(fig)
    print("wrote", p)
    return p


if __name__ == "__main__":
    run()
