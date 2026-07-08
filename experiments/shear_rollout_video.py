"""Render the held-out 3D shear rollout: truth vs FE vs Bingham, with a live force trace.

Two visuals from experiments/shear_cell_3d.py's recorded rollout (out/shear_cell_3d/rollout_3d.npz):
  shear_rollout_3d.mp4   3 panels (truth / FE / Bingham) of the dough sheared by the top
                         wall, particles coloured by speed (the shear gradient), plus a
                         bottom panel with the live wall shear force for all three. The
                         shapes track (displacement-controlled) while the force separates:
                         FE hugs truth, Bingham diverges. This is the self-consistent test.
  shear_3d_view.mp4      a rotating 3D perspective of the truth block shearing (shows it is
                         a genuine 3D dough block, not the quasi-2D slab).

Run:  python experiments/shear_rollout_video.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "out" / "shear_cell_3d"


def _load():
    d = np.load(OUT / "rollout_3d.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def make_comparison(stride=1, fps=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, gridspec
    d = _load()
    tags = ["truth", "FE", "Bingham"]
    X = {t: d[f"X_{t}"] for t in tags}
    V = {t: d[f"V_{t}"] for t in tags}
    Fx = {t: d[f"Fx_{t}"] for t in tags}
    t_force = d["t"]
    traj_t = d["traj_t"]
    floor = float(d["floor"])
    v_hold = float(d["v_holdout"])
    fr = {"FE": float(d["fr_FE"]), "Bingham": float(d["fr_Bingham"])}
    nft = min(len(X[t]) for t in tags)
    frames = list(range(0, nft, stride))

    allx = np.concatenate([X[t][:, :, 0].ravel() for t in tags])
    allz = np.concatenate([X[t][:, :, 2].ravel() for t in tags])
    xlim = (allx.min(), allx.max())
    zlim = (floor - 0.005, allz.max() + 0.005)
    vmax = float(np.percentile(np.concatenate(
        [np.linalg.norm(V[t][0], axis=1) for t in tags]), 99)) or v_hold
    vmax = max(vmax, v_hold)

    fig = plt.figure(figsize=(12.5, 6.6))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.3, 1.0], hspace=0.34, wspace=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axF = fig.add_subplot(gs[1, :])
    titles = {"truth": "truth (Herschel-Bulkley)", "FE": "FE (recovered, tabulated)",
              "Bingham": "Bingham fit"}
    sc = {}
    for ax, tag in zip(axes, tags, strict=True):
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        sub = titles[tag] if tag == "truth" else f"{titles[tag]}  ({fr.get(tag,0)*100:.0f}% force)"
        ax.set_title(sub, fontsize=10)
        ax.axhline(floor, color="#5b4636", lw=3)
        sc[tag] = ax.scatter(X[tag][0, :, 0], X[tag][0, :, 2], s=4,
                             c=np.zeros(X[tag].shape[1]), cmap="viridis", vmin=0, vmax=vmax,
                             edgecolors="none")

    C = {"truth": "k", "FE": "#1864ab", "Bingham": "#e8590c"}
    for tag in tags:
        lab = tag if tag == "truth" else f"{tag} ({fr[tag]*100:.0f}%)"
        axF.plot(t_force, -Fx[tag], color=C[tag], lw=2.4 if tag == "truth" else 1.9,
                 ls="-" if tag != "Bingham" else "--", label=lab)
    axF.set_xlabel("time (s)")
    axF.set_ylabel("wall shear force  -F_x  (N)")
    axF.set_xlim(t_force[0], t_force[-1])
    axF.legend(loc="lower right", fontsize=9)
    axF.grid(alpha=0.3)
    vline = axF.axvline(traj_t[0], color="0.5", lw=1)
    sup = fig.suptitle("", fontsize=13, y=0.99)

    def upd(fi):
        for tag in tags:
            spd = np.linalg.norm(V[tag][fi], axis=1)
            sc[tag].set_offsets(X[tag][fi][:, [0, 2]])
            sc[tag].set_array(spd)
        vline.set_xdata([traj_t[fi], traj_t[fi]])
        sup.set_text(f"Held-out 3D shear rollout (v={v_hold} m/s):  shapes track, FORCE "
                     f"separates: FE {fr['FE']*100:.0f}% vs Bingham {fr['Bingham']*100:.0f}%"
                     f"   [t={traj_t[fi]:.2f}s]")
        return (*sc.values(), vline, sup)

    anim = animation.FuncAnimation(fig, upd, frames=frames, blit=False)
    p = OUT / "shear_rollout_3d.mp4"
    anim.save(str(p), writer=animation.FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)
    print(f"wrote {p}  ({len(frames)} frames)")
    return p


def make_3d_view(stride=1, fps=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    d = _load()
    X = d["X_truth"]
    V = d["V_truth"]
    traj_t = d["traj_t"]
    floor = float(d["floor"])
    nft = len(X)
    frames = list(range(0, nft, stride))
    vmax = float(np.percentile(np.linalg.norm(V[min(nft - 1, nft // 2)], axis=1), 99)) or 0.16

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    x0 = X[:, :, 0]
    lim = (float(x0.min()), float(x0.max()))
    yl = (float(X[0, :, 1].min()), float(X[0, :, 1].max()))
    zl = (floor, float(X[:, :, 2].max()) + 0.005)
    sc = ax.scatter(X[0, :, 0], X[0, :, 1], X[0, :, 2], s=5,
                    c=np.zeros(X.shape[1]), cmap="viridis", vmin=0, vmax=vmax, edgecolors="none")

    def upd(fi):
        ax.clear()
        spd = np.linalg.norm(V[fi], axis=1)
        ax.scatter(X[fi][:, 0], X[fi][:, 1], X[fi][:, 2], s=5, c=spd, cmap="viridis",
                   vmin=0, vmax=vmax, edgecolors="none")
        ax.set_xlim(*lim)
        ax.set_ylim(*yl)
        ax.set_zlim(*zl)
        ax.set_box_aspect((lim[1] - lim[0], (yl[1] - yl[0]) * 2.2, (zl[1] - zl[0]) * 2.2))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=18, azim=-60 + 0.25 * fi)
        ax.set_title(f"3D dough block sheared by the top wall  [t={traj_t[fi]:.2f}s]\n"
                     "particles coloured by speed (the shear gradient)", fontsize=10)
        return (sc,)

    anim = animation.FuncAnimation(fig, upd, frames=frames, blit=False)
    p = OUT / "shear_3d_view.mp4"
    anim.save(str(p), writer=animation.FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)
    print(f"wrote {p}  ({len(frames)} frames)")
    return p


if __name__ == "__main__":
    make_comparison()
    make_3d_view()
