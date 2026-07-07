"""Does CoTracker work on a continuous dough SURFACE? Smooth vs material-textured.

A smooth reconstructed surface is nearly featureless -> point tracking fails (aperture
problem; shading follows geometry/light, not material; the mesh re-triangulates each frame).
The fix: paint the particles' MATERIAL-LOCKED speckle onto the surface (colour each surface
vertex by its nearest particle's fixed brightness) -> a continuous, realistic dough that
still carries a material-attached texture CoTracker can follow.

We render the same squeeze both ways, run CoTracker3 on each, and compare tracking quality
(CoTracker visibility/confidence + how many points stay confidently tracked). Run:
  ../.venv/bin/python examples/surface_track_test.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from warpmpm import GridConfig, Solver, block, newtonian
from warpmpm.coupling.backend import WarpMPMBackend

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from dough_surface_render import _surface

OUT = Path(__file__).resolve().parents[1] / "out" / "surface_track"
LIGHT = np.array([0.35, 0.55, 0.78]); LIGHT = LIGHT / np.linalg.norm(LIGHT)
DOUGH = np.array([0.93, 0.80, 0.55])


def _render_seq(stem, textured, speckle, frames_x, floor, ch, cx, cy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fdir = OUT / stem; fdir.mkdir(parents=True, exist_ok=True)
    for o in fdir.glob("f_*.png"):
        o.unlink()
    for i, x in enumerate(frames_x):
        vw, faces, fn = _surface(x)
        sh = np.clip(fn @ LIGHT, 0, 1) * 0.55 + 0.45
        if textured:
            tree = cKDTree(x)
            _, idx = tree.query(vw[faces].mean(1))           # nearest particle per face
            tex = 0.45 + 0.55 * speckle[idx]                 # material-locked flecks
            fc = np.clip((sh * tex)[:, None] * DOUGH, 0, 1)
        else:
            fc = np.clip(sh[:, None] * DOUGH, 0, 1)
        fig = plt.figure(figsize=(4.6, 4.0), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        tri = Poly3DCollection(vw[faces], facecolors=fc, edgecolors="none")
        tri.set_alpha(1.0); ax.add_collection3d(tri)
        ax.set_xlim(cx - 0.10, cx + 0.10); ax.set_ylim(cy - 0.10, cy + 0.10)
        ax.set_zlim(floor - 0.004, floor + 0.085); ax.set_box_aspect((1, 1, 0.5))
        ax.view_init(elev=18, azim=-60); ax.set_axis_off()
        fig.savefig(fdir / f"f_{i:04d}.png", dpi=120, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    return fdir


def _cotrack(fdir, spacing=10, device="cpu"):
    import torch
    from PIL import Image
    files = sorted(Path(fdir).glob("f_*.png"))
    imgs = np.stack([np.asarray(Image.open(f).convert("RGB").resize((460, 400)))
                     for f in files]).astype(np.float32)
    video = torch.from_numpy(imgs).permute(0, 3, 1, 2)[None]
    f0 = imgs[0]; H, W = f0.shape[:2]
    ys = np.arange(spacing, H - spacing, spacing); xs = np.arange(spacing, W - spacing, spacing)
    GX, GY = np.meshgrid(xs, ys); pts = np.stack([GX.ravel(), GY.ravel()], -1).astype(np.float32)
    val = f0[pts[:, 1].astype(int), pts[:, 0].astype(int)].sum(1)
    pts = pts[val < 720]                                     # on the dough (not white bg)
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1)
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()
    with torch.no_grad():
        tr, vis = model(video.to(device), queries=torch.from_numpy(q)[None].to(device))
    return tr[0].cpu().numpy(), vis[0].cpu().numpy()


def run(geom=(0.16, 0.16, 0.06), n_grid=52, nframes=22, device="auto"):
    OUT.mkdir(parents=True, exist_ok=True)
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    cw, cd, ch = geom
    pos, vol, floor = block(grid, size=geom, ppc=2)
    s = Solver(grid=grid, device=device).load_particles(pos, vol)
    s.set_material(newtonian(eta=40.0, density=1000.0).with_yield(200.0))
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    cx = cy = grid.grid_lim * 0.5
    bh = (0.5 * cw + 0.015, 0.5 * cd + 0.015, 0.6 * grid.dx)
    be = WarpMPMBackend(solver=s); z = floor + ch + bh[2]; tool = be.attach_tool((cx, cy, z), bh)
    speckle = np.random.default_rng(0).uniform(0.0, 1.0, len(pos))
    fdt = 2e-3; prev = z; stride = round(0.5 * ch / 0.08 / fdt / nframes)
    frames_x = []
    for f in range(nframes * stride + 1):
        zn = z - 0.08 * fdt if f > 0 else z; vz = (zn - prev) / fdt
        if f > 0:
            be.set_tool_kinematics(tool, center=(cx, cy, prev), velocity=(0, 0, vz))
            be.step(1e-4, 20)
        z = zn; prev = zn
        if f % stride == 0:
            frames_x.append(s.x().copy())
    print(f"rendering {len(frames_x)} frames smooth + textured, then CoTracker each...")
    res = {}
    for stem, tex in (("smooth", False), ("textured", True)):
        fdir = _render_seq(stem, tex, speckle, frames_x, floor, ch, cx, cy)
        tr, vis = _cotrack(fdir)
        # tracking-quality proxies: mean CoTracker confidence + fraction kept confident to end
        meanvis = float(vis.mean())
        kept = float((vis[-1] > 0.8).mean())
        # coherence: how rigidly neighbouring tracks move (lower spread of pairwise-distance
        # change = more coherent material tracking; sliding/drift inflates it)
        d0 = np.linalg.norm(tr[0][:, None] - tr[0][None], axis=-1)
        dT = np.linalg.norm(tr[-1][:, None] - tr[-1][None], axis=-1)
        near = (d0 > 1) & (d0 < 30)
        coh = float(np.median(np.abs(dT[near] - d0[near]) / d0[near])) if near.any() else np.nan
        res[stem] = {"n": int(tr.shape[1]), "mean_vis": meanvis, "kept_conf": kept,
                     "local_drift": coh, "tr": tr, "vis": vis, "fdir": str(fdir)}
        print(f"  {stem:9s}: {tr.shape[1]} pts  mean-confidence {meanvis:.2f}  "
              f"kept-confident {kept*100:.0f}%  local-drift {coh:.2f}")
    _overlay(res, geom)
    return {k: {kk: v[kk] for kk in ("n", "mean_vis", "kept_conf", "local_drift")}
            for k, v in res.items()}


def _overlay(res, geom):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, stem in zip(axes, ("smooth", "textured"), strict=False):
        r = res[stem]
        files = sorted(Path(r["fdir"]).glob("f_*.png"))
        img = np.asarray(Image.open(files[-1]).convert("RGB").resize((460, 400)))
        ax.imshow(img)
        tr, vis = r["tr"], r["vis"]
        for n in range(tr.shape[1]):
            v = vis[:, n] > 0.5
            if v.sum() > 3:
                ax.plot(tr[v, n, 0], tr[v, n, 1], "-", color="#1c7ed6", lw=0.4, alpha=0.6)
        ax.scatter(tr[-1][:, 0], tr[-1][:, 1], s=3, c="#ff3030")
        ax.set_title(f"{stem}: conf {r['mean_vis']:.2f}, drift {r['local_drift']:.2f}",
                     fontsize=10); ax.axis("off")
    fig.suptitle("CoTracker on a dough SURFACE: smooth (featureless) vs material-textured",
                 fontsize=11)
    fig.tight_layout()
    p = OUT / "surface_track_compare.png"; fig.savefig(p, dpi=130); plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-framerate", "8", "-i", str(Path(res["textured"]["fdir"])
                    / "f_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(OUT / "textured.mp4")],
                   check=True, capture_output=True)
    print("wrote", p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="Warp MPM device, e.g. cuda:0 or cuda:1")
    args = parser.parse_args()
    run(device=args.device)
