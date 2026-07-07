"""Predict an unseen dough volume's DEFORMATION and validate it through the camera pipeline.

Closed loop: learn the rheology on one squeeze, then SIMULATE a different dough volume with
each candidate law, RENDER it to a trackable speckle video (front/wall-plane view), run
CoTracker3 (the real perception pipeline), and compare the predicted deformation ROLLOUT to
the ground truth -- the error is the tracked material motion, not the force.

We render three forward runs on the unseen volume:
  truth   (tau_y=200, eta=40)   learned (tau_y=192, eta=55, grid-impulse)
  wrong   (tau_y=800, eta=200)  -- a deliberately bad law, to test how much deformation
                                   actually discriminates the rheology.
All three start from the identical blob, share one camera, so a common CoTracker query grid
follows corresponding material points; we compare world-space tracks point-to-point.

NOTE: the plate descent is displacement-controlled, so vertical compression is prescribed;
the rheology-dependent signal is the lateral extrusion. This quantifies how well deformation
(vs the force) validates the learned law. Run:
  ../.venv/bin/python examples/predict_volume_rollout.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, block, newtonian
from warpmpm.coupling.backend import WarpMPMBackend

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))                       # to import the perception pipeline
OUT = Path(__file__).resolve().parents[1] / "out" / "rollout"

LAWS = {
    "truth":   (200.0, 40.0),
    "learned": (192.0, 55.0),
    "wrong":   (50.0, 10.0),   # a deliberately too-soft law: should spread more if deformation
}                              # discriminates the rheology (high-yield laws need a tiny dt)


def squeeze_dump(tau_y, eta, geom, n_grid=52, v_plate=0.08, press_strain=0.5,
                 dt=1.0e-4, substeps=20, frame_stride=2, device="auto"):
    """Forward squeeze of a `geom` dough blob; return (X[F,N,3], times) of dough particles."""
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    cw, cd, ch = geom
    pos, vol, floor = block(grid, size=geom, ppc=2)
    s = Solver(grid=grid, device=device).load_particles(pos, vol)
    s.set_material(newtonian(eta=eta, density=1000.0, bulk_modulus=9.0e5).with_yield(tau_y))
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    cx = cy = grid.grid_lim * 0.5
    dough_top = floor + ch
    bh = (0.5 * cw + 0.015, 0.5 * cd + 0.015, 0.6 * grid.dx)
    be = WarpMPMBackend(solver=s)
    z = dough_top + bh[2]
    tool = be.attach_tool((cx, cy, z), bh)
    fdt = dt * substeps
    nf = round(press_strain * ch / v_plate / fdt)
    prev = z
    X, T = [], []
    for f in range(nf + 1):
        zn = z - v_plate * fdt if f > 0 else z
        vz = (zn - prev) / fdt
        if f > 0:
            be.set_tool_kinematics(tool, center=(cx, cy, prev), velocity=(0, 0, vz))
            be.step(dt, substeps)
        z = zn; prev = zn
        if f % frame_stride == 0:
            X.append(s.x().copy()); T.append(f * fdt)
    Xa = np.array(X)
    if not np.isfinite(Xa).all():
        raise RuntimeError(f"squeeze blew up (tau_y={tau_y}, eta={eta}): reduce dt/substeps")
    return Xa, np.array(T)


def render_speckle(X, times, bbox, out_dir, stem, width=520, dot_px=3.2, margin=0.06):
    """Front-camera speckle video (orthographic along -y -> x,z), shared `bbox` so every law
    uses ONE camera. track.py-compatible cam.json. Adapted from perception.render_collapse3d."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Xc, Yc, Zc = X[..., 0], X[..., 1], X[..., 2]
    x0, x1, z0, z1 = bbox
    wx = x1 - x0; scale = width / wx; height = int(np.ceil((z1 - z0) * scale))
    nP = X.shape[1]
    rng = np.random.default_rng(0); speckle = rng.uniform(0.4, 1.0, nP)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = out_dir / (stem + "_speckle")
    if fdir.exists():
        for o in fdir.glob("f_*.png"):
            o.unlink()
    fdir.mkdir(parents=True, exist_ok=True)
    dpi = 100; fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    files, saved = [], []
    for fi in range(X.shape[0]):
        ax.clear(); ax.set_facecolor("black")
        order = np.argsort(-Yc[fi])
        px = (Xc[fi][order] - x0) * scale; py = (z1 - Zc[fi][order]) * scale
        yn = (Yc[fi][order] - Yc[fi].min()) / max(np.ptp(Yc[fi]), 1e-9)
        bright = speckle[order] * (0.45 + 0.55 * (1 - yn))
        ax.scatter(px, py, c=bright, cmap="gray", s=dot_px ** 2, vmin=0, vmax=1, linewidths=0)
        ax.set_xlim(0, width); ax.set_ylim(height, 0); ax.axis("off")
        fp = fdir / f"f_{len(files):04d}.png"; fig.savefig(fp, dpi=dpi, facecolor="black")
        files.append(str(fp)); saved.append(int(fi))
    plt.close(fig)
    dt = float(times[1] - times[0])
    meta = {"width": width, "height": height, "scale_px_per_m": float(scale),
            "x0": float(x0), "z1": float(z1), "frame_dt": dt, "frame_indices": saved,
            "n_frames": len(files), "frames_dir": str(fdir)}
    cam = out_dir / (stem + "_cam.json")
    with open(cam, "w") as fh:
        json.dump(meta, fh, indent=2)
    subprocess.run(["ffmpeg", "-y", "-framerate", "20", "-i", str(fdir / "f_%04d.png"),
                    "-c:v", "libx264", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt", "yuv420p", str(out_dir / (stem + ".mp4"))],
                   check=True, capture_output=True)
    return str(cam)


def run(geom=(0.16, 0.16, 0.06), n_grid=52, query_spacing=9, device="auto"):
    from perception.track import track as cotracker_track
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"unseen volume {geom} (V={np.prod(geom)*1e6:.0f} cm^3); identified on "
          f"0.12x0.12x0.07.  rendering + CoTracker each law...\n")
    dumps = {n: squeeze_dump(ty, eta, geom, n_grid=n_grid, device=device)
             for n, (ty, eta) in LAWS.items()}
    allX = np.concatenate([X.reshape(-1, 3) for X, _ in dumps.values()], axis=0)
    m = 0.06
    bbox = (allX[:, 0].min() - m * 0.16, allX[:, 0].max() + m * 0.16,
            allX[:, 2].min() - m * 0.10, allX[:, 2].max() + m * 0.10)
    tracks = {}
    for name, (X, times) in dumps.items():
        cam = render_speckle(X, times, bbox, OUT, name)
        tr = cotracker_track(cam, out_path=str(OUT / f"{name}_tracks.npz"),
                             query_spacing=query_spacing, device="cpu")
        d = np.load(OUT / f"{name}_tracks.npz")
        tracks[name] = (d["world_tracks"], d["visibility"])
        print(f"  {name:8s}: tracked {tr['world_tracks'].shape[1]} pts over "
              f"{tr['world_tracks'].shape[0]} frames")

    Wt, Vt = tracks["truth"]
    scale = float(geom[0])                       # normalize deformation error by dough width
    print("\n[deformation rollout error vs truth]  (tracked material motion, % of dough width)")
    scores = {}
    for name, (W, V) in tracks.items():
        if name == "truth":
            continue
        n = min(W.shape[0], Wt.shape[0])
        vis = (V[:n] > 0.5) & (Vt[:n] > 0.5)
        disp = np.linalg.norm(W[:n] - Wt[:n], axis=-1)        # (T,N) per-track error
        per_t = np.array([disp[t][vis[t]].mean() if vis[t].any() else np.nan
                          for t in range(n)])
        final = float(np.nanmean(per_t[-5:])) / scale * 100
        peak = float(np.nanmax(per_t)) / scale * 100
        scores[name] = (final, peak, per_t)
        print(f"  {name:8s}: final {final:5.2f}%   peak {peak:5.2f}%  of dough width")

    _figure(dumps, tracks, scores, geom)
    scores["_device"] = device
    return scores


def _figure(dumps, tracks, scores, geom):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Wt = tracks["truth"][0]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.3))
    cols = {"learned": "#1c7ed6", "wrong": "#e8590c"}
    # left: deformation-error rollout vs time
    for name, (final, _peak, per_t) in scores.items():
        a0.plot(np.arange(len(per_t)), per_t / geom[0] * 100, color=cols.get(name, "0.5"),
                lw=2, label=f"{name} (final {final:.1f}%)")
    a0.set_xlabel("frame"); a0.set_ylabel("tracked deformation error vs truth  (% width)")
    a0.set_title("Predicted deformation rollout error (CoTracker)")
    a0.legend(fontsize=9); a0.grid(alpha=0.3)
    # right: final tracked positions (truth vs learned vs wrong) in x-z
    a1.scatter(Wt[-1][:, 0], Wt[-1][:, 1], s=8, c="#2b8a3e", label="truth", alpha=0.7)
    for name in ("learned", "wrong"):
        W = tracks[name][0]
        a1.scatter(W[-1][:, 0], W[-1][:, 1], s=8, c=cols[name], label=name, alpha=0.5)
    a1.set_xlabel("x (m)"); a1.set_ylabel("z (m)"); a1.set_aspect("equal")
    a1.set_title("final tracked shape (front view)"); a1.legend(fontsize=9); a1.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "predict_volume_rollout.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print("\nwrote", p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="Warp MPM device, e.g. cuda:0 or cuda:1")
    args = parser.parse_args()
    run(device=args.device)
