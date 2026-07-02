"""Real-data-shaped ingestion + identification, evaluated on 1x and 1.5x dough volume.

Treats the simulation output the way a real rig would deliver it: a TEXTURED-SURFACE video
(-> CoTracker tracks, the deformation) plus a MEASURED plate force CSV (a load cell / wrist
FT sensor) plus camera calibration + plate kinematics. NOTHING from the simulator internals
is used downstream -- only the video, the force file, and the calibration.

Quasi-2D plane strain (thin y-slab, slip y/x walls) so the front-face surface tracks
represent the volume. The identification:
  tracks -> world (x,z) -> StreamFunctionField (div-free velocity field) -> strain rate |gd|
  -> volume integrals; the MEASURED force gives the plate power; the mechanical power balance
  INT tau:D = P_plate + P_grav - dKE/dt  with  INT tau:D = tau_y INT|gd| + eta INT|gd|^2
  is regressed over the press -> (tau_y, eta).

We build datasets for 1x and 1.5x volume, identify each from "measurements", and compare to
the ground-truth (tau_y, eta)=(200,40). Run:  ../.venv/bin/python examples/realdata_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from warpmpm import GridConfig, Solver, newtonian
from warpmpm.coupling.backend import WarpMPMBackend

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parents[1] / "out" / "realdata"
G = 9.81
EPS = 0.05   # match the warp-mpm kernel's shear-rate regularization
TRUTH = (200.0, 40.0)
LIGHT = np.array([0.3, 0.4, 0.86]); LIGHT = LIGHT / np.linalg.norm(LIGHT)
DOUGH = np.array([0.93, 0.80, 0.55])


def _loadj(p):
    return json.loads(Path(p).read_text())


def _savej(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=float))


# ----------------------------------------------------------------------------- dataset gen
def make_dataset(scale, tag, n_grid=64, v_plate=0.08, press_strain=0.5, dt=1.0e-4,
                 substeps=20, frame_stride=3, width=520, device="auto"):
    """Quasi-2D plane-strain squeeze; write a real-data-shaped dataset: textured-surface ortho
    frames + cam.json (perception.track homography) + force.csv (measured) + meta.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    dx = grid.dx
    s_lin = float(scale) ** 0.5
    col_w = 0.13 * s_lin; col_h = 0.06 * s_lin; slab = 6 * dx       # x, z scale; y slab fixed
    cx = cy = grid.grid_lim * 0.5
    floor = 3 * dx
    # build a thin y-slab block centred at cx,cy
    h = dx / 2
    xs = np.arange(cx - 0.5 * col_w + 0.5 * h, cx + 0.5 * col_w, h)
    ys = np.arange(cy - 0.5 * slab + 0.5 * h, cy + 0.5 * slab, h)
    zs = np.arange(floor + 0.5 * h, floor + col_h, h)
    pos = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    pos += np.random.default_rng(0).uniform(-0.2 * h, 0.2 * h, pos.shape).astype(np.float32)
    vol = np.full(len(pos), h ** 3, dtype=np.float32)
    s = Solver(grid=grid, device=device).load_particles(pos, vol)
    s.set_material(newtonian(eta=TRUTH[1], density=1000.0,
                             bulk_modulus=9.0e5).with_yield(TRUTH[0]))
    pad = 3 * dx
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")                 # no-slip floor
    s.add_plane((pad, 0, 0), (1, 0, 0), "slip")
    s.add_plane((grid.grid_lim - pad, 0, 0), (-1, 0, 0), "slip")
    s.add_plane((0, cy - 0.5 * slab, 0), (0, 1, 0), "slip")         # slip y-walls -> plane strain
    s.add_plane((0, cy + 0.5 * slab, 0), (0, -1, 0), "slip")
    bh = (0.5 * col_w + 0.012, 0.5 * slab + 0.012, 0.6 * dx)
    be = WarpMPMBackend(solver=s); z = floor + col_h + bh[2]; tool = be.attach_tool((cx, cy, z), bh)
    fdt = dt * substeps; nf = round(press_strain * col_h / v_plate / fdt)
    speckle = np.random.default_rng(1).uniform(0.0, 1.0, len(pos))
    # fixed ortho camera bbox over the whole press
    bb = [cx - 0.62 * col_w, cx + 0.62 * col_w, floor - 0.01, floor + col_h + 0.012]
    scale_px = width / (bb[1] - bb[0]); height = int(np.ceil((bb[3] - bb[2]) * scale_px))
    fdir = OUT / tag / "frames"; fdir.mkdir(parents=True, exist_ok=True)
    for o in fdir.glob("f_*.png"):
        o.unlink()
    prev = z; rows = []; saved = []; k = 0
    for f in range(nf + 1):
        zn = z - v_plate * fdt if f > 0 else z; vz = (zn - prev) / fdt
        if f > 0:
            be.set_tool_kinematics(tool, center=(cx, cy, prev), velocity=(0, 0, vz))
            be.reset_tool_force(tool); be.step(dt, substeps)
        z = zn; prev = zn
        Fz = abs(float(be.get_tool_reaction(tool, fdt)[2])) if f > 0 else 0.0
        rows.append((f * fdt, Fz))                                  # MEASURED force (load cell)
        if f % frame_stride:
            continue
        x = s.x()
        # marching-cubes surface -> front (x,z) projection, painter-sorted by y, textured
        mn = x.min(0) - 0.01; mx = x.max(0) + 0.01; cell = 0.0032
        dims = np.ceil((mx - mn) / cell).astype(int)
        rng = [(mn[i], mn[i] + dims[i] * cell) for i in range(3)]
        Hd, _ = np.histogramdd(x, bins=dims, range=rng)
        Hd = gaussian_filter(Hd.astype(float), 1.2); Hd /= max(Hd.max(), 1e-9)
        vts, faces, _, _ = marching_cubes(Hd, level=0.2)
        vw = mn + vts * cell
        fn2 = np.cross(vw[faces[:, 1]] - vw[faces[:, 0]], vw[faces[:, 2]] - vw[faces[:, 0]])
        fn2 /= np.linalg.norm(fn2, axis=1, keepdims=True) + 1e-9
        sh = np.clip(fn2 @ LIGHT, 0, 1) * 0.55 + 0.45
        from scipy.spatial import cKDTree
        _, idx = cKDTree(x).query(vw[faces].mean(1))
        tex = np.clip(0.55 + 0.6 * speckle[idx], 0.4, 1.25)
        fc = np.clip((sh * tex)[:, None] * DOUGH, 0, 1)
        order = np.argsort(vw[faces].mean(1)[:, 1])                 # far (large y) first
        polys = [vw[faces[i]][:, [0, 2]] for i in order]            # (x,z) front projection
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
        ax.add_collection(PolyCollection(polys, facecolors=fc[order], edgecolors="none"))
        ax.set_xlim(bb[0], bb[1]); ax.set_ylim(bb[2], bb[3]); ax.axis("off")
        fig.savefig(fdir / f"f_{k:04d}.png", dpi=100, facecolor="black"); plt.close(fig)
        saved.append(f); k += 1
    rows = np.array(rows)
    np.savetxt(OUT / tag / "force.csv", rows, delimiter=",", header="t,F_z", comments="")
    cam = {"width": width, "height": height, "scale_px_per_m": float(scale_px),
           "x0": float(bb[0]), "z1": float(bb[3]), "frame_dt": fdt * frame_stride,
           "frame_indices": saved, "n_frames": k, "frames_dir": str(fdir)}
    _savej(OUT / tag / "cam.json", cam)
    meta = {"v_plate": v_plate, "slab": slab, "rho": 1000.0, "col_w": col_w, "col_h": col_h,
            "floor": floor, "cx": cx, "frame_dt": fdt * frame_stride, "scale": scale,
            "device": device}
    _savej(OUT / tag / "meta.json", meta)
    subprocess.run(["ffmpeg", "-y", "-framerate", "14", "-i", str(fdir / "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(OUT / tag / "dough.mp4")],
                   check=True, capture_output=True)
    print(f"[{tag}] dataset: {k} frames, scale {scale}x (col_w={col_w:.3f}, col_h={col_h:.3f})")
    return OUT / tag


# ----------------------------------------------------------------------- ingestion + identify
def identify(tag, t_lo_frac=0.12, t_hi_frac=0.92, reuse_tracks=True):
    from ident.weakform.field_reconstruction import StreamFunctionField
    from perception.track import track
    d = OUT / tag
    meta = _loadj(d / "meta.json")
    force = np.loadtxt(d / "force.csv", delimiter=",", skiprows=1)     # (t, F) measured
    tp = d / "tracks.npz"
    if reuse_tracks and tp.exists():                  # tracks are eps-independent: reuse them
        _t = np.load(tp)
        tr = {k: _t[k] for k in _t.files}
    else:
        tr = track(str(d / "cam.json"), out_path=str(tp), query_spacing=6, device="cpu")
    W = tr["world_tracks"]; vis = tr["visibility"]; times = tr["times"]   # (T,N,2)=(x,z)
    n = len(times); dt = float(meta["frame_dt"]); rho = meta["rho"]; slab = meta["slab"]
    vp = meta["v_plate"]
    # central-difference world velocities along tracks
    V = np.zeros_like(W)
    V[1:-1] = (W[2:] - W[:-2]) / (2 * dt); V[0] = (W[1] - W[0]) / dt; V[-1] = (W[-1] - W[-2]) / dt
    Fmeas = np.interp(times, force[:, 0], force[:, 1])
    rows = []
    KE = np.zeros(n)
    cache = []
    for f in range(n):
        m = vis[f] > 0.6
        if m.sum() < 30:
            cache.append(None); continue
        x = W[f][m]; v = V[f][m]
        fld = StreamFunctionField(x, v, n_knots=(14, 8), lam_smooth=4e-4)
        # integration grid over the material footprint (track bbox), masked to coverage
        x0, x1 = x[:, 0].min(), x[:, 0].max(); z0, z1 = x[:, 1].min(), x[:, 1].max()
        gx = np.linspace(x0, x1, 40); gz = np.linspace(z0, z1, 24)
        GX, GZ = np.meshgrid(gx, gz); grid = np.stack([GX.ravel(), GZ.ravel()], 1)
        cell_a = (gx[1] - gx[0]) * (gz[1] - gz[0])
        # coverage mask: grid cells near a track
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(x).query(grid)
        cov = dist < 1.6 * max(gx[1] - gx[0], gz[1] - gz[0])
        L = fld.grad_v(grid)                                          # (M,2,2)
        D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
        # StreamFunction field is divergence-free so D is already deviatoric (plane strain,
        # sigma_yy carries no in-plane shear), matching the kernel's dev(D); only eps differs.
        gd = np.sqrt(2.0 * np.einsum("...ij,...ij->...", D, D) + EPS ** 2)
        vg = fld.velocity(grid)
        X1 = float(np.sum(gd[cov]) * cell_a * slab)
        X2 = float(np.sum(gd[cov] ** 2) * cell_a * slab)
        Pg = float(np.sum(rho * (-G) * vg[cov, 1]) * cell_a * slab)
        KE[f] = float(np.sum(0.5 * rho * (vg[cov] ** 2).sum(1)) * cell_a * slab)
        cache.append((X1, X2, Pg))
    tlo, thi = t_lo_frac * times[-1], t_hi_frac * times[-1]
    for f in range(n):
        if cache[f] is None or not (tlo <= times[f] <= thi):
            continue
        X1, X2, Pg = cache[f]
        dKE = (KE[min(f + 1, n - 1)] - KE[max(f - 1, 0)]) / (2 * dt)
        P_plate = vp * Fmeas[f]
        diss = P_plate + Pg - dKE
        rows.append((X1, X2, diss))
    R = np.array(rows); A = R[:, :2]; b = R[:, 2]
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    relres = float(np.linalg.norm(A @ theta - b) / max(np.linalg.norm(b), 1e-9))
    out = {"tag": tag, "tau_y_hat": float(theta[0]), "eta_hat": float(theta[1]),
           "tau_y_err": abs(theta[0] - TRUTH[0]) / TRUTH[0],
           "eta_err": abs(theta[1] - TRUTH[1]) / TRUTH[1], "n_times": len(rows),
           "fit_relres": relres, "cond": float(np.linalg.cond(A.T @ A))}
    print(f"[{tag}]  recovered (tau_y, eta) = ({out['tau_y_hat']:.0f}, {out['eta_hat']:.1f})  "
          f"truth (200, 40)  err ({out['tau_y_err']*100:.0f}%, {out['eta_err']*100:.0f}%)  "
          f"n={out['n_times']} relres={relres:.2f}")
    _savej(d / "identify.json", out)
    return out


def _slab_force_series(scale, tau_y, eta, n_grid=64, v_plate=0.08, press_strain=0.5,
                       dt=1.0e-4, substeps=20, device="auto"):
    """Re-sim the same quasi-2D slab squeeze with a given law; return (strain%, |Fz|)."""
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4); dx = grid.dx; s_lin = float(scale) ** 0.5
    col_w = 0.13 * s_lin; col_h = 0.06 * s_lin; slab = 6 * dx; cx = cy = 0.2; floor = 3 * dx
    h = dx / 2
    xs = np.arange(cx - 0.5 * col_w + 0.5 * h, cx + 0.5 * col_w, h)
    ys = np.arange(cy - 0.5 * slab + 0.5 * h, cy + 0.5 * slab, h)
    zs = np.arange(floor + 0.5 * h, floor + col_h, h)
    pos = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    pos += np.random.default_rng(0).uniform(-0.2 * h, 0.2 * h, pos.shape).astype(np.float32)
    s = Solver(grid=grid, device=device).load_particles(pos, np.full(len(pos), h ** 3, np.float32))
    s.set_material(newtonian(eta=eta, density=1000.0, bulk_modulus=9.0e5).with_yield(tau_y))
    pad = 3 * dx
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    s.add_plane((pad, 0, 0), (1, 0, 0), "slip")
    s.add_plane((grid.grid_lim - pad, 0, 0), (-1, 0, 0), "slip")
    s.add_plane((0, cy - 0.5 * slab, 0), (0, 1, 0), "slip")
    s.add_plane((0, cy + 0.5 * slab, 0), (0, -1, 0), "slip")
    bh = (0.5 * col_w + 0.012, 0.5 * slab + 0.012, 0.6 * dx)
    be = WarpMPMBackend(solver=s); z = floor + col_h + bh[2]; tool = be.attach_tool((cx, cy, z), bh)
    fdt = dt * substeps; nf = round(press_strain * col_h / v_plate / fdt); prev = z
    st, Fz = [], []
    for f in range(nf + 1):
        zn = z - v_plate * fdt if f > 0 else z; vz = (zn - prev) / fdt
        if f > 0:
            be.set_tool_kinematics(tool, center=(cx, cy, prev), velocity=(0, 0, vz))
            be.reset_tool_force(tool); be.step(dt, substeps)
        z = zn; prev = zn
        st.append((floor + col_h - (z - bh[2])) / col_h * 100)
        Fz.append(abs(float(be.get_tool_reaction(tool, fdt)[2])) if f > 0 else 0.0)
    return np.array(st), np.array(Fz)


def rollout_error(tag, device="auto"):
    """Re-sim the RECOVERED law and compare its force rollout to ground truth -- the metric
    that matters (tau_y/eta trade off; what counts is reproducing the dynamics)."""
    r = _loadj(OUT / tag / "identify.json")
    scale = _loadj(OUT / tag / "meta.json")["scale"]
    st, F_t = _slab_force_series(scale, *TRUTH, device=device)
    _, F_r = _slab_force_series(scale, r["tau_y_hat"], r["eta_hat"], device=device)
    w = st > 5
    relL2 = float(np.linalg.norm((F_r - F_t)[w]) / max(np.linalg.norm(F_t[w]), 1e-9))
    print(f"[{tag}]  PARAM err (tau_y,eta)=({r['tau_y_err']*100:.0f}%,{r['eta_err']*100:.0f}%) "
          f"-> ROLLOUT force err {relL2*100:.1f}% (law {r['tau_y_hat']:.0f},{r['eta_hat']:.0f})")
    return {"tag": tag, "scale": scale, "rollout_relL2": relL2, "st": st, "F_t": F_t,
            "F_r": F_r, "param_err": (r["tau_y_err"], r["eta_err"]),
            "law": (r["tau_y_hat"], r["eta_hat"])}


def eval_rollout(tags=("vol_1x", "vol_1p5x"), device="auto"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = [rollout_error(t, device=device) for t in tags]
    fig, axes = plt.subplots(1, len(res), figsize=(5.4 * len(res), 4.2))
    for ax, r in zip(np.atleast_1d(axes), res, strict=False):
        ax.plot(r["st"], r["F_t"], color="#2b8a3e", lw=2.4, label="ground truth (200,40)")
        ax.plot(r["st"], r["F_r"], color="#1c7ed6", lw=2.0, ls="--",
                label=f"recovered ({r['law'][0]:.0f},{r['law'][1]:.0f})")
        ax.set_title(f"{r['tag']} ({r['scale']}x vol)\nparam err "
                     f"({r['param_err'][0]*100:.0f}%,{r['param_err'][1]*100:.0f}%)  ->  "
                     f"ROLLOUT err {r['rollout_relL2']*100:.0f}%", fontsize=10)
        ax.set_xlabel("strain (%)"); ax.set_ylabel("plate force |F_z| (N)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); p = OUT / "rollout_error.png"; fig.savefig(p, dpi=130); plt.close(fig)
    print("\nrecovered-PARAMS scatter widely, but the re-sim FORCE ROLLOUT matches truth:")
    for r in res:
        print(f"  {r['tag']:10s}: param err ~{max(r['param_err'])*100:.0f}%  ->  "
              f"rollout force err {r['rollout_relL2']*100:.0f}%")
    print("wrote", p)
    return res


def run(device="auto"):
    OUT.mkdir(parents=True, exist_ok=True)
    make_dataset(1.0, "vol_1x", device=device); identify("vol_1x")
    make_dataset(1.5, "vol_1p5x", device=device); identify("vol_1p5x")
    return eval_rollout(device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    args = parser.parse_args()
    run(device=args.device)
