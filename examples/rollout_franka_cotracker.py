"""3D arm-driven dough squeeze, rendered WITH the Franka, deformation extracted by CoTracker,
and the rollout-prediction error (learned law vs ground truth).

For each law we run the same displacement-controlled squeeze of an unseen dough volume,
render the full 3D scene (MuJoCo Franka + mounted plate + dough) in one camera with the
dough as warm-colored speckle (trackable, and colour-separable from the grey arm/table), run
CoTracker3 on the dough to extract the material deformation, then compare the predicted
rollout (learned law) to the truth rollout. The error is normalized by how far the material
actually moves: "the predicted deformation differs from truth by X% of the observed motion."
Arm motion is identical for both laws (displacement control), so the difference isolates the
dough. Run:  ../.venv/bin/python examples/rollout_franka_cotracker.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, block, newtonian
from warpmpm.adapters.mujoco_adapter import FrankaArm
from warpmpm.coupling.backend import WarpMPMBackend

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parents[1] / "out" / "rollout_arm"
LAWS = {"truth": (200.0, 40.0), "learned": (192.0, 55.0)}


def _cotrack(frames_dir, spacing=7, device="cpu"):
    """Run CoTracker3 on a frame folder; query only DOUGH (warm/red-dominant) pixels. Returns
    pixel tracks (T,N,2), visibility (T,N)."""
    import torch
    from PIL import Image
    files = sorted(Path(frames_dir).glob("f_*.png"))
    imgs = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files]).astype(np.float32)
    video = torch.from_numpy(imgs).permute(0, 3, 1, 2)[None]        # (1,T,3,H,W)
    f0 = imgs[0]
    H, W = f0.shape[:2]
    ys = np.arange(spacing, H - spacing, spacing)
    xs = np.arange(spacing, W - spacing, spacing)
    GX, GY = np.meshgrid(xs, ys)
    pts = np.stack([GX.ravel(), GY.ravel()], -1).astype(np.float32)
    r = f0[pts[:, 1].astype(int), pts[:, 0].astype(int), 0]
    g = f0[pts[:, 1].astype(int), pts[:, 0].astype(int), 1]
    b = f0[pts[:, 1].astype(int), pts[:, 0].astype(int), 2]
    dough = (r > b + 12) & (g > b + 4) & (r > 70)                   # warm (low blue) = pizza dough
    pts = pts[dough]
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1)
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()
    with torch.no_grad():
        tr_, vis = model(video.to(device), queries=torch.from_numpy(q)[None].to(device))
    return tr_[0].cpu().numpy(), vis[0].cpu().numpy()


def squeeze_composite(arm, ee, tau_y, eta, geom, stem, n_grid=52, v_plate=0.08,
                      press_strain=0.5, dt=1.0e-4, substeps=20, frame_stride=3,
                      device="auto"):
    """Run the squeeze for one law and render the composite (arm + plate + speckled dough)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    # pizza-dough colour: a warm cream/tan with per-particle brightness flecks (trackable),
    # warm (R,G >> B) so it separates from the neutral-grey arm/table for the CoTracker mask
    sp = np.random.default_rng(0).uniform(0.82, 1.0, len(pos))
    base = np.array([1.0, 0.78, 0.34])                        # pizza-dough golden tan
    rgba = np.column_stack([sp * base[0], sp * base[1], sp * base[2], np.ones_like(sp)])
    fdir = OUT / stem
    fdir.mkdir(parents=True, exist_ok=True)
    for o in fdir.glob("f_*.png"):
        o.unlink()
    prev = z
    k = 0
    for f in range(nf + 1):
        zn = z - v_plate * fdt if f > 0 else z
        vz = (zn - prev) / fdt
        if f > 0:
            be.set_tool_kinematics(tool, center=(cx, cy, prev), velocity=(0, 0, vz))
            be.step(dt, substeps)
        z = zn; prev = zn
        if f % frame_stride == 0:
            arm.set_descent(ee["a_of"](z + bh[2]), fdt, track_camera=False)
            x = s.x()
            plate_world = (ee["ex0"], ee["ey0"], z + ee["z_off"])
            rgb = arm.render_with_particles(
                ee["to_world"](x), rgba, radius=0.0033,
                table=(ee["ex0"], ee["ey0"], floor + ee["z_off"], 0.30),
                boxes=[(plate_world, bh, (0.72, 0.74, 0.8, 1.0))])
            fig = plt.figure(figsize=(rgb.shape[1] / 100, rgb.shape[0] / 100), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(rgb); ax.axis("off")
            fig.savefig(fdir / f"f_{k:04d}.png", dpi=100); plt.close(fig)
            k += 1
    subprocess.run(["ffmpeg", "-y", "-framerate", "16", "-i", str(fdir / "f_%04d.png"),
                    "-c:v", "libx264", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt", "yuv420p", str(OUT / f"{stem}.mp4")], check=True,
                   capture_output=True)
    return str(fdir)


def run(geom=(0.16, 0.16, 0.06), n_grid=52, width=720, height=520, device="auto"):
    OUT.mkdir(parents=True, exist_ok=True)
    arm = FrankaArm(height=height, width=width, hide_gripper=True)
    # shared EE-inversion + fixed camera (identical arm motion for every law)
    ch = geom[2]
    dx = 0.4 / n_grid
    floor = 3 * dx; dough_top = floor + ch; plate_hz = 0.6 * dx
    cx = cy = 0.2
    a_grid = np.linspace(0.0, 1.0, 80)
    ee_xyz = np.array([arm.set_descent(float(a), 1.0)["pos"] for a in a_grid])
    arm._prev_ee = None
    ez = ee_xyz[:, 2]
    ex0, ey0 = float(ee_xyz[len(ee_xyz) // 2, 0]), float(ee_xyz[len(ee_xyz) // 2, 1])
    z0_box = dough_top + plate_hz
    z_off = float(np.interp(0.28, a_grid, ez)) - (z0_box + plate_hz)

    def a_of(box_top):
        return float(np.interp(box_top + z_off, ez[::-1], a_grid[::-1]))

    def to_world(p):
        out = np.empty_like(p)
        out[:, 0] = ex0 - cx + p[:, 0]; out[:, 1] = ey0 - cy + p[:, 1]
        out[:, 2] = p[:, 2] + z_off
        return out
    ee = {"a_of": a_of, "to_world": to_world, "ex0": ex0, "ey0": ey0, "z_off": z_off}
    arm.cam.lookat[:] = [ex0, ey0, floor + z_off + 0.03]
    arm.cam.distance = 0.52; arm.cam.azimuth = 140; arm.cam.elevation = -12  # zoom on gripper+dough

    print(f"unseen volume {geom}; rendering arm-driven squeeze + CoTracker per law...\n")
    fdirs, tracks = {}, {}
    for name, (ty, eta) in LAWS.items():
        fdirs[name] = squeeze_composite(arm, ee, ty, eta, geom, name, n_grid=n_grid,
                                        device=device)
        tr, vis = _cotrack(fdirs[name])
        tracks[name] = (tr, vis)
        print(f"  {name:8s}: CoTracker tracked {tr.shape[1]} dough pts over {tr.shape[0]} frames")

    Tt, Vt = tracks["truth"]; Tl, Vl = tracks["learned"]
    n = min(Tt.shape[0], Tl.shape[0])
    vis = (Vt[:n] > 0.5) & (Vl[:n] > 0.5)
    moved = np.linalg.norm(Tt[:n] - Tt[0][None], axis=-1)            # truth displacement (px)
    diff = np.linalg.norm(Tl[:n] - Tt[:n], axis=-1)                  # pred-vs-truth (px)
    err_t = np.array([(diff[t][vis[t]].sum() / max(moved[t][vis[t]].sum(), 1e-6))
                      for t in range(n)])
    final = float(np.nanmean(err_t[-5:]) * 100)
    print(f"\n[rollout prediction error] learned vs truth = {final:.2f}% of the observed "
          f"material motion (CoTracker, dough only)")
    _figure(tracks, err_t, final, geom)
    _side_by_side(fdirs, tracks, err_t)
    return {"rollout_err_pct": final, "device": device}


def _figure(tracks, err_t, final, geom):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Tt = tracks["truth"][0]; Tl = tracks["learned"][0]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.3))
    a0.plot(np.arange(len(err_t)), err_t * 100, color="#1c7ed6", lw=2)
    a0.set_xlabel("frame"); a0.set_ylabel("rollout error (% of observed motion)")
    a0.set_title(f"Deformation rollout prediction error (final {final:.1f}%)")
    a0.grid(alpha=0.3)
    a1.scatter(Tt[-1][:, 0], Tt[-1][:, 1], s=10, c="#2b8a3e", label="truth tracks", alpha=0.6)
    a1.scatter(Tl[-1][:, 0], Tl[-1][:, 1], s=10, c="#1c7ed6", label="learned tracks", alpha=0.5)
    a1.invert_yaxis(); a1.set_aspect("equal"); a1.set_xlabel("px"); a1.set_ylabel("px")
    a1.set_title("final tracked dough (image space)"); a1.legend(fontsize=9)
    fig.tight_layout()
    p = OUT / "rollout_franka_cotracker.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)


def _side_by_side(fdirs, tracks, err_t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    ft = sorted(Path(fdirs["truth"]).glob("f_*.png"))
    fl = sorted(Path(fdirs["learned"]).glob("f_*.png"))
    Tt = tracks["truth"][0]; Tl = tracks["learned"][0]
    tmp = OUT / "_sbs"; tmp.mkdir(exist_ok=True)
    for o in tmp.glob("*.png"):
        o.unlink()
    n = min(len(ft), len(fl), Tt.shape[0])
    for i in range(n):
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="black")
        for ax, fp, T, ttl in ((a0, ft[i], Tt, "truth (200,40)"),
                               (a1, fl[i], Tl, "learned (192,55)")):
            ax.imshow(np.asarray(Image.open(fp)))
            ax.scatter(T[i][:, 0], T[i][:, 1], s=4, c="#74e000", alpha=0.7)
            ax.axis("off"); ax.set_title(ttl, color="w", fontsize=10)
        fig.suptitle(f"arm-driven squeeze + CoTracker dough tracks   "
                     f"rollout err {err_t[i]*100:4.1f}%", color="w", fontsize=11)
        fig.savefig(tmp / f"s_{i:04d}.png", dpi=96, facecolor="black"); plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-framerate", "14", "-i", str(tmp / "s_%04d.png"),
                    "-c:v", "libx264", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt", "yuv420p", str(OUT / "rollout_sbs.mp4")], check=True,
                   capture_output=True)
    print("wrote", OUT / "rollout_sbs.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="Warp MPM device, e.g. cuda:0 or cuda:1")
    args = parser.parse_args()
    run(device=args.device)
