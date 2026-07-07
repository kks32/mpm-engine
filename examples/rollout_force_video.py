"""Predicted vs ground-truth on the unseen volume: side-by-side 3D render + a live force trace.

Reuses the arm-driven squeeze frames already rendered by rollout_franka_cotracker.py
(out/rollout_arm/{truth,learned}/), re-captures the Newton-exact grid-impulse plate force per
frame for each law, and assembles one video: ground-truth render | predicted render, with the
truth vs predicted plate-force-vs-strain trace below (marker at the current frame). Also writes
a clean static force graph. Run:  ../.venv/bin/python examples/rollout_force_video.py
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, block, newtonian
from warpmpm.coupling.backend import WarpMPMBackend

OUT = Path(__file__).resolve().parents[1] / "out" / "rollout_arm"
LAWS = {"truth": (200.0, 40.0), "learned": (192.0, 55.0)}


def force_series(tau_y, eta, geom, n_grid=52, v_plate=0.08, press_strain=0.5,
                 dt=1.0e-4, substeps=20, frame_stride=3, device="auto"):
    """Re-run the squeeze; record (strain%, grid-impulse |Fz|) at the rendered frames."""
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
    strain, Fz = [], []
    for f in range(nf + 1):
        zn = z - v_plate * fdt if f > 0 else z
        vz = (zn - prev) / fdt
        if f > 0:
            be.set_tool_kinematics(tool, center=(cx, cy, prev), velocity=(0, 0, vz))
            be.reset_tool_force(tool)
            be.step(dt, substeps)
        z = zn; prev = zn
        if f % frame_stride == 0:
            fzz = abs(float(be.get_tool_reaction(tool, fdt)[2])) if f > 0 else 0.0
            strain.append((dough_top - (z - bh[2])) / ch * 100.0)
            Fz.append(fzz)
    return np.array(strain), np.array(Fz)


def run(geom=(0.16, 0.16, 0.06), device="auto"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    st_t, F_t = force_series(*LAWS["truth"], geom, device=device)
    st_l, F_l = force_series(*LAWS["learned"], geom, device=device)
    # interp learned onto truth strain for the error number
    F_l_i = np.interp(st_t, st_l, F_l)
    ferr = float(np.linalg.norm(F_l_i - F_t) / max(np.linalg.norm(F_t), 1e-9)) * 100
    print(f"force prediction error (learned vs truth) = {ferr:.1f}%   "
          f"truth peak {F_t.max():.1f} N, predicted peak {F_l.max():.1f} N")

    # static force graph
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(st_t, F_t, color="#2b8a3e", lw=2.4, label="ground truth (200, 40)")
    ax.plot(st_l, F_l, color="#1c7ed6", lw=2.0, ls="--", label="predicted / learned (192, 55)")
    ax.set_xlabel("engineering strain  (%)"); ax.set_ylabel("plate force  |F_z|  (N)")
    ax.set_title(f"Plate force on the unseen volume: predicted vs truth (err {ferr:.0f}%)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    gpath = OUT / "rollout_force_graph.png"
    fig.savefig(gpath, dpi=130); plt.close(fig)
    print("wrote", gpath)

    # combined video: truth render | learned render, with the live force trace below
    ft = sorted((OUT / "truth").glob("f_*.png"))
    fl = sorted((OUT / "learned").glob("f_*.png"))
    n = min(len(ft), len(fl), len(F_t), len(F_l))
    tmp = OUT / "_fv"; tmp.mkdir(exist_ok=True)
    for o in tmp.glob("*.png"):
        o.unlink()
    fmax = max(F_t[:n].max(), F_l[:n].max()) * 1.1
    for i in range(n):
        fig = plt.figure(figsize=(11, 6.4), facecolor="black")
        aT = fig.add_axes([0.02, 0.40, 0.47, 0.55]); aL = fig.add_axes([0.51, 0.40, 0.47, 0.55])
        aT.imshow(np.asarray(Image.open(ft[i]))); aT.axis("off")
        aT.set_title("GROUND TRUTH  (200, 40)", color="#74e000", fontsize=11)
        aL.imshow(np.asarray(Image.open(fl[i]))); aL.axis("off")
        aL.set_title("PREDICTED  (learned 192, 55)", color="#4dabf7", fontsize=11)
        ax = fig.add_axes([0.08, 0.07, 0.86, 0.28]); ax.set_facecolor("#111")
        ax.plot(st_t[:n], F_t[:n], color="#2b8a3e", lw=2, label="truth")
        ax.plot(st_l[:n], F_l[:n], color="#4dabf7", lw=2, ls="--", label="predicted")
        ax.axvline(st_t[i], color="w", lw=1)
        ax.plot(st_t[i], F_t[i], "o", color="#74e000")
        ax.plot(st_l[i], F_l[i], "o", color="#4dabf7")
        ax.set_xlim(0, st_t[:n].max()); ax.set_ylim(0, fmax)
        ax.set_xlabel("strain (%)", color="w"); ax.set_ylabel("plate force |F_z| (N)", color="w")
        ax.tick_params(colors="w"); ax.legend(loc="upper left", fontsize=9, labelcolor="w")
        for sp in ax.spines.values():
            sp.set_color("w")
        fig.suptitle(f"Franka plate squeeze, unseen volume {geom} -- predicted vs ground truth"
                     f"   (force err {ferr:.0f}%)", color="w", fontsize=12)
        fig.savefig(tmp / f"v_{i:04d}.png", dpi=96, facecolor="black"); plt.close(fig)
    mp4 = OUT / "rollout_force_video.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", "12", "-i", str(tmp / "v_%04d.png"),
                    "-c:v", "libx264", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt", "yuv420p", str(mp4)], check=True, capture_output=True)
    print("wrote", mp4)
    return {"force_err_pct": ferr, "video": str(mp4), "graph": str(gpath), "device": device}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    args = parser.parse_args()
    run(device=args.device)
