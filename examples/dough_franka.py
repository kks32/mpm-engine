"""Franka arm pressing an MPM dough blob -- the full robot <-> deformable integration.

A scripted Franka descent (MuJoCo, native on this Mac) drives a kinematic gripper box in
the Warp-MPM domain; the dough deforms, and we read the reaction wrench the dough exerts
on the gripper (Newton's third law, stress-integral estimator). Renders a 2-panel video:
the MuJoCo Franka (left) and the MPM dough + gripper box coloured by speed (right), with a
live reaction-force readout -- the reference-image aesthetic, robot + material + labels.

Run:  ../.venv/bin/python examples/dough_franka.py
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from warpmpm import GridConfig, Solver
from warpmpm.adapters.mujoco_adapter import FrankaArm
from warpmpm.coupling.wrench import box_contact_wrench
from warpmpm.scenes import block, dough

OUT = Path(__file__).resolve().parents[1] / "out"


def _box_edges(c, h):
    cx, cy, cz = c; hx, hy, hz = h
    pts = np.array([[sx, sy, sz] for sx in (cx - hx, cx + hx)
                    for sy in (cy - hy, cy + hy) for sz in (cz - hz, cz + hz)])
    # 12 edges of the box
    E = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return [[pts[a], pts[b]] for a, b in E]


def run(n_grid=48, ticks=60, substeps=40, dt=2.0e-5, press_depth=0.025,
        render_every=2, device="auto"):
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    dough_size = (0.12, 0.08, 0.06)
    pos, vol, floor = block(grid, size=dough_size, ppc=2)
    s = Solver(grid=grid, device=device).load_particles(pos, vol)
    s.set_material(dough())
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    cx = cy = grid.grid_lim * 0.5
    dough_top = floor + dough_size[2]
    box_half = (0.5 * dough_size[0] + 0.01, 0.5 * dough_size[1] + 0.01, 0.6 * grid.dx)
    z0 = dough_top + box_half[2] + 0.005
    handle = s.add_box((cx, cy, z0), box_half, velocity=(0, 0, 0))

    arm = FrankaArm()
    dt_ctrl = dt * substeps
    frames, log = [], []
    prev_z = z0
    tmp = Path(tempfile.mkdtemp())
    for t in range(ticks + 1):
        frac = t / ticks
        arm.set_descent(frac, dt_ctrl)   # poses the Franka for the render
        box_z = z0 - press_depth * frac  # end-of-tick target
        vz = (box_z - prev_z) / dt_ctrl
        # drive from the START-of-tick position; modify_bc integrates it to box_z over
        # the substeps (passing box_z as center would leave the box one tick ahead)
        s.set_box(handle, center=(cx, cy, prev_z), velocity=(0, 0, vz))
        if t > 0:
            s.step(dt, substeps)
        prev_z = box_z
        w = box_contact_wrench(s.x(), s.cauchy(), s.vol(), (cx, cy, box_z), box_half)
        log.append((t * dt_ctrl, box_z, w["Fz"], w["n_contact"]))
        if t % render_every == 0:
            x = s.x(); spd = np.linalg.norm(s.v(), axis=1)
            fig = plt.figure(figsize=(11, 4.6), facecolor="white")
            axL = fig.add_subplot(1, 2, 1); axL.imshow(arm.render_rgb()); axL.axis("off")
            axL.set_title("Franka (MuJoCo)", fontsize=10)
            axR = fig.add_subplot(1, 2, 2, projection="3d")
            axR.scatter(x[:, 0], x[:, 1], x[:, 2], c=spd, cmap="YlOrBr_r", s=4,
                        vmin=0, vmax=0.6, depthshade=True, edgecolors="none")
            axR.add_collection3d(Line3DCollection(_box_edges((cx, cy, box_z), box_half),
                                                  colors="0.4", linewidths=1.2))
            axR.set_xlim(cx - 0.1, cx + 0.1); axR.set_ylim(cy - 0.1, cy + 0.1)
            axR.set_zlim(floor, z0 + 0.02); axR.set_box_aspect((1, 1, 0.8))
            axR.set_xticks([]); axR.set_yticks([]); axR.set_zticks([])
            axR.set_title("MPM dough + gripper", fontsize=10)
            fig.suptitle(f"Franka presses dough   depth={press_depth*frac*1e3:4.1f} mm   "
                         f"reaction Fz={w['Fz']:6.2f} N", fontsize=12)
            fig.tight_layout()
            fp = tmp / f"f_{len(frames):04d}.png"
            fig.savefig(fp, dpi=120, facecolor="white"); plt.close(fig)
            frames.append(fp)
        print(f"tick {t:3d}/{ticks} z={box_z:.3f} Fz={w['Fz']:6.1f}N nc={w['n_contact']}")

    OUT.mkdir(exist_ok=True)
    mp4 = OUT / "dough_franka.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", "12", "-i", str(tmp / "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)],
                   check=True, capture_output=True)
    np.savez(OUT / "dough_franka.npz", log=np.array(log))
    print("wrote", mp4)
    return mp4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="Warp MPM device, e.g. cuda:0 or cuda:1")
    args = parser.parse_args()
    run(device=args.device)
