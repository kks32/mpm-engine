"""Single-view composite: a Franka gripper presses MPM dough on a table, rendered in ONE
MuJoCo camera (arm meshes + dough particles + table) -- the reference-image aesthetic.

The dough lives in the MPM domain; we map its particles into MuJoCo world coordinates,
placed under the gripper on a table, and render them as spheres in the same scene as the
arm. The gripper box (MPM) follows the Franka end-effector, so the visible gripper presses
the visible dough. Reaction force is read each frame.

Run:  ../.venv/bin/python examples/dough_franka_composite.py
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import colormaps

from warpmpm import GridConfig, Solver
from warpmpm.adapters.mujoco_adapter import FrankaArm
from warpmpm.coupling.wrench import box_contact_wrench
from warpmpm.scenes import block, dough

OUT = Path(__file__).resolve().parents[1] / "out"


def run(n_grid=48, ticks=50, substeps=30, dt=2.0e-5, render_every=1,
        clearance=0.02, press=0.05, device="auto"):
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    dough_size = (0.12, 0.10, 0.07)
    pos, vol, floor = block(grid, size=dough_size, ppc=2)
    s = Solver(grid=grid, device=device).load_particles(pos, vol)
    s.set_material(dough())
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    cx = cy = grid.grid_lim * 0.5
    box_half = (0.5 * dough_size[0] + 0.01, 0.5 * dough_size[1] + 0.01, 0.6 * grid.dx)
    dough_top = floor + dough_size[2]

    # box trajectory in MPM coords: a gentle press from `clearance` above the dough top down
    # to `press` deep, over frac in [0,1]. The press is the whole story (not a long approach).
    travel = clearance + press
    def box_z_of(frac: float) -> float:
        top0 = dough_top + box_half[2] + clearance      # box centre at frac=0
        return top0 - travel * frac

    arm = FrankaArm(height=600, width=800)
    dt_ctrl = dt * substeps

    # Probe the Franka EE z over the descent fraction, then INVERT it so the gripper tip
    # tracks the box top: a(frac) is chosen so ee_z(a) descends exactly `travel` over the
    # clip, ending at full-down. The MPM<->world map is then a constant offset (the honest
    # transform: gripper and dough share one metric frame).
    a_grid = np.linspace(0.0, 1.0, 80)
    ee = np.array([arm.set_descent(float(a), dt_ctrl)["pos"] for a in a_grid])
    arm._prev_ee = None
    ee_z = ee[:, 2]
    ex0, ey0 = float(ee[len(ee) // 2, 0]), float(ee[len(ee) // 2, 1])  # representative xy
    box_top0_mpm = box_z_of(0.0) + box_half[2]
    z_off = float(ee_z[-1]) + travel - box_top0_mpm      # constant MPM->world z offset

    def a_of(frac: float) -> float:
        # target gripper-tip world z, descending `travel` over the clip, ending full-down
        target = float(ee_z[-1]) + travel * (1.0 - frac)
        return float(np.interp(target, ee_z[::-1], a_grid[::-1]))  # ee_z decreasing in a

    def to_world(p):
        out = np.empty_like(p)
        out[:, 0] = ex0 - cx + p[:, 0]
        out[:, 1] = ey0 - cy + p[:, 1]
        out[:, 2] = p[:, 2] + z_off
        return out

    table_z = floor + z_off
    arm.cam.lookat[:] = [ex0, ey0, table_z + 0.04]
    arm.cam.distance = 0.85
    arm.cam.azimuth = 140
    arm.cam.elevation = -16

    frames, log = [], []
    prev_z = None
    tmp = Path(tempfile.mkdtemp())
    import imageio.v2 as imageio
    for t in range(ticks + 1):
        frac = t / ticks
        arm.set_descent(a_of(frac), dt_ctrl, track_camera=False)  # pose arm; fixed camera
        box_z = box_z_of(frac)                             # end-of-tick target (MPM coords)
        start_z = box_z if prev_z is None else prev_z      # start-of-tick box position
        vz = 0.0 if prev_z is None else (box_z - start_z) / dt_ctrl
        if t == 0:
            s.add_box((cx, cy, box_z), box_half, velocity=(0, 0, 0))
        else:
            # modify_bc integrates start_z -> start_z + dt_ctrl*vz = box_z over the substeps
            s.set_box(0, center=(cx, cy, start_z), velocity=(0, 0, vz))
            s.step(dt, substeps)
        prev_z = box_z
        w = box_contact_wrench(s.x(), s.cauchy(), s.vol(), (cx, cy, box_z), box_half)
        log.append((t * dt_ctrl, box_z, w["Fz"], w["n_contact"]))
        if t % render_every == 0:
            x = s.x(); spd = np.linalg.norm(s.v(), axis=1)
            col = colormaps["YlOrBr_r"](np.clip(spd / 0.5, 0, 1))
            col[:, 3] = 1.0
            img = arm.render_with_particles(to_world(x), col, radius=0.0035,
                                            table=(ex0, ey0, table_z, 0.28))
            imageio.imwrite(tmp / f"f_{len(frames):04d}.png", img)
            frames.append(1)
        print(f"tick {t:3d}/{ticks} box_z={box_z:.3f} Fz={w['Fz']:6.1f}N nc={w['n_contact']}")

    OUT.mkdir(exist_ok=True)
    mp4 = OUT / "dough_franka_composite.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", "12", "-i", str(tmp / "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)],
                   check=True, capture_output=True)
    print("wrote", mp4)
    return mp4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="Warp MPM device, e.g. cuda:0 or cuda:1")
    args = parser.parse_args()
    run(device=args.device)
