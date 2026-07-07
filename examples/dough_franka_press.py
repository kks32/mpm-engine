"""Two-way (force-feedback) Franka press: the dough STOPS the arm.

The end-effector is force-regulated, not scripted: a first-order admittance descends the
gripper until the dough's measured reaction reaches a target force, then holds. The stopping
depth is decided by the material, so the arm halts on the dough instead of driving through to
the floor. The loop is:

    read reaction wrench  ->  admittance picks descent velocity  ->  drive MPM tool (one
    control tick of substeps)  ->  read new reaction  ->  repeat,

with the MuJoCo Franka posed by inverting its EE kinematics so the gripper tip tracks the
tool (one shared metric frame). Renders the composite single view with a live force/depth
HUD; returns metrics (penetration, force balance, halt) for headless sweeps/tests.

Run:  ../.venv/bin/python examples/dough_franka_press.py
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, block, newtonian
from warpmpm.coupling.admittance import ForceAdmittance
from warpmpm.coupling.backend import WarpMPMBackend

OUT = Path(__file__).resolve().parents[1] / "out"


def run(n_grid=48, ticks=130, substeps=24, dt=2.0e-5, f_target=40.0, v_max=0.45,
        damping=None, allow_retract=False, eta=80.0, tau_y=900.0, density=1200.0,
        clearance=0.006, f_lowpass=0.25, render=True, out_name="dough_franka_press.mp4",
        device="auto"):
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    dough_size = (0.12, 0.10, 0.07)
    pos, vol, floor = block(grid, size=dough_size, ppc=2)
    s = Solver(grid=grid, device=device).load_particles(pos, vol)
    s.set_material(newtonian(eta=eta, density=density).with_yield(tau_y))
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    cx = cy = grid.grid_lim * 0.5
    box_half = (0.5 * dough_size[0] + 0.01, 0.5 * dough_size[1] + 0.01, 0.6 * grid.dx)
    dough_top = floor + dough_size[2]
    dt_ctrl = dt * substeps

    backend = WarpMPMBackend(solver=s)
    ctrl = ForceAdmittance(f_target=f_target, v_max=v_max, damping=damping,
                           allow_retract=allow_retract)

    z = dough_top + box_half[2] + clearance               # tool centre (MPM), start above top
    z_floor = floor + box_half[2] + 0.003                 # hard safety clamp (box bottom > floor)
    tool = backend.attach_tool((cx, cy, z), box_half, velocity=(0, 0, 0))

    # --- robot pose by EE-kinematics inversion (only if rendering) -------------------
    arm = a_grid = ee_z = None
    ex0 = ey0 = z_off = 0.0
    if render:
        from warpmpm.adapters.mujoco_adapter import FrankaArm
        arm = FrankaArm(height=600, width=800)
        a_grid = np.linspace(0.0, 1.0, 80)
        ee = np.array([arm.set_descent(float(a), dt_ctrl)["pos"] for a in a_grid])
        arm._prev_ee = None
        ee_z = ee[:, 2]
        ex0, ey0 = float(ee[len(ee) // 2, 0]), float(ee[len(ee) // 2, 1])
        z_off = float(np.interp(0.30, a_grid, ee_z)) - (z + box_half[2])  # const MPM->world z

    def a_of(box_top_mpm: float) -> float:
        return float(np.interp(box_top_mpm + z_off, ee_z[::-1], a_grid[::-1]))

    def to_world(p):
        out = np.empty_like(p)
        out[:, 0] = ex0 - cx + p[:, 0]
        out[:, 1] = ey0 - cy + p[:, 1]
        out[:, 2] = p[:, 2] + z_off
        return out

    if render:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        table_z = floor + z_off
        arm.cam.lookat[:] = [ex0, ey0, table_z + 0.04]
        arm.cam.distance = 0.85
        arm.cam.azimuth = 140
        arm.cam.elevation = -16
        tmp = Path(tempfile.mkdtemp())

    f_filt = 0.0
    f_max = 0.0
    log = []
    for t in range(ticks + 1):
        v_down = 0.0 if t == 0 else ctrl.velocity_down(f_filt)
        z_new = z - v_down * dt_ctrl
        z_new = max(z_new, z_floor)                       # safety net (physics should stop first)
        vz = (z_new - z) / dt_ctrl                        # actual tool velocity this tick
        if t > 0:
            backend.set_tool_kinematics(tool, center=(cx, cy, z), velocity=(0, 0, vz))
            backend.step(dt, substeps)
        z = z_new
        # static-contact stress integral for the force controller (the grid impulse collapses
        # at a quasi-static halt; see WarpMPMBackend.get_tool_wrench)
        w = backend.get_tool_wrench(tool, at_center=(cx, cy, z))
        f_react = float(w["Fz"])
        f_filt = f_lowpass * f_react + (1.0 - f_lowpass) * f_filt
        f_max = max(f_max, f_react)
        depth_mm = max(0.0, (dough_top - (z - box_half[2]))) * 1e3
        log.append((t * dt_ctrl, z, f_react, f_filt, w["n_contact"]))
        if render and arm is not None:
            arm.set_descent(a_of(z + box_half[2]), dt_ctrl, track_camera=False)
            x = s.x(); spd = np.linalg.norm(s.v(), axis=1)
            col = colormaps["YlOrBr_r"](np.clip(spd / 0.4, 0, 1)); col[:, 3] = 1.0
            rgb = arm.render_with_particles(to_world(x), col, radius=0.0035,
                                            table=(ex0, ey0, table_z, 0.28))
            fig = plt.figure(figsize=(8, 6), facecolor="black")
            ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(rgb); ax.axis("off")
            held = "HELD" if (abs(vz) < 0.01 and f_react > 0.5 * f_target) else \
                   ("CONTACT" if w["n_contact"] > 0 else "APPROACH")
            ax.text(0.02, 0.97, f"force-controlled press   target {f_target:.0f} N",
                    color="w", fontsize=12, transform=ax.transAxes, va="top")
            ax.text(0.02, 0.91, f"reaction Fz = {f_react:5.1f} N   [{held}]",
                    color="#ffd27f", fontsize=12, transform=ax.transAxes, va="top")
            ax.text(0.02, 0.86, f"press depth = {depth_mm:4.1f} mm   "
                    f"gap to floor = {(z - box_half[2] - floor) * 1e3:4.1f} mm",
                    color="#9fd0ff", fontsize=11, transform=ax.transAxes, va="top")
            # force gauge
            frac = float(np.clip(f_react / max(f_target, 1e-6), 0, 1.2))
            ax.barh(0.04, 0.4 * min(frac, 1.0), height=0.025, left=0.02, color="#ff8c42",
                    transform=ax.transAxes)
            ax.plot([0.42, 0.42], [0.03, 0.07], color="w", lw=1, transform=ax.transAxes)
            fig.savefig(tmp / f"f_{t:04d}.png", dpi=110, facecolor="black"); plt.close(fig)
        gap_mm = (z - box_half[2] - floor) * 1e3
        print(f"tick {t:3d}/{ticks} z={z:.4f} v={vz:+.3f} Fz={f_react:6.1f}N "
              f"depth={depth_mm:4.1f}mm gap={gap_mm:4.1f}mm nc={w['n_contact']}")

    box_bottom = z - box_half[2]
    zs = np.array([row[1] for row in log])
    # settled = net position drift over the last ~12 ticks is sub-mm (robust to the small
    # force limit-cycle), with a real reaction force present
    win = min(12, len(zs) - 1)
    settle_drift_mm = float(abs(zs[-1] - zs[-1 - win])) * 1e3 if win > 0 else 0.0
    f_tail = float(np.mean([row[2] for row in log[-win:]])) if win > 0 else log[-1][2]
    metrics = {
        "f_target": f_target, "f_final": log[-1][2], "f_tail_mean": f_tail, "f_max": f_max,
        "box_bottom_mpm": box_bottom, "floor_mpm": floor,
        "gap_to_floor_mm": (box_bottom - floor) * 1e3,
        "penetrated_floor": bool(box_bottom <= floor + 1e-4),
        "press_depth_mm": max(0.0, (dough_top - box_bottom)) * 1e3,
        "settle_drift_mm": settle_drift_mm,
        "halted": bool(settle_drift_mm < 0.8 and f_tail > 0.4 * f_target),
        "n_contact_final": int(log[-1][4]), "ticks": ticks, "device": device, "mp4": None,
    }
    if render and arm is not None:
        OUT.mkdir(exist_ok=True)
        mp4 = OUT / out_name
        subprocess.run(["ffmpeg", "-y", "-framerate", "16", "-i", str(tmp / "f_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)],
                       check=True, capture_output=True)
        metrics["mp4"] = str(mp4)
        print("wrote", mp4)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    parser.add_argument("--no-render", action="store_true", help="skip video rendering")
    args = parser.parse_args()
    m = run(device=args.device, render=not args.no_render)
    print("\nmetrics:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
