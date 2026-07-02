"""Wrist force-torque sensor reads back the MPM contact force (Newton's third law).

The dough's reaction on the plate is the EXACT MPM grid impulse (the simulation's load
cell at the contact). Here we feed that reaction to a DYNAMIC Franka in MuJoCo as an
external force on the hand, hold the arm with its position actuators, and read a wrist
force-torque sensor -- the load cell a real robot actually has. The wrist reading equals
the MPM reaction (the loop conserves force), which is the whole point: "the force from
MuJoCo" and "the exact MPM impulse" are the SAME number, measured at the wrist versus at
the contact. This validates the two-way readout end to end.

Run:  ../.venv/bin/python examples/wrist_ft_franka.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, block, newtonian
from warpmpm.adapters.mujoco_adapter import FrankaArm
from warpmpm.coupling.backend import WarpMPMBackend

OUT = Path(__file__).resolve().parents[1] / "out"


def run(n_grid=44, frames=70, substeps=20, dt=1.0e-4, v_plate=0.08, settle=220,
        every=2, device="auto"):
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    ds = (0.12, 0.12, 0.07)
    pos, vol, floor = block(grid, size=ds, ppc=2)
    s = Solver(grid=grid, device=device).load_particles(pos, vol)
    s.set_material(newtonian(eta=40.0, density=1000.0).with_yield(200.0))
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    cx = cy = grid.grid_lim * 0.5
    dough_top = floor + ds[2]
    bh = (0.5 * ds[0] + 0.015, 0.5 * ds[1] + 0.015, 0.6 * grid.dx)
    be = WarpMPMBackend(solver=s)
    z = dough_top + bh[2]
    tool = be.attach_tool((cx, cy, z), bh)
    fdt = dt * substeps

    arm = FrankaArm(ft_sensor=True)              # MuJoCo Franka with a wrist load cell
    prev = z
    rec = []                                     # (strain%, F_grid, F_wrist)
    for f in range(1, frames + 1):
        zn = z - v_plate * fdt
        vz = (zn - prev) / fdt
        be.set_tool_kinematics(tool, center=(cx, cy, prev), velocity=(0, 0, vz))
        be.reset_tool_force(tool)
        be.step(dt, substeps)
        z = zn; prev = zn
        f_grid = float(be.get_tool_reaction(tool, fdt)[2])      # exact MPM reaction (+z up)
        strain = (dough_top - (z - bh[2])) / ds[2]
        if f % every == 0:
            frac = 0.30 + 0.20 * float(np.clip(strain, 0, 1))   # arm descends as it presses
            f_wrist = float(arm.wrist_load_cell(frac, [0, 0, f_grid], settle=settle)[2])
            rec.append((strain * 100.0, f_grid, f_wrist))
            print(f"strain={strain*100:5.1f}%  F_grid={f_grid:6.2f} N   "
                  f"F_wrist={f_wrist:6.2f} N   diff={abs(f_wrist - f_grid):.3f} N")

    R = np.array(rec)
    fg, fw = R[:, 1], R[:, 2]
    relerr = float(np.linalg.norm(fw - fg) / max(np.linalg.norm(fg), 1e-9))
    print(f"\n[wrist FT vs MPM grid-impulse]  rel L2 diff = {relerr*100:.2f}%   "
          f"(wrist load cell reads back the MPM reaction)")

    OUT.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
    a0.plot(R[:, 0], fg, color="#d9480f", lw=2, label="MPM grid-impulse (at contact)")
    a0.plot(R[:, 0], fw, color="#1c7ed6", lw=0, marker="o", ms=4,
            label="MuJoCo wrist load cell")
    a0.set_xlabel("engineering strain  (%)"); a0.set_ylabel("plate reaction  F_z  (N)")
    a0.set_title("Two-way readout: wrist sensor = MPM reaction")
    a0.legend(fontsize=8); a0.grid(alpha=0.3)
    lim = [min(fg.min(), fw.min()), max(fg.max(), fw.max())]
    a1.plot(lim, lim, "k--", lw=1)
    a1.scatter(fg, fw, s=18, color="#1c7ed6", alpha=0.8)
    a1.set_xlabel("MPM grid-impulse  F_z  (N)"); a1.set_ylabel("MuJoCo wrist  F_z  (N)")
    a1.set_title(f"agreement (rel L2 {relerr*100:.1f}%)"); a1.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "wrist_ft_franka.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)
    return {"rel_l2": relerr, "n": len(rec), "fig": str(p), "device": device}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    args = parser.parse_args()
    run(device=args.device)
