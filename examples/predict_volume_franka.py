"""Generalization test: learn the dough on ONE squeeze, predict a DIFFERENT volume.

The real test of whether we learned the physics (vs overfit one experiment): take the
rheology identified from the arm-plate squeeze and use it to PREDICT the plate force on a
dough of a different size/shape, then compare to the ground-truth simulation. We overlay
three forward runs on the new (bigger, flatter) blob:

  - truth                 (tau_y=200, eta=40)   -- what a real robot would measure
  - grid-impulse-learned  (tau_y=192, eta=55)   -- our recovered law (calibrated force)
  - stress-integral-learned (tau_y=384, eta=56) -- the old biased law (uncalibrated force)

If the learned law reproduces the truth on an unseen volume, the rheology transfers. The
biased law should over-predict, showing why the calibrated grid-impulse force mattered.
Forces are the Newton-exact grid impulse. Run:
  ../.venv/bin/python examples/predict_volume_franka.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, block, newtonian
from warpmpm.coupling.backend import WarpMPMBackend

OUT = Path(__file__).resolve().parents[1] / "out"

# laws to compare on the unseen volume (identified on a 0.12 x 0.12 x 0.07 blob)
LAWS = {
    "truth (200, 40)":               (200.0, 40.0, "#2b8a3e"),
    "grid-impulse learned (192, 55)": (192.0, 55.0, "#1c7ed6"),
    "stress-integral learned (384, 56)": (384.0, 56.0, "#e8590c"),
}


def press_force(tau_y, eta, geom, n_grid=56, v_plate=0.08, press_strain=0.5,
                dt=1.0e-4, substeps=20, density=1000.0, device="auto"):
    """Forward sim: squeeze a dough blob of size `geom` and return (strain%, F_z grid-impulse,
    final 95th-pct radial spread)."""
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    cw, cd, ch = geom
    pos, vol, floor = block(grid, size=geom, ppc=2)
    s = Solver(grid=grid, device=device).load_particles(pos, vol)
    s.set_material(newtonian(eta=eta, density=density, bulk_modulus=9.0e5).with_yield(tau_y))
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
        Fg = float(be.get_tool_reaction(tool, fdt)[2]) if f > 0 else 0.0
        strain.append((dough_top - (z - bh[2])) / ch * 100.0)
        Fz.append(Fg)
    x = s.x()
    spread = float(np.percentile(np.hypot(x[:, 0] - cx, x[:, 1] - cy), 95))
    return np.array(strain), np.array(Fz), spread


def run(geom=(0.16, 0.16, 0.06), n_grid=56, device="auto"):
    print(f"unseen volume {geom} (V={np.prod(geom)*1e6:.0f} cm^3) vs identified "
          f"0.12x0.12x0.07 (V={0.12*0.12*0.07*1e6:.0f} cm^3)\n")
    results = {}
    for name, (ty, eta, _c) in LAWS.items():
        st, Fz, sp = press_force(ty, eta, geom, n_grid=n_grid, device=device)
        results[name] = (st, Fz, sp)
        print(f"{name:36s}  F_peak={np.max(np.abs(Fz)):6.1f} N  spread={sp*1e3:5.1f} mm")

    # interpolate to a common strain grid and score predictions vs truth
    st_t, F_t, sp_t = results["truth (200, 40)"]
    grid_s = np.linspace(5, st_t.max() * 0.98, 80)
    F_t_i = np.interp(grid_s, st_t, np.abs(F_t))
    scores = {}
    for name, (st, Fz, sp) in results.items():
        if name == "truth (200, 40)":
            continue
        F_i = np.interp(grid_s, st, np.abs(Fz))
        ferr = float(np.linalg.norm(F_i - F_t_i) / np.linalg.norm(F_t_i))
        serr = abs(sp - sp_t) / sp_t
        scores[name] = (ferr, serr)
        print(f"\n{name}:  force rel-L2 vs truth = {ferr*100:4.1f}%   "
              f"spread err = {serr*100:4.1f}%")

    OUT.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for name, (_ty, _eta, c) in LAWS.items():
        st, Fz, _ = results[name]
        lbl = name if name == "truth (200, 40)" else \
            f"{name}  (force err {scores[name][0]*100:.0f}%)"
        ax.plot(st, np.abs(Fz), color=c, lw=2.2 if "truth" in name else 1.8,
                ls="-" if "truth" in name else "--", label=lbl)
    ax.set_xlabel("engineering strain  (%)")
    ax.set_ylabel("predicted plate force  |F_z|  (N)")
    ax.set_title(f"Predict an unseen dough volume {geom}\n"
                 "learned law vs ground truth (forward closed-loop test)")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "predict_volume_franka.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print("\nwrote", p)
    return {"scores": scores, "fig": str(p), "device": device}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    args = parser.parse_args()
    run(device=args.device)
