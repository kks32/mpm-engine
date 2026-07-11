"""Gate 2 for the quasi-static implicit solver: implicit press vs slow explicit press.

The same elastic column is pressed 1 percent by a plate, once with
QuasiStaticSolver (displacement-controlled node plane, 4 load steps) and once with
the explicit engine (box collider descending at 1 cm/s, grid-impulse tool_force).
As the explicit loading rate goes to zero its response approaches equilibrium, so
plate force and top-surface displacement must agree. Constitutive laws differ in
form (Hencky vs fixed corotated) but coincide to O(strain^2) at 1 percent.

Run:  python experiments/qs_press_ab.py
"""
from __future__ import annotations

import numpy as np

from warpmpm.core.solver import GridConfig, Solver
from warpmpm.implicit import QuasiStaticSolver
from warpmpm.materials import elastic

DX = 0.03
FLOOR_K = 3
COL = (0.12, 0.12, 0.36)
E, NU, RHO = 1.0e5, 0.3, 1000.0
STRAIN = 0.01
H, A = COL[2], COL[0] * COL[1]
DELTA = STRAIN * H


def column():
    org = np.array([4, 4, FLOOR_K]) * DX
    hp = DX / 2
    ax = [np.arange(hp / 2, c, hp) for c in COL]
    pos = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3) + org
    return pos, np.full(len(pos), hp ** 3), org


def run_implicit():
    pos, vol, org = column()
    qs = QuasiStaticSolver(pos, vol, rho=RHO, E=E, nu=NU,
                           grid=GridConfig(n_grid=20, grid_lim=20 * DX), device="cpu")
    qs.fix_floor(FLOOR_K, components="z")
    top_z = org[2] + H
    kz = np.arange(qs.n_nodes) % qs.ng[2]
    plate = qs._active & (kz * DX >= top_z - 0.6 * DX)
    n_steps = 4
    qs.prescribe_nodes(plate, (0.0, 0.0, -DELTA / n_steps), components="z")
    info = qs.solve_gravity(g=0.0, n_steps=n_steps)
    F = np.array(info["tool_force"])[-1]
    top = qs.x()[:, 2].max() - (org[2] + H)
    return F[2], top


def run_explicit():
    pos, vol, org = column()
    s = Solver(grid=GridConfig(n_grid=20, grid_lim=20 * DX), device="cpu")
    s.load_particles(pos.astype(np.float32), vol.astype(np.float32))
    # set_material defaults gravity to -9.81; the implicit side solves at g = 0, so
    # override it or the column settles under its own weight during the press
    s.set_material(elastic(E=E, nu=NU, density=RHO), g=[0.0, 0.0, 0.0])
    s.add_plane((0, 0, FLOOR_K * DX), (0, 0, 1), "slip")
    s.add_domain_walls()
    top_z = org[2] + H
    v_plate = 0.01                              # 1 cm/s: quasi-static loading
    half = (COL[0] / 2 + 2 * DX, COL[1] / 2 + 2 * DX, 2 * DX)
    center = np.array([org[0] + COL[0] / 2, org[1] + COL[1] / 2,
                       top_z + half[2]], dtype=float)
    handle = s.add_box(center, half, velocity=(0.0, 0.0, -v_plate))

    # explicit stability: dt from the elastic wave speed
    c = np.sqrt((E * (1 - NU) / ((1 + NU) * (1 - 2 * NU))) / RHO)
    dt = 0.28 * DX / c
    t_total = DELTA / v_plate
    ticks = 200
    substeps = int(np.ceil(t_total / ticks / dt))
    dt = t_total / ticks / substeps
    F_hist = []
    dt_tick = t_total / ticks
    zc = center[2]
    for _tick in range(ticks):
        zc -= v_plate * dt_tick
        s.reset_tool_force(handle)
        # start-of-tick center; the fork integrates the descent over the substeps
        s.set_box(handle, center=(center[0], center[1], zc + v_plate * dt_tick),
                  velocity=(0.0, 0.0, -v_plate))
        s.step(dt, substeps)
        F_hist.append(s.tool_force(handle, dt_tick))
    # hold the plate and damp the ring-down: the static reaction is the honest
    # quasi-static reference, not the mean of an oscillating press
    s._sim.mpm_model.grid_v_damping_scale = 0.999
    hold = []
    for _ in range(150):
        s.reset_tool_force(handle)
        s.set_box(handle, center=(center[0], center[1], zc), velocity=(0.0, 0.0, 0.0))
        s.step(dt, substeps)
        hold.append(s.tool_force(handle, dt_tick))
    F_end = np.mean([f[2] for f in hold[-30:]])
    top = s.x()[:, 2].max() - top_z
    return F_end, top, np.array(F_hist), np.array(hold)


if __name__ == "__main__":
    F_imp, top_imp = run_implicit()
    print(f"implicit:  F_z = {F_imp:8.2f} N   top displacement {top_imp*1000:+.2f} mm")
    F_exp, top_exp, hist, hold = run_explicit()
    print(f"explicit:  F_z = {F_exp:8.2f} N   top displacement {top_exp*1000:+.2f} mm"
          f"   (held, damped)")
    print(f"analytic:  F_z = {E * A * DELTA / H:8.2f} N (E A delta / h)")
    print(f"force ratio implicit/explicit: {F_imp / F_exp:.3f}")
    q = len(hist) // 4
    print("press-phase F_z trace (quarter means):",
          np.round([hist[i * q:(i + 1) * q, 2].mean() for i in range(4)], 2))
    print("hold-phase F_z (first 5, last 5):",
          np.round(hold[:5, 2], 2), np.round(hold[-5:, 2], 2))
