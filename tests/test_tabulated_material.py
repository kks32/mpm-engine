"""Lock the tabulated-viscosity material (fork id 12) to the parametric Newtonian kernel.

The report claims the tabulated eta_app(gd) material reproduces the parametric Herschel-Bulkley
kernel to ~3 decimals when fed the same curve. This test enforces that: it shears a small dough
block once with newtonian(eta, tau_y, pk, pn) and once with tabulated_viscous sampled from the
SAME analytic eta_app(gd), and asserts the wall reaction forces (and particle velocities) agree.
A constant-viscosity table is also checked to differ, so the test is not vacuous.
"""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm import GridConfig, Solver, newtonian, tabulated_viscous
from warpmpm.coupling.backend import WarpMPMBackend

EPS = 0.05
HB = dict(eta=10.0, tau_y=40.0, pk=60.0, pn=0.4)


def _eta_app(gd):
    g = np.sqrt(gd ** 2 + EPS ** 2)
    return HB["eta"] + HB["tau_y"] / g + HB["pk"] * g ** (HB["pn"] - 1.0)


def _shear_once(material, n_frames=6, dt=1.0e-4, substeps=10):
    grid = GridConfig(n_grid=32, grid_lim=0.4)
    dx = grid.dx
    cx = cy = grid.grid_lim * 0.5
    floor = 3 * dx
    h = dx / 2
    geom = (0.10, 0.04, 0.045)
    xs = np.arange(cx - 0.5 * geom[0] + 0.5 * h, cx + 0.5 * geom[0], h)
    ys = np.arange(cy - 0.5 * geom[1] + 0.5 * h, cy + 0.5 * geom[1], h)
    zs = np.arange(floor + 0.5 * h, floor + geom[2], h)
    pos = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    pos += np.random.default_rng(0).uniform(-0.2 * h, 0.2 * h, pos.shape).astype(np.float32)
    s = Solver(grid=grid).load_particles(pos, np.full(len(pos), h ** 3, np.float32))
    s.set_material(material)
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    be = WarpMPMBackend(solver=s)
    bh = (0.45 * grid.grid_lim, 0.5 * geom[1] + 0.012, 0.6 * dx)
    z_wall = floor + geom[2]
    tool = be.attach_tool((cx, cy, z_wall), bh)
    fdt = dt * substeps
    F = np.zeros(3)
    for _f in range(n_frames):
        be.set_tool_kinematics(tool, center=(cx, cy, z_wall), velocity=(0.1, 0.0, 0.0))
        be.reset_tool_force(tool)
        be.step(dt, substeps)
        F = be.get_tool_reaction(tool, fdt)
    return np.asarray(F, float), s.v()


def test_tabulated_matches_parametric_kernel():
    s_grid = np.linspace(-1.0, 2.0, 128)
    tab_hb = _eta_app(10.0 ** s_grid)
    par = (newtonian(eta=HB["eta"], density=1000.0, bulk_modulus=9.0e5)
           .with_yield(HB["tau_y"]).with_powerlaw(K=HB["pk"], n=HB["pn"]))
    tab = tabulated_viscous(tab_hb, smin=-1.0, smax=2.0, density=1000.0, bulk_modulus=9.0e5)

    F_par, v_par = _shear_once(par)
    F_tab, v_tab = _shear_once(tab)
    # forces agree to a tight relative tolerance (the report's "to three decimals")
    rel_F = np.linalg.norm(F_tab - F_par) / max(np.linalg.norm(F_par), 1e-12)
    rel_v = np.linalg.norm(v_tab - v_par) / max(np.linalg.norm(v_par), 1e-12)
    assert rel_F < 5e-3, f"tabulated vs parametric force mismatch: relL2={rel_F:.2e}"
    assert rel_v < 5e-3, f"tabulated vs parametric velocity mismatch: relL2={rel_v:.2e}"


def test_tabulated_constant_differs_from_truth():
    # not vacuous: a wrong (constant) table gives a materially different force
    s_grid = np.linspace(-1.0, 2.0, 128)
    par = (newtonian(eta=HB["eta"], density=1000.0, bulk_modulus=9.0e5)
           .with_yield(HB["tau_y"]).with_powerlaw(K=HB["pk"], n=HB["pn"]))
    tab_const = tabulated_viscous(np.full_like(s_grid, 30.0), smin=-1.0, smax=2.0,
                                  density=1000.0, bulk_modulus=9.0e5)
    F_par, _ = _shear_once(par)
    F_c, _ = _shear_once(tab_const)
    rel = np.linalg.norm(F_c - F_par) / max(np.linalg.norm(F_par), 1e-12)
    assert rel > 0.05, f"constant table should differ from HB truth, got relL2={rel:.2e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
