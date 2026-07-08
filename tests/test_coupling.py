"""Two-way coupling tests: the force controllers, the backend contract, and the load-bearing
property that a force-regulated press into firm dough halts on the dough and never reaches the
floor (the reaction force, not a scripted endpoint, decides the stopping depth)."""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm import (
    ForceAdmittance,
    GridConfig,
    Impedance1D,
    Solver,
    WarpMPMBackend,
    block,
    newtonian,
)


def test_force_admittance_descends_in_free_space():
    # no reaction -> descend at the full speed cap (default damping = f_target/v_max)
    c = ForceAdmittance(f_target=40.0, v_max=0.4)
    assert c.velocity_down(0.0) == pytest.approx(0.4)


def test_force_admittance_halts_at_target():
    c = ForceAdmittance(f_target=40.0, v_max=0.4)
    assert c.velocity_down(40.0) == pytest.approx(0.0)


def test_force_admittance_retracts_on_overshoot():
    # reaction beyond target -> back off (regulate down to the target force)
    c = ForceAdmittance(f_target=40.0, v_max=0.4)
    assert c.velocity_down(80.0) < 0.0


def test_force_admittance_monotone_in_reaction():
    c = ForceAdmittance(f_target=40.0, v_max=0.4)
    f = np.linspace(0, 80, 20)
    v = np.array([c.velocity_down(fi) for fi in f])
    assert np.all(np.diff(v) <= 1e-12)


def test_impedance_converges_to_force_balance():
    # constant reaction -> settle where k*(z_eq - z_ref) = f_react, stable, no blow-up
    imp = Impedance1D(m=1.0, b=120.0, k=4000.0, z=0.10)
    z_ref, f = 0.05, 60.0
    for _ in range(4000):
        z, zd = imp.step(2.0e-4, f, z_ref)
    assert abs(zd) < 1e-3                                  # came to rest
    assert z == pytest.approx(z_ref + f / 4000.0, abs=1e-3)
    assert z > z_ref                                       # yielded short of the target


def _press_backend(n_grid=40, eta=150.0, tau_y=2500.0, density=1200.0):
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    dough_size = (0.12, 0.10, 0.07)
    pos, vol, floor = block(grid, size=dough_size, ppc=2)
    s = Solver(grid=grid).load_particles(pos, vol)
    s.set_material(newtonian(eta=eta, density=density).with_yield(tau_y))
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    return WarpMPMBackend(solver=s), grid, floor, dough_size


def test_backend_press_gives_compressive_reaction():
    # smoke: pressing the tool into the dough yields a positive (upward) reaction
    backend, grid, floor, ds = _press_backend(n_grid=36)
    cx = cy = grid.grid_lim * 0.5
    box_half = (0.5 * ds[0] + 0.01, 0.5 * ds[1] + 0.01, 0.6 * grid.dx)
    z = floor + ds[2] + box_half[2]
    tool = backend.attach_tool((cx, cy, z), box_half)
    dt, sub = 2.0e-5, 20
    f_seen = 0.0
    for _ in range(20):
        z -= 0.3 * dt * sub
        backend.set_tool_kinematics(tool, center=(cx, cy, z + 0.3 * dt * sub),
                                    velocity=(0, 0, -0.3))
        backend.step(dt, sub)
        f_seen = max(f_seen, backend.get_tool_wrench(tool, at_center=(cx, cy, z))["Fz"])
    assert f_seen > 0.0


@pytest.mark.slow
def test_two_way_press_halts_above_floor():
    # the load-bearing two-way property: a force-regulated descent into firm dough reaches
    # its target force and halts, with the box bottom comfortably above the floor; the
    # dough stops it, not the safety clamp.
    backend, grid, floor, ds = _press_backend(n_grid=40)
    cx = cy = grid.grid_lim * 0.5
    box_half = (0.5 * ds[0] + 0.01, 0.5 * ds[1] + 0.01, 0.6 * grid.dx)
    dough_top = floor + ds[2]
    dt, sub = 2.0e-5, 24
    dt_ctrl = dt * sub
    ctrl = ForceAdmittance(f_target=45.0, v_max=0.45)
    z = dough_top + box_half[2] + 0.006
    z_floor = floor + box_half[2] + 0.003
    tool = backend.attach_tool((cx, cy, z), box_half)
    f_filt = 0.0
    for t in range(130):
        v_down = 0.0 if t == 0 else ctrl.velocity_down(f_filt)
        z_new = max(z - v_down * dt_ctrl, z_floor)
        vz = (z_new - z) / dt_ctrl
        if t > 0:
            backend.set_tool_kinematics(tool, center=(cx, cy, z), velocity=(0, 0, vz))
            backend.step(dt, sub)
        z = z_new
        f = float(backend.get_tool_wrench(tool, at_center=(cx, cy, z))["Fz"])
        f_filt = 0.4 * f + 0.6 * f_filt
    box_bottom = z - box_half[2]
    assert box_bottom > floor + 1e-3, "tool reached the floor (no two-way stopping)"
    assert box_bottom > z_floor + 1e-4, "tool sat on the safety clamp, not stopped by dough"
    assert f_filt > 0.3 * ctrl.f_target, "never built a real reaction force (no contact)"
    assert abs(vz) < 0.05, "tool had not decelerated/halted"
