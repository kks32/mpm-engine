"""Restricted BC launches must be physics-identical to full-grid launches.

The collider kernels write each grid node independently, so restricting a launch to the
collider's bounding box cannot change any node the full launch would have written, only
skip nodes it provably cannot touch. Positions must therefore match bitwise between the
two modes; the force accumulators may differ by float reordering only (atomic adds
enumerate the same node set in a different order).
"""
from __future__ import annotations

import numpy as np

from warpmpm import GridConfig, Solver
from warpmpm.materials import newtonian

from tests.test_sdf_collider import _cup_sdf, _fill_cavity


def _scene(restrict: bool):
    grid = GridConfig(n_grid=48, grid_lim=0.4)
    sdf = _cup_sdf()
    center = np.array([0.2, 0.2, 0.06])
    pts, vol = _fill_cavity(center, grid.dx)
    s = Solver(grid=grid, device="cpu").load_particles(pts, vol)
    s.set_material(newtonian(eta=5.0, density=1000.0, bulk_modulus=5.0e5))
    s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "slip", friction=0.2)   # axis-aligned slab
    sdf_h = s.add_sdf_collider(sdf, center=center, surface="separable", friction=0.3)
    box_h = s.add_box((0.30, 0.20, 0.10), (0.02, 0.02, 0.02), velocity=(-0.05, 0.0, -0.02))
    s._sim.restrict_bc = restrict
    s._sim.restrict_grid = restrict   # zero/normalize sweeps restricted too
    return s, sdf_h, box_h


def test_restricted_matches_full():
    sa, ha, _ = _scene(restrict=True)
    sb, hb, _ = _scene(restrict=False)
    dt = 2.0e-4
    for _ in range(25):
        sa.reset_sdf_force(ha)
        sb.reset_sdf_force(hb)
        sa.step(dt, substeps=4)
        sb.step(dt, substeps=4)
    xa, xb = sa.x(), sb.x()
    assert np.array_equal(xa, xb), (
        f"restricted vs full positions diverged: max |dx| = {np.abs(xa - xb).max()}")
    wa = sa.sdf_wrench(ha, dt * 4)
    wb = sb.sdf_wrench(hb, dt * 4)
    np.testing.assert_allclose(wa["force"], wb["force"], rtol=1e-4, atol=1e-6)


def test_collider_outside_domain_skips_launch():
    # an SDF collider parked far outside the grid must produce an empty box (launch
    # skipped) and leave the material untouched
    grid = GridConfig(n_grid=48, grid_lim=0.4)
    sdf = _cup_sdf()
    center = np.array([0.2, 0.2, 0.06])
    pts, vol = _fill_cavity(center, grid.dx)
    s = Solver(grid=grid, device="cpu").load_particles(pts, vol)
    s.set_material(newtonian(eta=5.0, density=1000.0, bulk_modulus=5.0e5))
    s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "slip", friction=0.2)
    h = s.add_sdf_collider(sdf, center=(5.0, 5.0, 5.0), surface="separable")   # far away
    box = s._sim.collider_aabbs[-1]()
    assert box is None, f"far-away collider should yield an empty launch box, got {box}"
    s.step(2.0e-4, substeps=2)   # must not crash and must not constrain anything
    assert np.isfinite(s.x()).all()


def test_particle_near_grid_edge_raises():
    # a particle within the P2G stencil reach of the grid edge means out-of-bounds atomic
    # writes (silent memory corruption); Solver.step must refuse instead
    import pytest

    grid = GridConfig(n_grid=32, grid_lim=0.4)
    pts = np.array([[0.2, 0.2, grid.grid_lim - 2.0 * grid.dx]], dtype=np.float32)
    vol = np.full(1, 1e-6, dtype=np.float32)
    s = Solver(grid=grid, device="cpu").load_particles(pts, vol)
    s.set_material(newtonian(eta=5.0, density=1000.0, bulk_modulus=5.0e5))
    with pytest.raises(RuntimeError, match="out of bounds"):
        s.step(2.0e-4, substeps=1)
