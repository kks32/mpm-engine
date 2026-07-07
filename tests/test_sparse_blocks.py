"""Active-block sparse compute must be physics-identical to the dense/box paths.

The block set (marked particle blocks dilated by one) is a superset of every node the
transfers touch, and zeroing runs over the union with the previous tick, so no node ever
carries stale state. On CPU the runs are deterministic, so positions must match bitwise.
The two-blob scene is the case the particle-box path cannot exploit: two bodies at
opposite ends of the domain make the AABB span mostly empty space, while the active
blocks hug the material.
"""
from __future__ import annotations

import numpy as np

from warpmpm import GridConfig, Solver
from warpmpm.materials import newtonian

from tests.test_sdf_collider import _cup_sdf, _fill_cavity


def _two_blob_solver(sparse: bool, n_grid: int = 64):
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    h = grid.dx / 2
    pts = []
    for cx in (0.08, 0.32):                     # far-separated blobs
        for x in np.arange(-0.02, 0.02, h):
            for y in np.arange(-0.02, 0.02, h):
                for z in np.arange(0.0, 0.04, h):
                    pts.append((cx + x, 0.2 + y, 0.06 + z))
    pts = np.asarray(pts, dtype=np.float32)
    vol = np.full(len(pts), h**3, dtype=np.float32)
    s = Solver(grid=grid, device="cpu", sparse=sparse).load_particles(pts, vol)
    s.set_material(newtonian(eta=5.0, density=1000.0, bulk_modulus=5.0e5))
    s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "slip", friction=0.2)
    return s


def test_sparse_matches_dense_two_blobs():
    sa = _two_blob_solver(sparse=True)
    sb = _two_blob_solver(sparse=False)
    for _ in range(20):
        sa.step(2.0e-4, substeps=3)
        sb.step(2.0e-4, substeps=3)
    xa, xb = sa.x(), sb.x()
    assert np.array_equal(xa, xb), (
        f"sparse vs dense diverged: max |dx| = {np.abs(xa - xb).max()}")
    # and the active set must be genuinely sparse, tighter than the blob-spanning AABB
    frac = sa._sim.active_block_fraction()
    assert 0.0 < frac < 0.15, f"active-block fraction {frac:.3f} not sparse"


def test_sparse_matches_dense_with_sdf_collider():
    def scene(sparse):
        grid = GridConfig(n_grid=48, grid_lim=0.4)
        sdf = _cup_sdf()
        center = np.array([0.2, 0.2, 0.06])
        pts, vol = _fill_cavity(center, grid.dx)
        s = Solver(grid=grid, device="cpu", sparse=sparse).load_particles(pts, vol)
        s.set_material(newtonian(eta=5.0, density=1000.0, bulk_modulus=5.0e5))
        s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "slip", friction=0.2)
        h = s.add_sdf_collider(sdf, center=center, surface="separable", friction=0.3)
        return s, h

    sa, ha = scene(True)
    sb, hb = scene(False)
    dt = 2.0e-4
    for _ in range(15):
        sa.reset_sdf_force(ha)
        sb.reset_sdf_force(hb)
        sa.step(dt, substeps=4)
        sb.step(dt, substeps=4)
    assert np.array_equal(sa.x(), sb.x())
    wa, wb = sa.sdf_wrench(ha, dt * 4), sb.sdf_wrench(hb, dt * 4)
    np.testing.assert_allclose(wa["force"], wb["force"], rtol=1e-4, atol=1e-6)
