"""CDF collider registration and node stamping (step 3 of the CPIC plan). Transfers
do not read the tags yet; these tests cover the stamp bits against an analytic
plane, the guards, and that a registered collider leaves the simulation bitwise
untouched (the tags are write-only at this stage)."""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm.core.solver import GridConfig, Solver
from warpmpm.geometry import build_surface_cdf
from warpmpm.kernels.mpm_solver_warp import MAX_CDF
from warpmpm.materials import newtonian


def _sheet_cdf(size=0.3, res=64):
    """Horizontal square sheet at z = 0 in the body frame, normals +z."""
    v = np.array([[-size / 2, -size / 2, 0], [size / 2, -size / 2, 0],
                  [size / 2, size / 2, 0], [-size / 2, size / 2, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return build_surface_cdf(v, f, res=res, band_cells=6.0)


def _blob_solver(fused, with_cdf, cdf_center=(0.30, 0.30, 0.30)):
    g = GridConfig(n_grid=32, grid_lim=0.4)
    hp = g.dx / 2
    ax = np.arange(hp / 2, 0.06, hp)
    pos = (np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
           + np.array([0.08, 0.08, 0.08])).astype(np.float32)
    s = Solver(grid=g, device="cpu", fused=fused)
    s.load_particles(pos, np.full(len(pos), hp ** 3, np.float32))
    s.set_material(newtonian(eta=0.2, density=1000.0, bulk_modulus=5.0e4))
    s.add_plane((0, 0, 3 * g.dx), (0, 0, 1), "slip")
    s.add_domain_walls()
    if with_cdf:
        s.add_cdf_collider(_sheet_cdf(), center=cdf_center)
    return s


def test_stamp_bits_match_analytic_plane():
    g = GridConfig(n_grid=32, grid_lim=0.4)
    s = Solver(grid=g, device="cpu", fused=False)
    s.load_particles(np.full((8, 3), 0.2, np.float32) +
                     np.random.default_rng(0).uniform(0, 0.01, (8, 3)).astype(np.float32),
                     np.full(8, (g.dx / 2) ** 3, np.float32))
    s.set_material(newtonian(eta=0.2, density=1000.0, bulk_modulus=5.0e4))
    z0 = 0.2
    # band 2.5 dx so the +-2-node probes sit comfortably inside it (a probe exactly
    # AT the band edge is rejected or kept by trilerp noise, by construction)
    h = s.add_cdf_collider(_sheet_cdf(), center=(0.2, 0.2, z0), band=2.5 * g.dx)
    assert h == 0
    s._sim._stamp_cdf(1e-4, "cpu")
    tag = s._sim.mpm_state.grid_cdf_tag.numpy()
    k0 = z0 / g.dx                                # sheet plane in node units (16.0)
    ij = np.arange(int(0.15 / g.dx), int(0.25 / g.dx))   # nodes well inside the sheet
    above = tag[np.ix_(ij, ij, [int(k0 + 1), int(k0 + 2)])]
    below = tag[np.ix_(ij, ij, [int(k0 - 1), int(k0 - 2)])]
    assert np.all(above & 1), "valid bit above the sheet"
    assert np.all(above & 2), "side bit set above (positive normal side)"
    assert np.all(below & 1), "valid bit below the sheet"
    assert np.all(below & 2 == 0), "side bit clear below"
    far = tag[np.ix_(ij, ij, [int(k0 + 5), int(k0 - 5)])]
    assert np.all(far == 0), "beyond the band untagged"
    # rim: nodes within band of the sheet edge (x ~ 0.05 world) untagged
    edge_i = int(0.05 / g.dx)
    assert np.all(tag[edge_i, ij, int(k0 + 1)] == 0), "rim tube untagged"
    # distances: owner distance equals |z - z0| at tagged nodes
    d = s._sim.mpm_state.grid_cdf_d.numpy()
    zs = (int(k0 + 1)) * g.dx - z0
    assert np.allclose(d[np.ix_(ij, ij, [int(k0 + 1)])], abs(zs), atol=0.1 * g.dx)


def test_guards():
    s = _blob_solver(fused=False, with_cdf=False)
    cdf = _sheet_cdf()
    with pytest.raises(ValueError, match="runtime band"):
        s.add_cdf_collider(cdf, center=(0.2, 0.2, 0.2), band=10.0)
    with pytest.warns(RuntimeWarning, match="below 1.5 dx"):
        h = s.add_cdf_collider(cdf, center=(0.2, 0.2, 0.2), band=0.5 * s.grid.dx)
    with pytest.warns(RuntimeWarning, match="jumped the surface"):
        s.set_cdf_pose(h, center=(0.2, 0.2, 0.35))
    for _ in range(MAX_CDF - 1):
        s.add_cdf_collider(cdf, center=(0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="at most"):
        s.add_cdf_collider(cdf, center=(0.2, 0.2, 0.2))


@pytest.mark.parametrize("fused", [False, True])
def test_inert_collider_is_bitwise_invisible(fused):
    """A CDF collider whose band never touches the material must not change one bit
    of the trajectory: at this step the transfers ignore the tags entirely, and this
    pins that property per pipeline before the masking lands."""
    a = _blob_solver(fused, with_cdf=False)
    b = _blob_solver(fused, with_cdf=True)     # sheet far from the blob
    for _ in range(6):
        a.step(2e-4, 10)
        b.step(2e-4, 10)
    assert np.array_equal(a.x(), b.x())
    assert np.array_equal(a.v(), b.v())
    assert np.array_equal(a.F(), b.F())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
