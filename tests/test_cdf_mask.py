"""The block-mask early-out for the CPIC tag vote: particles whose stencil touches
no tagged block skip the 27-node vote entirely. The skip is exact (an all-empty
stencil yields pc == 0 through the full path too), and these tests prove it
bitwise against the exhaustive path (model.cdf_mask_off = 1) in both pipelines,
with a second blob far from the sheet so the skip path really runs."""
from __future__ import annotations

import numpy as np

from warpmpm.core.solver import GridConfig, Solver
from warpmpm.geometry import build_surface_cdf
from warpmpm.materials import newtonian

G = GridConfig(n_grid=32, grid_lim=0.4)
DX = G.dx
DT, SUB = 2e-4, 10


def _sheet(normal: str, size=0.5, res=64):
    s = size / 2
    if normal == "x":
        v = np.array([[0, -s, -s], [0, s, -s], [0, s, s], [0, -s, s]], dtype=float)
    else:
        v = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return build_surface_cdf(v, f, res=res, band_cells=3.0)


def _water(x0, x1, y0, y1, z0, z1):
    hp = DX / 2
    ax = [np.arange(a + hp / 2, b, hp) for a, b in ((x0, x1), (y0, y1), (z0, z1))]
    return np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)


def _scene(fused: bool, mask_off: bool):
    """A column resting on a CDF shelf plus a second blob far from any sheet, both
    under gravity: near-sheet, in-band, and far-field particles in one run."""
    pos = np.concatenate([
        _water(0.14, 0.26, 0.14, 0.26, 0.21, 0.27),   # on the shelf
        _water(0.06, 0.12, 0.06, 0.12, 0.08, 0.14),   # far corner, no sheet nearby
    ])
    s = Solver(grid=G, device="cpu", fused=fused)
    s.load_particles(pos, np.full(len(pos), (DX / 2) ** 3, np.float32))
    s.set_material(newtonian(eta=0.5, density=1000.0, bulk_modulus=5.0e4),
                   g=[0.0, 0.0, -9.81])
    s.add_plane((0, 0, 3 * DX), (0, 0, 1), "slip")
    s.add_domain_walls()
    h = s.add_cdf_collider(_sheet("z"), center=(0.2, 0.2, 0.20), friction=0.3)
    s._sim.mpm_model.cdf_mask_off = 1 if mask_off else 0
    return s, h


def test_mask_skip_bitwise_static_both_pipelines():
    for fused in (False, True):
        out = {}
        for mask_off in (False, True):
            s, _ = _scene(fused, mask_off)
            for _ in range(5):
                s.step(DT, SUB)
            out[mask_off] = (s.x(), s.v())
        assert np.array_equal(out[False][0], out[True][0]), f"x diverged (fused={fused})"
        assert np.array_equal(out[False][1], out[True][1]), f"v diverged (fused={fused})"


def test_mask_skip_bitwise_moving_wall_both_pipelines():
    for fused in (False, True):
        out = {}
        for mask_off in (False, True):
            s, h = _scene(fused, mask_off)
            s.set_cdf_pose(h, velocity=(0.0, 0.0, 0.01))
            for _ in range(8):
                s.step(DT, SUB)
            out[mask_off] = (s.x(), s.v())
        assert np.array_equal(out[False][0], out[True][0]), f"x diverged (fused={fused})"
        assert np.array_equal(out[False][1], out[True][1]), f"v diverged (fused={fused})"
