"""The static-pose stamp skip: a CDF lane restamps only when its pose changes, so
resting colliders cost no per-substep host launches. Each test pins the skip
against the always-stamp path (_cdf_force_stamp) bitwise, in both pipelines,
and the counter proves the skip actually engages."""
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


def _solver(pos, fused, g=(0.0, 0.0, 0.0)):
    s = Solver(grid=G, device="cpu", fused=fused)
    s.load_particles(pos, np.full(len(pos), (DX / 2) ** 3, np.float32))
    s.set_material(newtonian(eta=0.5, density=1000.0, bulk_modulus=5.0e4), g=list(g))
    s.add_plane((0, 0, 3 * DX), (0, 0, 1), "slip")
    s.add_domain_walls()
    return s


def _run_static(fused: bool, force: bool, ticks: int = 5):
    """Column resting on a static CDF shelf under gravity."""
    pos = _water(0.14, 0.26, 0.14, 0.26, 0.21, 0.27)
    s = _solver(pos, fused, g=(0.0, 0.0, -9.81))
    s.add_cdf_collider(_sheet("z"), center=(0.2, 0.2, 0.20), friction=0.3)
    s._sim._cdf_force_stamp = force
    for _ in range(ticks):
        s.step(DT, SUB)
    return s.x(), s.v(), s._sim._cdf_stamp_count


def _run_stop_start(fused: bool, force: bool):
    """Wall sweeps a blob, holds two ticks, then sweeps again: exercises the
    dirty transitions on stop (one restamp then silence) and restart."""
    pos = _water(0.22, 0.30, 0.14, 0.26, 3 * DX, 3 * DX + 0.08)
    s = _solver(pos, fused)
    h = s.add_cdf_collider(_sheet("x"), center=(0.18, 0.2, 0.2),
                           velocity=(1.0, 0.0, 0.0))
    s._sim._cdf_force_stamp = force
    for _ in range(10):
        s.step(DT, SUB)
    s.set_cdf_pose(h, velocity=(0.0, 0.0, 0.0))
    for _ in range(2):
        s.step(DT, SUB)
    s.set_cdf_pose(h, velocity=(0.5, 0.0, 0.0))
    for _ in range(8):
        s.step(DT, SUB)
    return s.x(), s.v(), s._sim._cdf_stamp_count


def test_static_shelf_skip_bitwise_both_pipelines():
    for fused in (False, True):
        x_skip, v_skip, n_skip = _run_static(fused, force=False)
        x_all, v_all, n_all = _run_static(fused, force=True)
        assert np.array_equal(x_skip, x_all), f"positions diverged (fused={fused})"
        assert np.array_equal(v_skip, v_all), f"velocities diverged (fused={fused})"
        # a static pose stamps exactly once; the forced path stamps every substep
        assert n_skip == 1, f"expected one stamp, got {n_skip} (fused={fused})"
        assert n_all == 5 * SUB


def test_stop_start_wall_skip_bitwise_both_pipelines():
    for fused in (False, True):
        x_skip, v_skip, n_skip = _run_stop_start(fused, force=False)
        x_all, v_all, n_all = _run_stop_start(fused, force=True)
        assert np.array_equal(x_skip, x_all), f"positions diverged (fused={fused})"
        assert np.array_equal(v_skip, v_all), f"velocities diverged (fused={fused})"
        # moving substeps stamp, the hold goes quiet after one settling restamp
        assert n_skip < n_all, (n_skip, n_all)
        assert n_all == 20 * SUB
