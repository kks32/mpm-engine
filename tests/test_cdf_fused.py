"""CPIC on the fused pipeline (step 5): fused == split bitwise with a CDF collider,
static and MOVING (the moving case is what the prev/cur tag double-buffering
exists for: the fused gather half must read the previous substep's colors while
its scatter half reads this substep's)."""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm.core.solver import GridConfig, Solver
from warpmpm.geometry import build_surface_cdf
from warpmpm.materials import newtonian

G = GridConfig(n_grid=32, grid_lim=0.4)
DX = G.dx


def _sheet():
    s = 0.25
    v = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return build_surface_cdf(v, f, res=64, band_cells=3.0)


def _scene(fused, moving, device="cpu"):
    hp = DX / 2
    ax = [np.arange(a + hp / 2, b, hp)
          for a, b in ((0.14, 0.26), (0.14, 0.26), (0.21, 0.28))]
    pos = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    s = Solver(grid=G, device=device, fused=fused)
    s.load_particles(pos, np.full(len(pos), hp ** 3, np.float32))
    s.set_material(newtonian(eta=0.5, density=1000.0, bulk_modulus=5.0e4),
                   g=[0.0, 0.0, -9.81])
    s.add_plane((0, 0, 3 * DX), (0, 0, 1), "slip")
    s.add_domain_walls()
    vel = (0.0, 0.0, 0.01) if moving else (0.0, 0.0, 0.0)
    s.add_cdf_collider(_sheet(), center=(0.2, 0.2, 0.20), velocity=vel)
    return s


@pytest.mark.parametrize("moving", [False, True])
def test_cdf_fused_matches_split_bitwise(moving):
    a = _scene(fused=False, moving=moving)
    b = _scene(fused=True, moving=moving)
    for _ in range(8):
        a.step(2e-4, 10)
        b.step(2e-4, 10)
    assert np.array_equal(a.x(), b.x())
    assert np.array_equal(a.v(), b.v())
    assert np.array_equal(a.F(), b.F())
    assert np.array_equal(a.stress(), b.stress())


def test_cdf_fused_matches_split_with_cov():
    hp = DX / 2
    ax = [np.arange(a + hp / 2, b, hp)
          for a, b in ((0.16, 0.24), (0.16, 0.24), (0.21, 0.26))]
    pos = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    n = len(pos)
    cov = np.tile(np.array([1e-6, 0, 0, 1e-6, 0, 1e-6], np.float32), (n, 1))
    out = {}
    for fused in (False, True):
        s = Solver(grid=G, device="cpu", fused=fused)
        s.load_particles(pos, np.full(n, hp ** 3, np.float32), cov=cov,
                         cov_mode="step")
        s.set_material(newtonian(eta=0.5, density=1000.0, bulk_modulus=5.0e4),
                       g=[0.0, 0.0, -9.81])
        s.add_domain_walls()
        s.add_cdf_collider(_sheet(), center=(0.2, 0.2, 0.20))
        for _ in range(6):
            s.step(2e-4, 10)
        out[fused] = (s.x(), s.cov())
    assert np.array_equal(out[False][0], out[True][0])
    assert np.array_equal(out[False][1], out[True][1])


def test_cdf_fused_graph_replay_bitwise_on_cuda():
    """CUDA graphs replay the fused interior segment while the CDF stamp stays
    live: graphs on vs off must be bitwise-identical with a moving CDF collider
    (the graph only READS the tag grids, whose contents the live stamp updates in
    place). Needs a GPU; this is the Vista gate for CPIC under graphs."""
    import warp as wp
    if wp.get_cuda_device_count() == 0:
        pytest.skip("needs CUDA")
    import os
    out = {}
    for graphs in (True, False):
        os.environ.pop("WARPMPM_NO_CUDA_GRAPH", None)
        if not graphs:
            os.environ["WARPMPM_NO_CUDA_GRAPH"] = "1"
        s = _scene(fused=True, moving=True, device="cuda:0")
        for _ in range(6):
            s.step(2e-4, 10)
        out[graphs] = (s.x(), s.v(), s.F())
    os.environ.pop("WARPMPM_NO_CUDA_GRAPH", None)
    for a, b in zip(out[True], out[False], strict=True):
        np.testing.assert_array_equal(a, b)


def test_cdf_stamp_skip_under_graphs_on_cuda():
    """The stamp skip's CUDA paths: a static collider runs long stretches of graph
    replays with NO live stamp launches between them, and a stop-start collider
    crosses skip -> stamp -> skip transitions (the recorded stale box reconciles
    the prev tags at the first read after each stretch). Graphs on vs off must be
    bitwise-identical through both. Needs a GPU; a Vista gate."""
    import warp as wp
    if wp.get_cuda_device_count() == 0:
        pytest.skip("needs CUDA")
    import os
    out = {}
    for graphs in (True, False):
        os.environ.pop("WARPMPM_NO_CUDA_GRAPH", None)
        if not graphs:
            os.environ["WARPMPM_NO_CUDA_GRAPH"] = "1"
        s = _scene(fused=True, moving=False, device="cuda:0")
        for _ in range(4):
            s.step(2e-4, 10)
        s.set_cdf_pose(0, velocity=(0.0, 0.0, 0.01))
        for _ in range(3):
            s.step(2e-4, 10)
        s.set_cdf_pose(0, velocity=(0.0, 0.0, 0.0))
        for _ in range(3):
            s.step(2e-4, 10)
        out[graphs] = (s.x(), s.v(), s.F())
    os.environ.pop("WARPMPM_NO_CUDA_GRAPH", None)
    for a, b in zip(out[True], out[False], strict=True):
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
