"""CUDA-graph substep replay must match live launches (GPU only; skipped without CUDA).

Run on the GPU box:  pytest tests/test_cuda_graph.py -v
The graph path replays zero/stress/p2g/normalize and g2p as two captured graphs with the
BC launches live between them. GPU atomics reorder float sums, so positions are compared
allclose rather than bitwise. Also prints a substep-time comparison.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import warp as wp

from warpmpm import GridConfig, Solver
from warpmpm.materials import newtonian

from tests.test_sdf_collider import _cup_sdf, _fill_cavity

cuda = pytest.mark.skipif(wp.get_cuda_device_count() == 0, reason="needs a CUDA device")


def _scene(use_graph: bool):
    grid = GridConfig(n_grid=64, grid_lim=0.4)
    sdf = _cup_sdf()
    center = np.array([0.2, 0.2, 0.06])
    pts, vol = _fill_cavity(center, grid.dx)
    s = Solver(grid=grid, device="cuda:0", fused=False).load_particles(pts, vol)
    s.set_material(newtonian(eta=5.0, density=1000.0, bulk_modulus=5.0e5))
    s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "slip", friction=0.2)
    h = s.add_sdf_collider(sdf, center=center, surface="separable", friction=0.3)
    s._sim.use_cuda_graph = use_graph
    return s, h


@cuda
def test_graph_matches_live():
    sa, ha = _scene(use_graph=True)
    sb, hb = _scene(use_graph=False)
    dt = 2.0e-4
    for _ in range(20):
        sa.step(dt, substeps=4)
        sb.step(dt, substeps=4)
    assert sa._sim._graph_A is not None, "graph capture never engaged"
    np.testing.assert_allclose(sa.x(), sb.x(), rtol=1e-5, atol=1e-6)
    wa, wb = sa.sdf_wrench(ha, dt * 4), sb.sdf_wrench(hb, dt * 4)
    np.testing.assert_allclose(wa["force"], wb["force"], rtol=1e-3, atol=1e-5)


@cuda
def test_graph_bench():
    times = {}
    for name, flag in (("live", False), ("graph", True)):
        s, _ = _scene(use_graph=flag)
        s.step(2.0e-4, substeps=4)          # warm-up (JIT + capture)
        wp.synchronize_device("cuda:0")
        t0 = time.perf_counter()
        s.step(2.0e-4, substeps=200)
        wp.synchronize_device("cuda:0")
        times[name] = (time.perf_counter() - t0) / 200 * 1e3
    print(f"\nms/substep live={times['live']:.3f} graph={times['graph']:.3f} "
          f"speedup={times['live'] / times['graph']:.2f}x")
    assert times["graph"] <= times["live"] * 1.1   # graphs must not be slower
