"""Block sort (claymore 5a): reordering particles must not change the physics. The
permutation itself is exact (multiset of per-particle state identical, checked by
lexsorting rows); subsequent dynamics can differ only through P2G atomic accumulation
ORDER, so trajectories match to float tolerance, and conserved sums match tightly.
Array pointers must survive the sort (in-place copy-back, the captured-graph contract)."""
from __future__ import annotations

import numpy as np
import warp as wp

from warpmpm import GridConfig, Solver
from warpmpm.materials import newtonian


def _scene(sort_interval=0, n=3000, seed=1):
    grid = GridConfig(n_grid=48, grid_lim=0.4)
    rng = np.random.default_rng(seed)
    # two separated blobs so block keys are far from sorted at load
    a = rng.random((n // 2, 3), dtype=np.float32) * 0.08 + 0.12
    b = (rng.random((n - n // 2, 3), dtype=np.float32) * 0.08
         + np.array([0.24, 0.24, 0.12], np.float32))
    pts = np.concatenate([a, b]).astype(np.float32)
    pts = pts[rng.permutation(len(pts))]              # scramble the initial order
    vol = np.full(len(pts), 1.0e-7, np.float32)
    s = Solver(grid=grid, device="cpu", sort_interval=sort_interval).load_particles(pts, vol)
    s.set_material(newtonian(eta=2.0, density=1000.0, bulk_modulus=2.0e5))
    s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "separate", friction=0.3)
    s.add_domain_walls()
    return s


def _rows(s):
    st = s._sim.mpm_state
    return np.concatenate([s.x(), s.v(),
                           st.particle_F.numpy().reshape(len(s.x()), -1)], axis=1)


def test_sort_permutation_is_exact_multiset():
    s = _scene()
    before = _rows(s)
    assert s._sort_particles() is True                 # scrambled order -> sorts
    after = _rows(s)
    order_b = np.lexsort(before.T)
    order_a = np.lexsort(after.T)
    assert np.array_equal(before[order_b], after[order_a]), "sort changed particle state"
    assert s._sort_particles() is False                # second call: already ordered


def test_sort_keeps_particle_material_parameters_attached():
    s = _scene(n=600)
    model = s._sim.mpm_model
    n = s.n_particles
    values = {
        "mu": np.linspace(10.0, 20.0, n, dtype=np.float32),
        "lam": np.linspace(30.0, 50.0, n, dtype=np.float32),
        "yield_stress": np.linspace(60.0, 90.0, n, dtype=np.float32),
    }
    for name, value in values.items():
        wp.copy(getattr(model, name), wp.array(value, dtype=float, device="cpu"))

    before = np.column_stack([
        s.x(), model.mu.numpy(), model.lam.numpy(), model.yield_stress.numpy()
    ])
    ptrs = (model.mu.ptr, model.lam.ptr, model.yield_stress.ptr)
    assert s._sort_particles() is True
    after = np.column_stack([
        s.x(), model.mu.numpy(), model.lam.numpy(), model.yield_stress.numpy()
    ])
    np.testing.assert_array_equal(before[np.lexsort(before[:, :3].T)],
                                  after[np.lexsort(after[:, :3].T)])
    assert (model.mu.ptr, model.lam.ptr, model.yield_stress.ptr) == ptrs


def test_sort_keeps_pointers_and_dynamics_equivalent():
    ref = _scene(sort_interval=0)
    srt = _scene(sort_interval=2)
    ptrs = (srt._sim.mpm_state.particle_x.ptr, srt._sim.mpm_state.particle_F.ptr)
    for _ in range(8):
        ref.step(2.0e-4, 6)
        srt.step(2.0e-4, 6)
    assert (srt._sim.mpm_state.particle_x.ptr,
            srt._sim.mpm_state.particle_F.ptr) == ptrs, "sort replaced an array"
    xr, xs = ref.x(), srt.x()
    # conserved sums are permutation-invariant and only atomic order differs
    np.testing.assert_allclose(xs.sum(0), xr.sum(0), rtol=1e-5)
    vr, vs = ref.v(), srt.v()
    np.testing.assert_allclose(vs.sum(0), vr.sum(0), rtol=1e-4, atol=1e-6)
    # trajectories agree particle-for-particle after matching by lexsorted position
    np.testing.assert_allclose(xs[np.lexsort(xs.T)], xr[np.lexsort(xr.T)],
                               rtol=1e-4, atol=1e-6)
    assert np.isfinite(xs).all()


def test_sort_composes_with_fused_pipeline():
    a = _scene(sort_interval=0)
    b = _scene(sort_interval=2)
    b.fused = True
    for _ in range(6):
        a.step(2.0e-4, 6)
        b.step(2.0e-4, 6)
    xa, xb = a.x(), b.x()
    np.testing.assert_allclose(xb[np.lexsort(xb.T)], xa[np.lexsort(xa.T)],
                               rtol=1e-4, atol=1e-6)
