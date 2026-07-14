"""Periodic-x transfers (task: steady chute flows). Gates: a block advecting
across the wrap stays coherent, and a tilted-gravity mu(I) chute develops the
Bagnold velocity profile (v(z) ~ h^1.5 - (h-z)^1.5, the exact steady solution
for mu(I) rheology on an incline where tau/p = tan(theta) uniformly)."""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm.core.solver import GridConfig, Solver
from warpmpm.materials import elastic, granular

G = GridConfig(n_grid=32, grid_lim=0.4)
DX = G.dx


def _fill(x0, x1, y0, y1, z0, z1, h):
    ax = [np.arange(a + h / 2, b, h) for a, b in ((x0, x1), (y0, y1), (z0, z1))]
    return np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)


@pytest.mark.parametrize("fused", [False, True])
def test_block_advects_across_wrap_coherently(fused):
    """A stress-free elastic block translating at 1 m/s crosses the periodic
    boundary and comes out rigid: extents preserved, velocity still uniform.
    This is also the regression gate for the base-index floor fix (toward-zero
    int truncation gave a just-wrapped particle an invalid stencil with
    negative B-spline weights, which blew up within a substep)."""
    h = DX / 2
    pos = _fill(0.30, 0.38, 0.18, 0.22, 0.18, 0.22, h)
    s = Solver(grid=G, device="cpu", periodic_x=True, fused=fused)
    s.load_particles(pos, np.full(len(pos), h ** 3, np.float32))
    s.set_material(elastic(E=1.0e5, nu=0.3, density=1000.0), g=[0.0, 0.0, 0.0])
    v = s.v()
    v[:, 0] = 1.0
    s.set_v(v)
    for _ in range(200):                     # 0.4 m = one full domain length
        s.step(2e-4, 10)
    x, vf = s.x(), s.v()
    assert np.isfinite(x).all()
    # unwrap x for the extent measure if the block straddles the boundary
    xw = x[:, 0].copy()
    xs = np.sort(xw)
    if xs[-1] - xs[0] > 0.2:
        gap = np.argmax(np.diff(xs))
        xw[xw < 0.5 * (xs[gap] + xs[gap + 1])] += G.grid_lim
    span0 = np.array([0.08, 0.04, 0.04]) - h   # particle-center spans at load
    span = np.array([np.ptp(xw), np.ptp(x[:, 1]), np.ptp(x[:, 2])])
    assert np.all(np.abs(span - span0) < 2 * h), f"block deformed: {span} vs {span0}"
    assert abs(float(vf[:, 0].mean()) - 1.0) < 1e-3
    assert float(vf[:, 0].std()) < 1e-3


@pytest.mark.slow
def test_chute_develops_bagnold_profile():
    """Tilted gravity (tan theta = 0.5 inside the steady-flow window of
    mu_s = 0.38, mu_s + delta_mu = 0.64) over a sticky base: the mu(I) layer
    must develop the Bagnold profile. Measured corr = 0.9994 at t = 1.44 s
    while still accelerating toward the analytic steady surface speed; the
    profile SHAPE is the early, cheap gate, and full steady state (the curve-
    recovery sweep) is the follow-on stage."""
    MU_S, DMU, I0 = 0.38, 0.26, 0.3
    theta = np.arctan(0.5)
    z0 = 3 * DX
    h_layer = 0.125
    hp = DX / 2
    pos = _fill(0.0, G.grid_lim, 0.14, 0.26, z0, z0 + h_layer, hp)
    s = Solver(grid=G, device="cpu", periodic_x=True)
    s.load_particles(pos, np.full(len(pos), hp ** 3, np.float32))
    s.set_material(granular(mu_s=MU_S, delta_mu=DMU, I0=I0, density=1590.0,
                            grain_diameter=5.0e-3, grain_density=2650.0,
                            E=1.0e6, nu=0.3),
                   g=[9.81 * np.sin(theta), 0.0, -9.81 * np.cos(theta)])
    s.add_plane((0, 0, z0), (0, 0, 1), "sticky")
    s.add_plane((0, 0.14, 0), (0, 1, 0), "slip")
    s.add_plane((0, 0.26, 0), (0, -1, 0), "slip")
    v95_prev = 0.0
    accel = []
    for tick in range(900):                  # 1.08 s
        s.step(1.2e-4, 10)
        if (tick + 1) % 300 == 0:
            v95 = float(np.quantile(s.v()[:, 0], 0.95))
            accel.append(v95 - v95_prev)
            v95_prev = v95
    # flowing and monotonically approaching steady state (acceleration decays)
    assert v95_prev > 0.5, f"chute not flowing (v95 = {v95_prev:.3f})"
    assert accel[-1] < accel[0], f"not approaching steady state: {accel}"
    x, v = s.x(), s.v()
    zrel = x[:, 2] - z0
    nb = 8
    zb = np.linspace(0.5 * DX, h_layer - 0.5 * DX, nb + 1)
    zc = 0.5 * (zb[1:] + zb[:-1])
    prof = np.array([v[(zrel >= zb[i]) & (zrel < zb[i + 1]), 0].mean()
                     for i in range(nb)])
    h_eff = float(np.quantile(zrel, 0.99))
    bag = h_eff ** 1.5 - (h_eff - np.minimum(zc, h_eff)) ** 1.5
    corr = float(np.corrcoef(prof, bag)[0, 1])
    assert corr > 0.99, f"velocity profile is not Bagnold-shaped (corr {corr:.4f})"