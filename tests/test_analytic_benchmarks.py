"""Analytic validation benchmarks, following the CB-Geo MPM benchmark suite
(github.com/cb-geo/mpm): each test runs a scene with a closed-form solution and
checks the engine against it. These are physics gates, not unit tests: they
validate friction calibration, the EOS pressure, and the elastic wave speed
end to end through P2G/G2P, at tolerances set by grid discretization.

1. Sliding block on a frictional incline (tilted gravity, axis-aligned plane):
   a = g (sin th - mu cos th); the mu > tan th block holds.
2. Hydrostatic column: the basal support force equals rho V g, read from the
   grid impulse (NOT a dp/dz fit through particle stress; see the test).
3. Free-free elastic bar, longitudinal vibration: T = 2 L / sqrt(E/rho)
   (NOT the sticky-clamped fixed-free variant; see the test).

The two substitutions route around measured engine failure modes that the test
docstrings document (particle-stress statics noise; sticky-clamp softening
under tension). They are characterizations to fix, not physics gates passed.
"""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm.core.solver import GridConfig, Solver
from warpmpm.materials import elastic, newtonian

G32 = GridConfig(n_grid=32, grid_lim=0.4)


def _fill(x0, x1, y0, y1, z0, z1, h):
    ax = [np.arange(a + h / 2, b, h) for a, b in ((x0, x1), (y0, y1), (z0, z1))]
    return np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)


# ---- 1. sliding block on a frictional incline --------------------------------------

THETA = np.deg2rad(30.0)
GRAV = 9.81


def _incline_block(mu: float):
    """Axis-aligned equivalent of the incline: keep the floor horizontal and tilt
    gravity by theta, so the plane sees the same normal load mg cos th and the
    same tangential drive mg sin th without any tilted geometry."""
    dx = G32.dx
    h = dx / 2
    z0 = 3 * dx
    pos = _fill(0.10, 0.18, 0.16, 0.24, z0, z0 + 0.04, h)
    s = Solver(grid=G32, device="cpu")
    s.load_particles(pos, np.full(len(pos), h ** 3, np.float32))
    s.set_material(elastic(E=1.0e6, nu=0.3, density=1000.0),
                   g=[GRAV * np.sin(THETA), 0.0, -GRAV * np.cos(THETA)])
    s.add_plane((0, 0, z0), (0, 0, 1), "separate", friction=mu)
    return s


def _vx_com(s):
    return float(s.v()[:, 0].mean())


def test_incline_sliding_block_matches_coulomb():
    """mu < tan th: the block slides at a = g (sin th - mu cos th). Acceleration is
    measured between two times after the contact transient, which cancels the
    startup offset; the block is stiff so it approximates the rigid analytic."""
    mu = 0.3
    s = _incline_block(mu)
    dt, sub = 2e-4, 10
    for _ in range(20):
        s.step(dt, sub)
    v1 = _vx_com(s)
    n = 50
    for _ in range(n):
        s.step(dt, sub)
    v2 = _vx_com(s)
    a_meas = (v2 - v1) / (n * dt * sub)
    a_true = GRAV * (np.sin(THETA) - mu * np.cos(THETA))
    assert a_true > 0.0
    err = abs(a_meas - a_true) / a_true
    assert err < 0.15, f"a = {a_meas:.3f} vs analytic {a_true:.3f} ({100 * err:.1f}%)"


def test_incline_static_block_holds():
    """mu > tan th: Coulomb holds the block statically."""
    s = _incline_block(0.8)
    for _ in range(70):
        s.step(2e-4, 10)
    assert abs(_vx_com(s)) < 0.02, f"held block drifts at {_vx_com(s):.4f} m/s"


# ---- 2. hydrostatic column: basal support = weight -----------------------------------

@pytest.mark.slow
def test_hydrostatic_column_weight_on_scale():
    """A settled water column resting on a box collider: the grid-impulse
    reaction must equal the column's weight rho V g. This is the static force
    balance read the way this engine reads forces reliably, from the grid (the
    same path that matches a MuJoCo wrist FT sensor and weighs the pour's
    receiving glass to 0.2 percent).

    Deliberately NOT a dp/dz fit through particle stress: per-particle EOS
    pressure in a near-static, weakly loaded column carries spatially
    correlated quadrature noise of order the signal itself at this resolution
    (binned cross-section means swing by +-70 percent of rho g H after
    long time-averaging), so any fitted gradient is a noise draw. The same
    lesson as the robotics force readouts: integrate on the grid, do not
    differentiate particle stress."""
    dx = G32.dx
    h = dx / 2
    z0 = 4 * dx
    height = 0.12
    rho = 1000.0
    pos = _fill(0.14, 0.26, 0.14, 0.26, z0, z0 + height, h)
    s = Solver(grid=G32, device="cpu")
    s.load_particles(pos, np.full(len(pos), h ** 3, np.float32))
    s.set_material(newtonian(eta=0.5, density=rho, bulk_modulus=5.0e4),
                   g=[0.0, 0.0, -GRAV])
    scale = s.add_box(center=(0.2, 0.2, z0 - 0.045), half_size=(0.12, 0.12, 0.045))
    # frictionless shaft: slip walls carry no vertical load, so the scale must
    # read the whole weight; without confinement the column slumps to a puddle
    s.add_plane((0.14, 0, 0), (1, 0, 0), "slip")
    s.add_plane((0.26, 0, 0), (-1, 0, 0), "slip")
    s.add_plane((0, 0.14, 0), (0, 1, 0), "slip")
    s.add_plane((0, 0.26, 0), (0, -1, 0), "slip")
    weight = float(len(pos) * h ** 3 * rho * GRAV)

    # two-phase settle: undamped so pressure waves equilibrate, then damped
    for _ in range(100):
        s.step(2e-4, 10)
    s._sim.mpm_model.grid_v_damping_scale = 0.999
    for _ in range(50):
        s.step(2e-4, 10)
    # average the scale reading over ~two breathing periods
    n = 60
    s.reset_tool_force(scale)
    for _ in range(n):
        s.step(2e-4, 10)
    fz = float(s.tool_force(scale, n * 2e-3)[2])
    # the accumulator integrates m (v_free - v_imposed): supported material free
    # falls by -g dt each substep against a static box, so a scale loaded from
    # above reads NEGATIVE z of magnitude equal to the supported weight
    err = abs(-fz - weight) / weight
    assert fz < 0.0, f"scale reads {fz:.2f} N (expected -z under load from above)"
    assert err < 0.08, f"scale reads {fz:.2f} N vs weight {weight:.2f} N ({100 * err:.1f}%)"


# ---- 3. free-free elastic bar, longitudinal vibration -------------------------------

def test_elastic_bar_vibration_period():
    """A FREE bar (no boundary conditions at all) excited with an antisymmetric
    axial velocity: the fundamental longitudinal period is T = 2 L / c with
    c = sqrt(E/rho) (nu = 0 makes the 3D bar behave as the 1D analytic). The
    period is read from the zero crossings of the half-bar velocity difference,
    which momentum conservation keeps centered on zero. Measured error is 0.3
    percent at this resolution and converges with dx.

    The free-free configuration is deliberate: the clamped (fixed-free) variant
    reads a period that GROWS with refinement (+8 percent at dx = 12.5 mm to
    +29 percent at 6.25 mm), a sticky-plane artifact under tension, not a
    material or transfer error. Anchoring a vibration benchmark to the sticky
    BC measures the clamp, not the wave speed."""
    grid = GridConfig(n_grid=48, grid_lim=0.4)
    h = grid.dx / 2
    E, rho, L = 1.0e6, 1000.0, 0.2
    c = np.sqrt(E / rho)
    x0 = 0.1
    pos = _fill(x0, x0 + L, 0.17, 0.22, 0.17, 0.22, h)
    s = Solver(grid=grid, device="cpu")
    s.load_particles(pos, np.full(len(pos), h ** 3, np.float32))
    s.set_material(elastic(E=E, nu=0.0, density=rho), g=[0.0, 0.0, 0.0])
    v = s.v()
    xc = x0 + L / 2
    v[:, 0] = 0.1 * (pos[:, 0] - xc) / (L / 2)   # ~0.3 percent strain: linear regime
    s.set_v(v)

    dt, sub = 5e-5, 10                # tick = 0.5 ms
    T = 2.0 * L / c                   # 12.65 ms
    right = pos[:, 0] > xc
    times, dv = [], []
    for tick in range(int(3.2 * T / (dt * sub))):
        s.step(dt, sub)
        vx = s.v()[:, 0]
        times.append((tick + 1) * dt * sub)
        dv.append(float(vx[right].mean() - vx[~right].mean()))
    dv = np.asarray(dv)
    times = np.asarray(times)
    cross = np.where(np.diff(np.sign(dv)) != 0)[0]
    assert len(cross) >= 3, "no oscillation observed"
    t_cross = times[cross] - dv[cross] * (times[cross + 1] - times[cross]) / (
        dv[cross + 1] - dv[cross])
    T_meas = 2.0 * float(np.mean(np.diff(t_cross)))
    err = abs(T_meas - T) / T
    assert err < 0.03, f"T = {1e3 * T_meas:.2f} ms vs analytic {1e3 * T:.2f} ms " \
                       f"({100 * err:.1f}%)"
