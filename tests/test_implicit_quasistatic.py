"""Gate 1 for the quasi-static implicit solver (docs/implicit_plan.md): an elastic
column in equilibrium owes sigma_zz = -rho g (h - z) regardless of constitutive law.
Same configuration the numpy prototype (experiments/qs_prototype.py) passed with."""
from __future__ import annotations

import itertools

import numpy as np

from warpmpm.core.solver import GridConfig
from warpmpm.implicit import QuasiStaticSolver


def _column(dx=0.03, col=(0.12, 0.12, 0.36), floor_k=3, ppd=2):
    org = np.array([4, 4, floor_k]) * dx
    hp = dx / ppd
    ax = [np.arange(hp / 2, c, hp) for c in col]
    pos = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3) + org
    vol = np.full(len(pos), hp ** 3)
    return pos, vol, org


def test_displacement_press_matches_uniaxial_stiffness():
    """A frictionless plate pressing the column top by delta must react with
    F = E A delta / h (uniaxial stress, free sides), growing linearly per step."""
    dx = 0.03
    floor_k = 3
    pos, vol, org = _column(dx=dx, floor_k=floor_k)
    E, nu = 1.0e5, 0.3
    h, A = 0.36, 0.12 * 0.12
    qs = QuasiStaticSolver(pos, vol, rho=1000.0, E=E, nu=nu,
                           grid=GridConfig(n_grid=20, grid_lim=20 * dx), device="cpu")
    qs.fix_floor(floor_k, components="z")
    top_z = org[2] + h
    kz = np.arange(qs.n_nodes) % qs.ng[2]
    plate = qs._active & (kz * dx >= top_z - 0.6 * dx)
    n_steps, dz = 4, -0.0009                      # 4 x 0.9 mm = 1 percent strain
    qs.prescribe_nodes(plate, (0.0, 0.0, dz), components="z")
    info = qs.solve_gravity(g=0.0, n_steps=n_steps)

    F = np.array(info["tool_force"])              # (n_steps, 3)
    F_exact = E * A * abs(dz) * n_steps / h
    assert F[-1][2] > 0                           # compressed column pushes back up
    assert abs(F[-1][2] - F_exact) / F_exact < 0.10, f"F={F[-1][2]:.2f} vs {F_exact:.2f}"
    # linear growth with the load step (elastic, small strain)
    ratios = F[:, 2] / (np.arange(1, n_steps + 1) * F[0][2])
    assert np.allclose(ratios, 1.0, atol=0.05)


def test_vonmises_press_plateaus_at_yield():
    """Pressing past yield, the plate force must plateau at A sigma_y (uniaxial von
    Mises with free sides), well below the elastic extrapolation."""
    dx = 0.03
    floor_k = 3
    pos, vol, org = _column(dx=dx, floor_k=floor_k)
    E, nu, sy = 1.0e5, 0.3, 500.0
    h, A = 0.36, 0.12 * 0.12
    qs = QuasiStaticSolver(pos, vol, rho=1000.0, E=E, nu=nu, yield_stress=sy,
                           grid=GridConfig(n_grid=20, grid_lim=20 * dx), device="cpu")
    qs.fix_floor(floor_k, components="z")
    kz = np.arange(qs.n_nodes) % qs.ng[2]
    plate = qs._active & (kz * dx >= org[2] + h - 0.6 * dx)
    n_steps, dz = 8, -0.0009                     # 7.2 mm = 2 percent, yield at 0.5
    qs.prescribe_nodes(plate, (0.0, 0.0, dz), components="z")
    info = qs.solve_gravity(g=0.0, n_steps=n_steps, newton_max=40)

    F = np.array(info["tool_force"])[:, 2]
    F_yield = A * sy                              # 7.2 N
    F_elastic_extrap = E * A * abs(dz) * n_steps / h   # 28.8 N if it never yielded
    assert abs(F[-1] - F_yield) / F_yield < 0.15, f"plateau {F[-1]:.2f} vs {F_yield}"
    assert F[-1] < 0.5 * F_elastic_extrap
    # the last steps are flat (plastic flow), the first step is elastic and linear
    assert abs(F[-1] - F[-2]) / F_yield < 0.05


def test_equilibrium_column_matches_analytic_profile():
    dx = 0.03
    floor_k = 3
    pos, vol, org = _column(dx=dx, floor_k=floor_k)
    rho, g, E, nu = 1000.0, 9.81, 1.0e5, 0.3
    qs = QuasiStaticSolver(pos, vol, rho=rho, E=E, nu=nu,
                           grid=GridConfig(n_grid=20, grid_lim=20 * dx), device="cpu")
    qs.fix_floor(floor_k, components="z")
    info = qs.solve_gravity(g=g, n_steps=5)

    # Newton converges fast in the small-strain regime
    assert max(info["newton_iters"]) <= 5
    assert info["residual_norms"][-1] < 1e-6 * len(pos)

    sig_zz = qs.cauchy_stress()[:, 2, 2]
    z_rel = qs.x()[:, 2] - org[2]
    h = 0.36
    bins = np.linspace(0, h, 13)
    mid = 0.5 * (bins[:-1] + bins[1:])
    prof = np.array([sig_zz[(z_rel >= a) & (z_rel < b)].mean()
                     for a, b in itertools.pairwise(bins)])
    exact = -rho * g * (h - mid)
    err = np.abs(prof - exact) / (rho * g * h)
    # interior bins match the analytic profile; the basal boundary layer (first two
    # bins, within ~2 cells of the floor) is excluded, as in the explicit engine
    assert err[2:].max() < 0.02, f"interior profile error {err[2:].max():.4f}"

    # settlement against the free-sided analytic rho g h^2 / (2 E)
    settle = (qs.x()[:, 2] - pos[:, 2]).min()
    analytic = -rho * g * h * h / (2 * E)
    assert abs(settle - analytic) / abs(analytic) < 0.15
