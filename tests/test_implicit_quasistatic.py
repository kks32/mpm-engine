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
