"""Robust Mode F: joint multi-aspect grid-consistent solve with the family-coefficient prior.

A single collapse leaves the K-coefficient mu(I) under-determined; pooling several aspect
ratios widens the excited inertial-number band, and the offline family prior (theta_bar,
Sigma_theta in the frozen basis, from features/function_encoder_training/prior.py) shrinks
the fit onto the plausible-law manifold. The prior enters the existing constrained QP as
AUGMENTED ROWS: minimizing ||A theta - b||^2 + (theta-theta_bar)^T M (theta-theta_bar) with
M = rho Sigma^-1 equals ||[A; L] theta - [b; L theta_bar]||^2 for L = chol(M), so no solver
change is needed. Grid-consistent (Bubnov-Galerkin) assembly with the true MPM pressure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ident.features.function_encoder import FunctionEncoderDict
from ident.io.schema import load_dump
from ident.solve.qp import constrained_solve
from ident.weakform.grid_assembly import assemble_grid_consistent


def _load_fe_with_prior(fe_path="mpm_engine/fe-weights/granular_mu_i.npz"):
    d = np.load(fe_path)
    fe = FunctionEncoderDict(d["s_grid"], d["table"])
    if "theta_mean" not in d:
        raise RuntimeError("run features/function_encoder_training/prior.py first")
    return fe, np.asarray(d["theta_mean"], float), np.asarray(d["theta_cov"], float)


def joint_prior_solve(dumps, fe, theta_bar, cov, rho=1.0, lam=1.0e-6,
                      I_band=(0.006, 0.5), monotonic=True, mu_min=0.05):
    """Pool the grid-consistent rows over `dumps` (equal-weighted), add the family prior as
    augmented rows, solve the constrained QP. Returns theta."""
    A_all, b_all = [], []
    for dp in dumps:
        gs = assemble_grid_consistent(load_dump(dp), fe, flow_frac_min=0.90)
        if gs.n_rows == 0:
            continue
        sc = np.linalg.norm(gs.A) + 1e-30           # equal contribution per collapse
        A_all.append(gs.A / sc); b_all.append(gs.b / sc)
    A = np.vstack(A_all); b = np.concatenate(b_all)
    if rho > 1e-9:
        M = rho * np.linalg.inv(cov)
        L = np.linalg.cholesky(M)                    # M = L L^T (SPD)
        A = np.vstack([A, L]); b = np.concatenate([b, L @ theta_bar])
    Icon = np.logspace(np.log10(I_band[0]), np.log10(I_band[1]), 40)
    G = fe.gram((10.0 ** np.linspace(-4, 0, 257), np.ones(257)))
    qp = constrained_solve(A, b, fe, lam=lam, G=G, mu_min=mu_min,
                           I_constraint_grid=Icon, nonnegativity=False, monotonic=monotonic)
    return qp.theta, int(A.shape[0])


def run(dumps=None, rhos=(0.0, 0.3, 1.0, 3.0), I_band=(0.006, 0.5)):
    if dumps is None:
        dumps = [f"out/dumps/{x}.npz" for x in
                 ("poul_a1", "column_pouliquen_a2", "poul_a3", "poul_a4")]
    dumps = [d for d in dumps if Path(d).exists()]
    fe, theta_bar, cov = _load_fe_with_prior()
    lp = load_dump(dumps[1]).meta.law_params
    Ig = np.logspace(np.log10(I_band[0]), np.log10(I_band[1]), 80)
    mu_true = lp["mu_s"] + lp["delta_mu"] * Ig / (Ig + lp["I0"])
    print(f"truth pouliquen: mu_s={lp['mu_s']:.2f} delta_mu={lp['delta_mu']:.2f} I0={lp['I0']:.2f}"
          f"   pooled {len(dumps)} aspect ratios over I={I_band}")
    out = []
    for rho in rhos:
        theta, nrows = joint_prior_solve(dumps, fe, theta_bar, cov, rho=rho, I_band=I_band)
        mu_hat = fe.phi(Ig) @ theta
        relL2 = float(np.sqrt(np.mean((mu_hat - mu_true) ** 2)) / np.sqrt(np.mean(mu_true ** 2)))
        mono = bool(np.all(np.diff(mu_hat) >= -1e-6))
        tag = "constrained (no prior)" if rho == 0 else f"+ family prior rho={rho}"
        print(f"  {tag:28s}  relL2={relL2:.3f}  mu_hat[{mu_hat.min():.3f},{mu_hat.max():.3f}]"
              f"  monotone={mono}")
        out.append({"rho": rho, "relL2": relL2, "Ig": Ig, "mu_hat": mu_hat, "mu_true": mu_true})
    return out


if __name__ == "__main__":
    run()
