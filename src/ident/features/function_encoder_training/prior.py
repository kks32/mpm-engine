"""Offline family-coefficient prior for the learned mu(I) basis.

The learned basis spans the material family; a single under-excited flow leaves its K
coefficients under-determined, so the unconstrained fit overfits (it can go negative /
oscillate). Encoding each corpus material into the frozen basis (the same weighted
least-squares the basis was trained with) gives a coefficient cloud {theta_m}; its mean
theta_bar and covariance Sigma define a Gaussian prior over PLAUSIBLE laws. The EUCLID
solve then adds (theta - theta_bar)^T Sigma^-1 (theta - theta_bar), which shrinks the
recovery onto the family manifold under poor excitation. Pure numpy; writes theta_mean and
theta_cov into the fe table so the (numpy-only) solve can load them with no torch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .corpus import build_corpus, weight_vector


def compute_prior(table_path="mpm_engine/fe-weights/granular_mu_i.npz", n_materials=2000,
                  seed=0, eps_rel=1.0e-6, cov_reg=1.0e-3):
    d = dict(np.load(table_path))
    Phi = np.asarray(d["table"], float)                 # (N_S, K)
    s = np.asarray(d["s_grid"], float)
    w = np.asarray(d["weight"], float) if "weight" in d else weight_vector()
    K = Phi.shape[1]
    W = w * np.gradient(s)                               # trapezoid-weighted (N_S,)
    mus, _ = build_corpus(n_materials, seed=seed)        # (M, N_S)
    PtW = Phi.T * W                                      # (K, N_S)
    G = PtW @ Phi                                        # (K, K)
    G = G + eps_rel * (np.trace(G) / K) * np.eye(K)
    Theta = np.linalg.solve(G, PtW @ mus.T).T            # (M, K) per-material coefficients
    theta_mean = Theta.mean(0)
    theta_cov = np.cov(Theta, rowvar=False)
    theta_cov = theta_cov + cov_reg * (np.trace(theta_cov) / K) * np.eye(K)
    d["theta_mean"] = theta_mean
    d["theta_cov"] = theta_cov
    np.savez(table_path, **d)
    print(f"family prior: K={K}, {n_materials} materials, theta_mean="
          f"{np.round(theta_mean, 3)}, cond(cov)={np.linalg.cond(theta_cov):.1e}")
    return theta_mean, theta_cov


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "mpm_engine/fe-weights/granular_mu_i.npz"
    compute_prior(Path(p))
