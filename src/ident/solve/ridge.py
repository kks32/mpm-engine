"""Closed-form ridge estimator and conditional posterior (MATH_REFERENCE.md 8.1, 8.2).

theta = argmin |A theta - b|^2 + lambda theta^T G theta
      = (A^T A + lambda G)^{-1} A^T b

Constrained variants (admissibility, monotonicity, Mode P nonnegativity) use
OSQP and live in ident/solve/qp.py. Gate logic uses these two paths only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

EFFECTIVE_RANK_TOL: float = 1.0e-8


@dataclass
class RidgeResult:
    theta: np.ndarray
    cond_AtA: float
    effective_rank: int
    sigma2: float
    Sigma_theta: np.ndarray
    dof_lambda: float
    residual_rel: float


def ridge_solve(
    A: np.ndarray,
    b: np.ndarray,
    lam: float = 0.0,
    G: np.ndarray | None = None,
) -> RidgeResult:
    J, K = A.shape
    if G is None:
        G = np.eye(K)
    M = A.T @ A + lam * G

    sv = linalg.svdvals(A)
    sv2 = sv * sv
    cond = float(sv2[0] / sv2[-1]) if sv2[-1] > 0.0 else np.inf
    eff_rank = int(np.sum(sv2 > EFFECTIVE_RANK_TOL * sv2[0]))

    theta = linalg.solve(M, A.T @ b, assume_a="pos")

    r = A @ theta - b
    Minv_At = linalg.solve(M, A.T, assume_a="pos")  # (K, J)
    dof = float(np.einsum("kj,jk->", Minv_At, A))
    denom = max(J - dof, 1.0)
    sigma2 = float(r @ r) / denom
    Minv = linalg.solve(M, np.eye(K), assume_a="pos")
    Sigma_theta = sigma2 * Minv

    bnorm = float(np.linalg.norm(b))
    res_rel = float(np.linalg.norm(r)) / bnorm if bnorm > 0.0 else np.inf

    return RidgeResult(
        theta=theta,
        cond_AtA=cond,
        effective_rank=eff_rank,
        sigma2=sigma2,
        Sigma_theta=Sigma_theta,
        dof_lambda=dof,
        residual_rel=res_rel,
    )


def mu_band(dictionary_phi: np.ndarray, Sigma_theta: np.ndarray) -> np.ndarray:
    """var(mu(I)) = phi(I)^T Sigma_theta phi(I) per evaluation point."""
    return np.einsum("nk,kl,nl->n", dictionary_phi, Sigma_theta, dictionary_phi)
