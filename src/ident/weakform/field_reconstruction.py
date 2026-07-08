"""Streamfunction B-spline velocity-field reconstruction (MATH_REFERENCE 6.5).

For perceived/scattered kinematics (CoTracker3 tracks, or oracle particles) we
reconstruct a smooth, divergence-free in-plane velocity field by fitting a
streamfunction psi(x, z) on a tensor cubic B-spline grid:

    v = curl(psi e_y):   v_x = -d psi/d z,   v_z = +d psi/d x

so div v = 0 by construction. psi's coefficients are linear in the fitted
velocity, so the fit is a regularized linear least squares (confidence-weighted
track misfit + Laplacian smoothness). The analytic spline derivatives give the
rate of deformation D = sym(grad v) exactly:

    dv_x/dx = -psi_xz,  dv_x/dz = -psi_zz,  dv_z/dx = psi_xx,  dv_z/dz = psi_xz

(so dv_x/dx + dv_z/dz = 0). Evaluating the weak form on this smooth field by
dense quadrature avoids the particle-scatter quadrature bias that the
grid-consistent assembly was built to dodge; the smooth field plus regular
quadrature is the G0 regime. This is the engine the G3 rendered-video closure
feeds; here it is validated on oracle kinematics first.

Pure ident (numpy + scipy); no warp/torch.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from common.conventions import sym


def _clamped_uniform_knots(x0: float, x1: float, n_interior: int, k: int = 3):
    """Clamped uniform cubic knot vector with n_interior interior knots."""
    interior = np.linspace(x0, x1, n_interior + 2)[1:-1]
    return np.concatenate([[x0] * (k + 1), interior, [x1] * (k + 1)])


def _basis_and_derivs(t: np.ndarray, k: int, x: np.ndarray, n_basis: int, nderiv: int = 2):
    """Return [B, B', B''] each (len(x), n_basis) for the 1D B-spline basis."""
    x = np.clip(x, t[0], t[-1] - 1e-12)
    out = []
    for d in range(nderiv + 1):
        Bd = np.zeros((x.shape[0], n_basis))
        for a in range(n_basis):
            c = np.zeros(n_basis)
            c[a] = 1.0
            spl = BSpline(t, c, k)
            Bd[:, a] = spl(x) if d == 0 else spl.derivative(d)(x)
        out.append(Bd)
    return out


class StreamFunctionField:
    """Fit psi on a tensor cubic B-spline grid; expose v and D at query points."""

    def __init__(
        self,
        x: np.ndarray,            # (P, 2) in-plane positions
        v: np.ndarray,            # (P, 2) in-plane velocities
        n_knots: tuple[int, int] = (16, 8),
        lam_smooth: float = 1e-4,
        weights: np.ndarray | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ):
        self.k = 3
        x = np.asarray(x, float)
        v = np.asarray(v, float)
        if bbox is None:
            pad_x = 0.05 * (np.ptp(x[:, 0]) + 1e-9)
            pad_z = 0.05 * (np.ptp(x[:, 1]) + 1e-9)
            bbox = (x[:, 0].min() - pad_x, x[:, 0].max() + pad_x,
                    x[:, 1].min() - pad_z, x[:, 1].max() + pad_z)
        self.bbox = bbox
        self.tx = _clamped_uniform_knots(bbox[0], bbox[1], n_knots[0], self.k)
        self.tz = _clamped_uniform_knots(bbox[2], bbox[3], n_knots[1], self.k)
        self.nx = len(self.tx) - self.k - 1
        self.nz = len(self.tz) - self.k - 1
        self.ncoef = self.nx * self.nz

        Bx, dBx, _ = _basis_and_derivs(self.tx, self.k, x[:, 0], self.nx)
        Bz, dBz, _ = _basis_and_derivs(self.tz, self.k, x[:, 1], self.nz)
        # v_x = -sum c_ab Bx_a dBz_b ;  v_z = sum c_ab dBx_a Bz_b
        Mvx = -_kron_rows(Bx, dBz)      # (P, ncoef)
        Mvz = _kron_rows(dBx, Bz)
        M = np.vstack([Mvx, Mvz])       # (2P, ncoef)
        target = np.concatenate([v[:, 0], v[:, 1]])
        if weights is not None:
            w = np.concatenate([weights, weights])
            M = M * np.sqrt(w)[:, None]
            target = target * np.sqrt(w)
        S = self._smoothness_penalty()
        A = M.T @ M + lam_smooth * (M.T @ M).trace() / self.ncoef * S
        self.c = np.linalg.solve(A, M.T @ target)
        # fit diagnostics
        self._fit_resid = float(np.linalg.norm(M @ self.c - target) /
                                max(np.linalg.norm(target), 1e-12))

    def _smoothness_penalty(self) -> np.ndarray:
        """Second-difference penalty on the coefficient grid (approx Laplacian)."""
        nx, nz = self.nx, self.nz
        Dx = _second_diff(nx)
        Dz = _second_diff(nz)
        Sx = np.kron(Dx.T @ Dx, np.eye(nz))
        Sz = np.kron(np.eye(nx), Dz.T @ Dz)
        return Sx + Sz

    @property
    def fit_residual(self) -> float:
        return self._fit_resid

    def _eval(self, X: np.ndarray):
        X = np.atleast_2d(np.asarray(X, float))
        Bx, dBx, ddBx = _basis_and_derivs(self.tx, self.k, X[:, 0], self.nx)
        Bz, dBz, ddBz = _basis_and_derivs(self.tz, self.k, X[:, 1], self.nz)
        C = self.c.reshape(self.nx, self.nz)
        # helper: f = sum_ab C_ab Ux_a(x) Uz_b(z) = (Ux @ C) . Uz row-wise
        def comb(Ux, Uz):
            return np.einsum("pa,ab,pb->p", Ux, C, Uz)
        v_x = -comb(Bx, dBz)
        v_z = comb(dBx, Bz)
        dvx_dx = -comb(dBx, dBz)
        dvx_dz = -comb(Bx, ddBz)
        dvz_dx = comb(ddBx, Bz)
        dvz_dz = comb(dBx, dBz)
        v = np.stack([v_x, v_z], axis=-1)
        L = np.stack([np.stack([dvx_dx, dvx_dz], -1),
                      np.stack([dvz_dx, dvz_dz], -1)], axis=-2)
        return v, L

    def velocity(self, X: np.ndarray) -> np.ndarray:
        return self._eval(X)[0]

    def grad_v(self, X: np.ndarray) -> np.ndarray:
        return self._eval(X)[1]

    def D(self, X: np.ndarray) -> np.ndarray:
        return sym(self._eval(X)[1])


def _kron_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Kronecker: (P,na),(P,nb) -> (P, na*nb) with [p, a*nb+b]=A[p,a]B[p,b]."""
    P, na = A.shape
    nb = B.shape[1]
    return (A[:, :, None] * B[:, None, :]).reshape(P, na * nb)


def _second_diff(n: int) -> np.ndarray:
    if n < 3:
        return np.zeros((max(n - 2, 0), n))
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0; D[i, i + 1] = -2.0; D[i, i + 2] = 1.0
    return D
