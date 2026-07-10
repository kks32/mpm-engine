"""Divergence-free Bubnov-Galerkin weak form on the reconstruction basis.

The analytic single-bump test functions are intrinsically patch-scale-biased
for column-collapse flow (shown by the scale sweeps), and the grid-consistent
nodal form needs the true pressure gradient (unavailable for perceived data).
The form that is BOTH divergence-free (so a pressure closure can stand in --
the pressure term drops) AND unbiased (consistent test functions) uses

    w_i = curl(eta_i e_y),   eta_i = the streamfunction B-spline basis function i

i.e. exactly the methodology's w = curl(eta) (MATH_REFERENCE Section 3) but
with eta ranging over the reconstruction's own B-spline basis instead of a
single bump. This is Bubnov-Galerkin (test space = the field's trial space),
the perceived-data analogue of the grid-consistent (MPM-grid) form. Only
INTERIOR basis functions (support fully inside the flowing region) are used,
the analogue of EUCLID's free degrees of freedom.

  w_x = -d eta/d z = -B_a(x) B'_b(z)
  w_z = +d eta/d x =  B'_a(x) B_b(z)
  D[w] from second derivatives; div w = 0 identically.

Assembled by dense quadrature on the reconstructed fields. Pure ident.
"""

from __future__ import annotations

import numpy as np

from common.conventions import EPS_GAMMA_DEFAULT, equivalent_shear_rate, gravity_vector_inplane
from ident.features.base import Dictionary
from ident.weakform.field_reconstruction import _basis_and_derivs, _clamped_uniform_knots, _kron_rows


def _interior_mask(n_basis: int, k: int = 3, drop: int = 2) -> np.ndarray:
    """Keep basis functions whose full support is strictly inside a clamped domain."""
    drop = max(drop, k + 1)
    keep = np.ones(n_basis, bool)
    keep[:drop] = False
    keep[n_basis - drop:] = False
    return keep


def _support_inside_frame_mask(fr, tx, tz, k, nx, nz, keep_x, keep_z) -> np.ndarray:
    """Whether each retained tensor-product basis has support only on observed points.

    Frames used here are dense quadrature grids. A basis is admissible only if its
    sampled support is nonempty and every quadrature point in that support is inside
    the flowing mask; otherwise masking would introduce an unmodelled internal boundary.
    """
    X = np.asarray(fr.x, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or not np.isfinite(X).all():
        raise ValueError("frame quadrature coordinates must be finite with shape (Q, 2)")
    mask = np.ones(len(X), dtype=bool) if fr.mask is None else np.asarray(fr.mask, dtype=bool)
    if mask.shape != (len(X),):
        raise ValueError("frame mask must have one entry per quadrature point")
    Bx = _basis_and_derivs(tx, k, X[:, 0], nx, nderiv=0)[0][:, keep_x]
    Bz = _basis_and_derivs(tz, k, X[:, 1], nz, nderiv=0)[0][:, keep_z]
    sx = np.abs(Bx) > 1e-12
    sz = np.abs(Bz) > 1e-12
    inside_hits = sx[mask].astype(np.int64).T @ sz[mask].astype(np.int64)
    outside_hits = sx[~mask].astype(np.int64).T @ sz[~mask].astype(np.int64)
    return ((inside_hits > 0) & (outside_hits == 0)).reshape(-1)


def assemble_galerkin_field(
    frames,                       # list of dense-quad FrameData (v, D, p, I, vol, rho, mask)
    dictionary: Dictionary,
    bbox,                         # (x0,x1,z0,z1) test-function streamfunction grid extent
    n_knots=(14, 7),
    eps_gamma: float = EPS_GAMMA_DEFAULT,
    drop_boundary: int = 2,
):
    """Assemble A (J,K), b_acc (J,), b_tw (J,) with streamfunction-basis test fields."""
    k = 3
    tx = _clamped_uniform_knots(bbox[0], bbox[1], n_knots[0], k)
    tz = _clamped_uniform_knots(bbox[2], bbox[3], n_knots[1], k)
    nx = len(tx) - k - 1
    nz = len(tz) - k - 1
    keep_x = _interior_mask(nx, k, drop_boundary)
    keep_z = _interior_mask(nz, k, drop_boundary)
    # interior tensor test functions
    ia = np.where(keep_x)[0]
    ib = np.where(keep_z)[0]
    pairs = [(a, b) for a in ia for b in ib]
    valid = np.ones(len(pairs), dtype=bool)
    for fr in frames:
        valid &= _support_inside_frame_mask(fr, tx, tz, k, nx, nz, keep_x, keep_z)
    pair_keep = valid
    pairs = [pair for pair, use in zip(pairs, valid) if use]
    J = len(pairs)
    K = dictionary.K
    g = gravity_vector_inplane()

    A = np.zeros((J, K))
    b_acc = np.zeros(J)
    b_tw = np.zeros(J)
    times = np.array([f.t for f in frames])
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("frames must be strictly increasing in time")
    # trapezoid weights in time
    tw = np.zeros(len(frames))
    if len(frames) > 1:
        dt = np.diff(times)
        tw[:-1] += 0.5 * dt
        tw[1:] += 0.5 * dt

    endpoint_momentum = np.zeros((len(frames), J))

    for fi, (w_t, fr) in enumerate(zip(tw, frames)):
        if w_t == 0 or (fr.mask is not None and not np.any(fr.mask)):
            continue
        m = slice(None) if fr.mask is None else fr.mask
        X = fr.x[m]
        vol = fr.vol[m]; rho = fr.rho[m]; p = fr.p[m]
        D = fr.D[m]; v = fr.v[m]; I = fr.I[m]
        gd = equivalent_shear_rate(D, eps_gamma)
        flow = 2.0 * D / gd[:, None, None]
        f_xx = flow[:, 0, 0]; f_xz = flow[:, 0, 1]; f_zz = flow[:, 1, 1]
        Phi = dictionary.phi(np.clip(np.nan_to_num(I, posinf=1e6), 1e-12, 1e6))  # (Q,K)

        Bx, dBx, ddBx = _basis_and_derivs(tx, k, X[:, 0], nx)
        Bz, dBz, ddBz = _basis_and_derivs(tz, k, X[:, 1], nz)
        # restrict to interior basis columns
        Bx, dBx, ddBx = Bx[:, keep_x], dBx[:, keep_x], ddBx[:, keep_x]
        Bz, dBz, ddBz = Bz[:, keep_z], dBz[:, keep_z], ddBz[:, keep_z]

        # per quad point, per test function (a,b):
        #   Dw_xx = -Bx'_a Bz'_b ; Dw_zz = +Bx'_a Bz'_b ; Dw_xz = 0.5(-Bx_a Bz''_b + Bx''_a Bz_b)
        # contraction flow:Dw = (f_zz-f_xx) Bx'_a Bz'_b + f_xz(-Bx_a Bz''_b + Bx''_a Bz_b)
        dXdZ = _kron_rows(dBx, dBz)[:, pair_keep]         # (Q, J) : Bx'_a Bz'_b
        mixed = (-_kron_rows(Bx, ddBz) + _kron_rows(ddBx, Bz))[:, pair_keep]
        contr = (f_zz - f_xx)[:, None] * dXdZ + f_xz[:, None] * mixed  # (Q, J)

        # A[j,k] += w_t sum_q vol_q p_q phi_k(q) contr[q,j]
        vpphi = (vol * p)[:, None] * Phi                  # (Q, K)
        A += w_t * (contr.T @ vpphi)                      # (J, K)

        # test field components for the load: w_x=-Bx_a Bz'_b, w_z= Bx'_a Bz_b
        Wx = -_kron_rows(Bx, dBz)[:, pair_keep]             # (Q, J)
        Wz = _kron_rows(dBx, Bz)[:, pair_keep]               # (Q, J)
        # acceleration form b = sum vol rho (g-a).w
        body = rho[:, None] * (g[None, :] - fr.a[m])
        b_acc += w_t * (Wx.T @ (vol * body[:, 0]) + Wz.T @ (vol * body[:, 1]))
        # Time-weak interior integrand for a time-independent spatial test. The
        # non-vanishing temporal endpoints are handled explicitly below.
        #   (v outer v):grad w = v_i v_j d w_i/d x_j
        # grad w: dwx/dx=-Bx'_a Bz'_b, dwx/dz=-Bx_a Bz''_b, dwz/dx=Bx''_a Bz_b, dwz/dz=Bx'_a Bz'_b
        gwxx = -dXdZ; gwxz = -_kron_rows(Bx, ddBz)[:, pair_keep]
        gwzx = _kron_rows(ddBx, Bz)[:, pair_keep]; gwzz = dXdZ
        vv_xx = (v[:, 0] * v[:, 0])[:, None]; vv_xz = (v[:, 0] * v[:, 1])[:, None]
        vv_zz = (v[:, 1] * v[:, 1])[:, None]
        conv = (vv_xx * gwxx + vv_xz * (gwxz + gwzx) + vv_zz * gwzz)  # (Q,J)
        grav = g[0] * Wx + g[1] * Wz                       # (Q,J)
        b_tw += w_t * ((grav.T @ (vol * rho)) + (conv.T @ (vol * rho)))
        endpoint_momentum[fi] = (
            Wx.T @ (vol * rho * v[:, 0]) + Wz.T @ (vol * rho * v[:, 1])
        )

    # Integration by parts in time contributes -[INT rho v.w dV] at the clip
    # endpoints. Omitting this is only valid for temporally compact tests.
    if len(frames) > 1:
        b_tw -= endpoint_momentum[-1] - endpoint_momentum[0]

    return A, b_acc, b_tw, J
