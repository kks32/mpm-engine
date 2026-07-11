"""Compact, divergence-free space-time B-spline test functions.

This is the divergence-free analogue of the grid-consistent nodal balance. It uses
many local test functions of the form

    w_{abc}(x, z, t) = curl_xz( eta_a(x) eta_b(z) eta_c(t) e_y )
                     = ( -B_a(x) B'_b(z), +B'_a(x) B_b(z) ) * B_c(t)

with B_a, B_b local cubic streamfunction B-splines (compact over a few cells)
and B_c a local cubic temporal B-spline. Divergence free by construction (so a
pressure closure can stand in; the pressure term drops). Compact in space and
time. The temporal factor prevents the time-weak load from canceling over a moving
flowing region, as it does for a time-constant test function. Only interior
(a, b, c) are kept, with support inside the space-time flowing region; these are
the analogue of EUCLID's free DoFs.

Space-time separability factorizes the assembly: spatial integrands are formed
per frame, then combined with the temporal basis B_c(t_f) and its derivative
B'_c(t_f). The space-time weak form is

    INT_Q tau:D[w] = INT_Q rho (g - a).w   (admissible, div w = 0)

solved as the time-weak (M2) variant, so no acceleration data is needed:
    A[(ab,c),k] = sum_f dt B_c(t_f) [ INT_x V phi_k p (2D/|gd|):D[w_s^{ab}] ]_f
    b[(ab,c)]   = sum_f dt [ B_c(t_f)(INT rho g.w_s + INT rho (v x v):grad w_s)
                            + B'_c(t_f) INT rho v.w_s ]_f
Pure ident.
"""

from __future__ import annotations

import numpy as np

from common.conventions import EPS_GAMMA_DEFAULT, equivalent_shear_rate, gravity_vector_inplane
from ident.features.base import Dictionary
from ident.weakform.field_reconstruction import _basis_and_derivs, _clamped_uniform_knots, _kron_rows
from ident.weakform.galerkin_field import _interior_mask, _support_inside_frame_mask


def assemble_galerkin_spacetime(
    frames,                 # list of dense-quad FrameData (v, D, p, I, vol, rho, mask)
    dictionary: Dictionary,
    bbox,                   # (x0,x1,z0,z1) spatial streamfunction grid extent
    n_knots_space=(14, 7),
    n_knots_time=6,
    eps_gamma: float = EPS_GAMMA_DEFAULT,
    drop_boundary_space: int = 2,
    drop_boundary_time: int = 1,
):
    k = 3
    tx = _clamped_uniform_knots(bbox[0], bbox[1], n_knots_space[0], k)
    tz = _clamped_uniform_knots(bbox[2], bbox[3], n_knots_space[1], k)
    nx = len(tx) - k - 1
    nz = len(tz) - k - 1
    keep_x = _interior_mask(nx, k, drop_boundary_space)
    keep_z = _interior_mask(nz, k, drop_boundary_space)
    ia = np.where(keep_x)[0]
    ib = np.where(keep_z)[0]
    J_s = len(ia) * len(ib)
    K = dictionary.K
    g = gravity_vector_inplane()

    times = np.array([f.t for f in frames])
    tt = _clamped_uniform_knots(times.min(), times.max(), n_knots_time, k)
    nt = len(tt) - k - 1
    keep_t = _interior_mask(nt, k, drop_boundary_time)
    ic = np.where(keep_t)[0]
    J_t = len(ic)
    Bc_all, dBc_all, _ = _basis_and_derivs(tt, k, times, nt)
    Bc = Bc_all[:, keep_t]            # (F, J_t)
    dBc = dBc_all[:, keep_t]
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("frames must be strictly increasing in time")
    if np.any(np.abs(Bc[[0, -1]]) > 1e-12):
        raise ValueError("time-weak test functions must vanish at both clip endpoints")
    dt = np.zeros_like(times)
    frame_dt = np.diff(times)
    dt[:-1] += 0.5 * frame_dt
    dt[1:] += 0.5 * frame_dt

    valid_space = np.stack([
        _support_inside_frame_mask(fr, tx, tz, k, nx, nz, keep_x, keep_z)
        for fr in frames
    ])
    temporal_support = (np.abs(Bc) + np.abs(dBc)) > 1e-12
    valid_rows = np.all(
        ~temporal_support[:, :, None] | valid_space[:, None, :], axis=0
    ).reshape(-1)

    # per-frame spatial integrands
    F = len(frames)
    SA = np.zeros((F, J_s, K))        # stress-power (linear in mu via phi_k)
    Sgrav = np.zeros((F, J_s))
    Sconv = np.zeros((F, J_s))
    Svw = np.zeros((F, J_s))          # INT rho v . w_s

    for fi, fr in enumerate(frames):
        if fr.mask is not None and not np.any(fr.mask):
            continue
        m = slice(None) if fr.mask is None else fr.mask
        X = fr.x[m]; vol = fr.vol[m]; rho = fr.rho[m]; p = fr.p[m]
        D = fr.D[m]; v = fr.v[m]; I = fr.I[m]
        gd = equivalent_shear_rate(D, eps_gamma)
        flow = 2.0 * D / gd[:, None, None]
        f_xx = flow[:, 0, 0]; f_xz = flow[:, 0, 1]; f_zz = flow[:, 1, 1]
        Phi = dictionary.phi(np.clip(np.nan_to_num(I, posinf=1e6), 1e-12, 1e6))

        Bx, dBx, ddBx = _basis_and_derivs(tx, k, X[:, 0], nx)
        Bz, dBz, ddBz = _basis_and_derivs(tz, k, X[:, 1], nz)
        Bx, dBx, ddBx = Bx[:, keep_x], dBx[:, keep_x], ddBx[:, keep_x]
        Bz, dBz, ddBz = Bz[:, keep_z], dBz[:, keep_z], ddBz[:, keep_z]

        dXdZ = _kron_rows(dBx, dBz)                          # Bx'_a Bz'_b
        mixed = -_kron_rows(Bx, ddBz) + _kron_rows(ddBx, Bz)
        contr = (f_zz - f_xx)[:, None] * dXdZ + f_xz[:, None] * mixed
        Wx = -_kron_rows(Bx, dBz)
        Wz = _kron_rows(dBx, Bz)
        # grad w_s: dwx/dx=-Bx'Bz', dwx/dz=-Bx Bz'', dwz/dx=Bx''Bz, dwz/dz=Bx'Bz'
        gwxx = -dXdZ; gwxz = -_kron_rows(Bx, ddBz)
        gwzx = _kron_rows(ddBx, Bz); gwzz = dXdZ

        vpphi = (vol * p)[:, None] * Phi                     # (Q, K)
        SA[fi] = contr.T @ vpphi                             # (J_s, K)
        Sgrav[fi] = Wx.T @ (vol * rho * g[0]) + Wz.T @ (vol * rho * g[1])
        Svw[fi] = Wx.T @ (vol * rho * v[:, 0]) + Wz.T @ (vol * rho * v[:, 1])
        vv_xx = vol * rho * v[:, 0] * v[:, 0]
        vv_xz = vol * rho * v[:, 0] * v[:, 1]
        vv_zz = vol * rho * v[:, 1] * v[:, 1]
        Sconv[fi] = (gwxx.T @ vv_xx + (gwxz + gwzx).T @ vv_xz + gwzz.T @ vv_zz)

    # combine with temporal basis: rows indexed by (c, ab)
    wdt_Bc = dt[:, None] * Bc                                # (F, J_t)
    wdt_dBc = dt[:, None] * dBc
    A = np.einsum("fc,fak->cak", wdt_Bc, SA).reshape(J_t * J_s, K)
    b = (np.einsum("fc,fa->ca", wdt_Bc, Sgrav + Sconv)
         + np.einsum("fc,fa->ca", wdt_dBc, Svw)).reshape(J_t * J_s)
    return A[valid_rows], b[valid_rows], int(np.count_nonzero(valid_rows))
