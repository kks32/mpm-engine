"""Pressure sensitivity operator (MATH_REFERENCE.md Section 5).

theta(p) = (A^T A + lambda G)^{-1} A^T b. A pressure perturbation field
delta_p enters each A entry through two channels: the multiplicative p factor
and the basis argument with d phi_k / d p = (d phi_k / d I)(-I / (2 p)).
Per integrand:

  d/dp [ phi_k(I(p)) p ] delta_p = [ phi_k(I) - (I / 2) dphi_dI(I) ] delta_p

This module assembles dA for a given delta_p field and applies the closed
form, which G0 checks against finite differences. NOTE: dphi_dI is the
PHYSICAL-I derivative per the Section 7 contract, never dphi_dlogI.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from common.conventions import EPS_GAMMA_DEFAULT, equivalent_shear_rate
from ident.features.base import Dictionary
from ident.weakform.assembly import FrameData, _trapezoid_weights
from ident.weakform.test_functions import BumpTestFunction


def assemble_dA(
    frames: list[FrameData],
    rows: list[BumpTestFunction],
    dictionary: Dictionary,
    dp_frames: list[np.ndarray],
    eps_gamma: float = EPS_GAMMA_DEFAULT,
) -> np.ndarray:
    """dA[j, k] for the pressure perturbation delta_p given per frame (P,)."""
    times = np.array([f.t for f in frames])
    J, K = len(rows), dictionary.K
    dA = np.zeros((J, K))

    for j, tf in enumerate(rows):
        t_lo, t_hi = tf.t_window
        sel = np.where((times > t_lo) & (times < t_hi))[0]
        if sel.size < 2:
            continue
        lo = max(sel[0] - 1, 0)
        hi = min(sel[-1] + 1, len(frames) - 1)
        idx = np.arange(lo, hi + 1)
        tw = _trapezoid_weights(times[idx])

        for w_t, fi in zip(tw, idx):
            fr = frames[fi]
            dp = dp_frames[fi]
            if fr.mask is not None:
                m = fr.mask
                x, D = fr.x[m], fr.D[m]
                I, vol = fr.I[m], fr.vol[m]
                dp_ = dp[m]
            else:
                x, D, I, vol, dp_ = fr.x, fr.D, fr.I, fr.vol, dp
            xs, zs = x[:, 0], x[:, 1]
            inside = tf.in_support(xs, zs, fr.t)
            if not np.any(inside):
                continue
            xs, zs = xs[inside], zs[inside]
            D_, I_, vol_, dpi = D[inside], I[inside], vol[inside], dp_[inside]

            Dw = tf.D_w(xs, zs, fr.t)
            gd = equivalent_shear_rate(D_, eps_gamma)
            flow_dir = 2.0 * D_ / gd[:, None, None]
            contr = np.einsum("pij,pij->p", flow_dir, Dw)

            Phi = dictionary.phi(I_)
            dPhi = dictionary.dphi_dI(I_)
            chan = Phi - 0.5 * I_[:, None] * dPhi  # both channels combined

            dA[j] += w_t * np.einsum("p,pk->k", vol_ * dpi * contr, chan)

    return dA


def theta_sensitivity(
    A: np.ndarray,
    b: np.ndarray,
    theta: np.ndarray,
    dA: np.ndarray,
    lam: float = 0.0,
    G: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form d theta for the perturbation that produced dA.

    With M = A^T A + lambda G and theta = M^{-1} A^T b:
      d theta = M^{-1} [ dA^T (b - A theta) - A^T dA theta ]
    """
    K = A.shape[1]
    if G is None:
        G = np.eye(K)
    M = A.T @ A + lam * G
    rhs = dA.T @ (b - A @ theta) - A.T @ (dA @ theta)
    return linalg.solve(M, rhs, assume_a="pos")
