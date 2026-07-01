"""Momentum-closure diagnostic (MATH_REFERENCE.md Section 2.3).

Built BEFORE any regression. For the true material the master identity holds
for any admissible w. Two row families per patch:

  near-rigid w = chi e_i (not divergence free): checks the FULL identity
      INT_Q sigma : grad w = INT_Q rho (g - a) . w
  with the data stress sigma (oracle dump or closure pressure plus model tau).

  strictly admissible bumps: check the pressure-free identity
      INT_Q tau : D[w] = INT_Q rho (g - a) . w
  with tau = sigma + p Id recomposed from data (in-plane block; sigma_yy
  never enters in-plane rows).

The per-patch relative closure error localizes bad kinematics, pressure, or
masks before any theta is fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.conventions import G_MAG, gravity_vector_inplane
from ident.weakform.assembly import FrameData, _trapezoid_weights
from ident.weakform.test_functions import BumpTestFunction, bump, bump_d1


@dataclass(frozen=True)
class RigidTestField:
    """w = chi e_i with chi = B(u) B(s) B(q), compact but not divergence free."""

    x_c: float
    z_c: float
    t_c: float
    r_x: float
    r_z: float
    r_t: float
    component: int  # 0 -> e_x, 1 -> e_z

    def _usq(self, x, z, t):
        return (
            (np.asarray(x, dtype=float) - self.x_c) / self.r_x,
            (np.asarray(z, dtype=float) - self.z_c) / self.r_z,
            (np.asarray(t, dtype=float) - self.t_c) / self.r_t,
        )

    def chi(self, x, z, t):
        u, s, q = self._usq(x, z, t)
        return bump(u) * bump(s) * bump(q)

    def grad_chi(self, x, z, t):
        u, s, q = self._usq(x, z, t)
        gx = bump_d1(u) * bump(s) * bump(q) / self.r_x
        gz = bump(u) * bump_d1(s) * bump(q) / self.r_z
        return np.stack([gx, gz], axis=-1)

    def in_support(self, x, z, t):
        u, s, q = self._usq(x, z, t)
        return (np.abs(u) < 1.0) & (np.abs(s) < 1.0) & (np.abs(q) < 1.0)

    @property
    def t_window(self) -> tuple[float, float]:
        return (self.t_c - self.r_t, self.t_c + self.r_t)


@dataclass
class ClosureReport:
    rel_error_full: list[float]      # near-rigid rows, full identity
    rel_error_admissible: list[float]  # bump rows, pressure-free identity

    @property
    def worst_full(self) -> float:
        return max(self.rel_error_full) if self.rel_error_full else np.nan

    @property
    def worst_admissible(self) -> float:
        return max(self.rel_error_admissible) if self.rel_error_admissible else np.nan


def _row_frames(times: np.ndarray, t_window: tuple[float, float]):
    sel = np.where((times > t_window[0]) & (times < t_window[1]))[0]
    if sel.size < 2:
        return None, None
    lo = max(sel[0] - 1, 0)
    hi = min(sel[-1] + 1, len(times) - 1)
    idx = np.arange(lo, hi + 1)
    return idx, _trapezoid_weights(times[idx])


def closure_diagnostic(
    frames: list[FrameData],
    sigma_frames: list[np.ndarray],
    patches: list[tuple[float, float, float, float, float, float]],
) -> ClosureReport:
    """sigma_frames: in-plane Cauchy blocks (P, 2, 2) aligned with frames.

    patches: (x_c, z_c, t_c, r_x, r_z, r_t) boxes.
    """
    times = np.array([f.t for f in frames])
    g_vec = gravity_vector_inplane()
    full_errors: list[float] = []
    adm_errors: list[float] = []

    for patch in patches:
        # near-rigid rows, both components
        for comp in (0, 1):
            rf = RigidTestField(*patch, component=comp)
            idx, tw = _row_frames(times, rf.t_window)
            if idx is None:
                continue
            lhs = rhs = 0.0
            scale = 0.0
            for w_t, fi in zip(tw, idx):
                fr, sig = frames[fi], sigma_frames[fi]
                keep = fr.mask if fr.mask is not None else np.ones(len(fr.x), bool)
                xs, zs = fr.x[:, 0], fr.x[:, 1]
                inside = rf.in_support(xs, zs, fr.t) & keep
                if not np.any(inside):
                    continue
                chi = rf.chi(xs[inside], zs[inside], fr.t)
                gchi = rf.grad_chi(xs[inside], zs[inside], fr.t)
                # sigma : grad(chi e_i) = sum_j sigma_ij d_j chi
                lhs += w_t * np.einsum(
                    "p,pj->", fr.vol[inside], sig[inside][:, comp, :] * gchi
                )
                body = fr.rho[inside] * (g_vec[comp] - fr.a[inside][:, comp])
                if fr.extra_force is not None:
                    body = body + fr.extra_force[inside][:, comp]
                rhs += w_t * float(np.sum(fr.vol[inside] * body * chi))
                scale += w_t * float(
                    np.sum(fr.vol[inside] * fr.rho[inside] * G_MAG * np.abs(chi))
                )
            if scale > 0.0:
                full_errors.append(abs(lhs - rhs) / scale)

        # strictly admissible bump row, pressure-free identity
        tf = BumpTestFunction(*patch)
        idx, tw = _row_frames(times, tf.t_window)
        if idx is None:
            continue
        lhs = rhs = 0.0
        scale = 0.0
        for w_t, fi in zip(tw, idx):
            fr, sig = frames[fi], sigma_frames[fi]
            keep = fr.mask if fr.mask is not None else np.ones(len(fr.x), bool)
            xs, zs = fr.x[:, 0], fr.x[:, 1]
            inside = tf.in_support(xs, zs, fr.t) & keep
            if not np.any(inside):
                continue
            x_, z_ = xs[inside], zs[inside]
            Dw = tf.D_w(x_, z_, fr.t)
            w_vec = tf.w(x_, z_, fr.t)
            tau = sig[inside] + fr.p[inside][:, None, None] * np.eye(2)[None, :, :]
            lhs += w_t * np.einsum("p,pij,pij->", fr.vol[inside], tau, Dw)
            body = fr.rho[inside][:, None] * (g_vec[None, :] - fr.a[inside])
            if fr.extra_force is not None:
                body = body + fr.extra_force[inside]
            rhs += w_t * np.einsum("p,pi,pi->", fr.vol[inside], body, w_vec)
            scale += w_t * float(
                np.sum(
                    fr.vol[inside] * fr.rho[inside] * G_MAG
                    * np.linalg.norm(w_vec, axis=1)
                )
            )
        if scale > 0.0:
            adm_errors.append(abs(lhs - rhs) / scale)

    return ClosureReport(rel_error_full=full_errors, rel_error_admissible=adm_errors)
