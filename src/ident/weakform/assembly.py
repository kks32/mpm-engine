"""Particle-quadrature assembly of the weak-form system.

Assembly formulas (MATH_REFERENCE.md Sections 2 to 5, do not re-derive):

  A[j, k] = sum_t sum_p V_p dt phi_k(I_p) p_p (2 D_p / |gamma_dot|_eps,p) : D[w_j](x_p)
  b[j]    = sum_t sum_p V_p dt rho (g - a_p) . w_j(x_p)

Time-weak load (M2), no acceleration data:

  INT_Q rho a . w = - INT_Q rho [ v . dw/dt + (v (x) v) : grad w ]

so  b_tw[j] = sum_t sum_p V_p dt rho [ g . w + v . dw/dt + (v (x) v) : grad w ].

g is the in-plane gravity VECTOR. Quadrature: particle sums in space,
trapezoid in time over each row's temporal support. Pressure and I are DATA.
An optional extra body force channel exists for the G0 manufactured source
and is added to both load variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.conventions import (
    EPS_GAMMA_DEFAULT,
    G_MAG,
    equivalent_shear_rate,
    gravity_vector_inplane,
)
from ident.features.base import Dictionary
from ident.weakform.test_functions import BumpTestFunction


@dataclass
class FrameData:
    """In-plane particle data of one frame. All arrays share leading dim P."""

    t: float
    x: np.ndarray          # (P, 2) positions (x, z)
    v: np.ndarray          # (P, 2) velocities
    D: np.ndarray          # (P, 2, 2) rate of deformation sym(grad v)
    a: np.ndarray          # (P, 2) material acceleration
    p: np.ndarray          # (P,) pressure, DATA to the identifier
    I: np.ndarray          # (P,) inertial number, DATA
    vol: np.ndarray        # (P,) particle volume (quadrature weight)
    rho: np.ndarray        # (P,) bulk density
    mask: np.ndarray | None = None        # (P,) row-inclusion mask
    extra_force: np.ndarray | None = None  # (P, 2) manufactured body force (G0 only)


@dataclass
class WeakformSystem:
    A: np.ndarray        # (J, K)
    b_acc: np.ndarray    # (J,) acceleration-form load (M1)
    b_tw: np.ndarray     # (J,) time-weak load (M2)
    row_scale: np.ndarray  # (J,) inertia-gravity magnitude per row
    rows: list[BumpTestFunction] = field(default_factory=list)

    def scaled(self) -> "WeakformSystem":
        """Rows divided by their inertia-gravity magnitude (Section 8.1)."""
        s = np.where(self.row_scale > 0.0, self.row_scale, 1.0)
        return WeakformSystem(
            A=self.A / s[:, None],
            b_acc=self.b_acc / s,
            b_tw=self.b_tw / s,
            row_scale=np.ones_like(s),
            rows=self.rows,
        )


def _trapezoid_weights(times: np.ndarray) -> np.ndarray:
    """Composite trapezoid weights for samples at the given times."""
    tw = np.zeros_like(times)
    if times.shape[0] == 1:
        return tw  # a single frame carries no temporal measure
    dt = np.diff(times)
    tw[:-1] += 0.5 * dt
    tw[1:] += 0.5 * dt
    return tw


def assemble_system(
    frames: list[FrameData],
    rows: list[BumpTestFunction],
    dictionary: Dictionary,
    eps_gamma: float = EPS_GAMMA_DEFAULT,
    g_vec: np.ndarray | None = None,
) -> WeakformSystem:
    times = np.array([f.t for f in frames])
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("frames must be strictly increasing in time")
    # default is the pinned vertical gravity (0, -G_MAG); inclined-plane scenes
    # pass the tilted in-plane gravity vector recorded in the dump.
    g_vec = gravity_vector_inplane() if g_vec is None else np.asarray(g_vec, dtype=float)

    J, K = len(rows), dictionary.K
    A = np.zeros((J, K))
    b_acc = np.zeros(J)
    b_tw = np.zeros(J)
    row_scale = np.zeros(J)

    for j, tf in enumerate(rows):
        t_lo, t_hi = tf.t_window
        sel = np.where((times > t_lo) & (times < t_hi))[0]
        if sel.size < 2:
            continue  # row has no temporal quadrature support
        # extend by the boundary frames where B(q) vanishes when available,
        # so the trapezoid sees the support edges
        lo = max(sel[0] - 1, 0)
        hi = min(sel[-1] + 1, len(frames) - 1)
        idx = np.arange(lo, hi + 1)
        tw = _trapezoid_weights(times[idx])

        for w_t, fi in zip(tw, idx):
            fr = frames[fi]
            if fr.mask is not None:
                m = fr.mask
                if not np.any(m):
                    continue
                x, v, D, a = fr.x[m], fr.v[m], fr.D[m], fr.a[m]
                p, I, vol, rho = fr.p[m], fr.I[m], fr.vol[m], fr.rho[m]
                f_extra = fr.extra_force[m] if fr.extra_force is not None else None
            else:
                x, v, D, a = fr.x, fr.v, fr.D, fr.a
                p, I, vol, rho = fr.p, fr.I, fr.vol, fr.rho
                f_extra = fr.extra_force

            xs, zs, ts = x[:, 0], x[:, 1], fr.t
            inside = tf.in_support(xs, zs, ts)
            if not np.any(inside):
                continue
            xs, zs = xs[inside], zs[inside]
            v_, D_, a_ = v[inside], D[inside], a[inside]
            p_, I_, vol_, rho_ = p[inside], I[inside], vol[inside], rho[inside]

            w_vec = tf.w(xs, zs, ts)        # (P, 2)
            grad_w = tf.grad_w(xs, zs, ts)  # (P, 2, 2)
            Dw = 0.5 * (grad_w + np.swapaxes(grad_w, -1, -2))
            dw_dt = tf.dw_dt(xs, zs, ts)    # (P, 2)

            gd = equivalent_shear_rate(D_, eps_gamma)        # (P,)
            flow_dir = 2.0 * D_ / gd[:, None, None]          # (P, 2, 2)
            contr = np.einsum("pij,pij->p", flow_dir, Dw)    # (P,)
            Phi = dictionary.phi(I_)                          # (P, K)

            A[j] += w_t * np.einsum("p,pk->k", vol_ * p_ * contr, Phi)

            body = rho_[:, None] * (g_vec[None, :] - a_)
            if f_extra is not None:
                body = body + f_extra[inside]
            b_acc[j] += w_t * np.einsum("p,pi,pi->", vol_, body, w_vec)

            vv = np.einsum("pi,pj->pij", v_, v_)
            tw_term = (
                np.einsum("pi,pi->p", v_, dw_dt)
                + np.einsum("pij,pij->p", vv, grad_w)
            )
            body_tw = rho_[:, None] * g_vec[None, :]
            if f_extra is not None:
                body_tw = body_tw + f_extra[inside]
            b_tw[j] += w_t * (
                np.einsum("p,pi,pi->", vol_, body_tw, w_vec)
                + np.einsum("p,p,p->", vol_, rho_, tw_term)
            )

            row_scale[j] += w_t * np.einsum(
                "p,p->", vol_ * rho_ * G_MAG, np.linalg.norm(w_vec, axis=1)
            )

    return WeakformSystem(A=A, b_acc=b_acc, b_tw=b_tw, row_scale=row_scale, rows=list(rows))
