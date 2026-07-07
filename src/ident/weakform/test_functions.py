"""Divergence-free space-time test functions (MATH_REFERENCE.md Section 3).

Generator along the out-of-plane axis: w = curl(eta e_y) with

  eta(x, z, t) = r_x r_z B(u) B(s) B(q)
  u = (x - x_c) / r_x,  s = (z - z_c) / r_z,  q = (t - t_c) / r_t
  B(zeta)   = (1 - zeta^2)^3      for |zeta| < 1, else 0
  B'(zeta)  = -6 zeta (1 - zeta^2)^2
  B''(zeta) = 6 (1 - zeta^2)(5 zeta^2 - 1)

Components and gradient entries are implemented verbatim from the document;
do not re-derive them. div w = 0 holds identically by construction and is
checked to machine precision in tests/test_test_functions.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bump(zeta: np.ndarray) -> np.ndarray:
    """B(zeta) = (1 - zeta^2)^3 inside |zeta| < 1, else 0."""
    zeta = np.asarray(zeta, dtype=float)
    core = 1.0 - zeta * zeta
    return np.where(np.abs(zeta) < 1.0, core**3, 0.0)


def bump_d1(zeta: np.ndarray) -> np.ndarray:
    """B'(zeta) = -6 zeta (1 - zeta^2)^2 inside |zeta| < 1, else 0."""
    zeta = np.asarray(zeta, dtype=float)
    core = 1.0 - zeta * zeta
    return np.where(np.abs(zeta) < 1.0, -6.0 * zeta * core**2, 0.0)


def bump_d2(zeta: np.ndarray) -> np.ndarray:
    """B''(zeta) = 6 (1 - zeta^2)(5 zeta^2 - 1) inside |zeta| < 1, else 0."""
    zeta = np.asarray(zeta, dtype=float)
    core = 1.0 - zeta * zeta
    return np.where(np.abs(zeta) < 1.0, 6.0 * core * (5.0 * zeta * zeta - 1.0), 0.0)


@dataclass(frozen=True)
class BumpTestFunction:
    """One admissible test field on a space-time patch.

    Spatial compact support |u|, |s| < 1; temporal factor B(q) serves the
    time-weak form. All evaluators broadcast over input arrays.
    """

    x_c: float
    z_c: float
    t_c: float
    r_x: float
    r_z: float
    r_t: float

    def _usq(self, x: np.ndarray, z: np.ndarray, t: np.ndarray):
        u = (np.asarray(x, dtype=float) - self.x_c) / self.r_x
        s = (np.asarray(z, dtype=float) - self.z_c) / self.r_z
        q = (np.asarray(t, dtype=float) - self.t_c) / self.r_t
        return u, s, q

    def eta(self, x: np.ndarray, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        u, s, q = self._usq(x, z, t)
        return self.r_x * self.r_z * bump(u) * bump(s) * bump(q)

    def w(self, x: np.ndarray, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        """w = (w_x, w_z), shape (..., 2).

        w_x = - r_x B(u) B'(s) B(q)
        w_z = + r_z B'(u) B(s) B(q)
        """
        u, s, q = self._usq(x, z, t)
        Bq = bump(q)
        wx = -self.r_x * bump(u) * bump_d1(s) * Bq
        wz = +self.r_z * bump_d1(u) * bump(s) * Bq
        return np.stack([wx, wz], axis=-1)

    def grad_w(self, x: np.ndarray, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        """grad w with entries [i, j] = d w_i / d x_j, shape (..., 2, 2).

        d w_x / d x = - B'(u) B'(s) B(q)
        d w_x / d z = - (r_x / r_z) B(u) B''(s) B(q)
        d w_z / d x = + (r_z / r_x) B''(u) B(s) B(q)
        d w_z / d z = + B'(u) B'(s) B(q)
        """
        u, s, q = self._usq(x, z, t)
        Bq = bump(q)
        dwx_dx = -bump_d1(u) * bump_d1(s) * Bq
        dwx_dz = -(self.r_x / self.r_z) * bump(u) * bump_d2(s) * Bq
        dwz_dx = +(self.r_z / self.r_x) * bump_d2(u) * bump(s) * Bq
        dwz_dz = +bump_d1(u) * bump_d1(s) * Bq
        g = np.stack(
            [
                np.stack([dwx_dx, dwx_dz], axis=-1),
                np.stack([dwz_dx, dwz_dz], axis=-1),
            ],
            axis=-2,
        )
        return g

    def D_w(self, x: np.ndarray, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        """D[w] = sym(grad w), shape (..., 2, 2)."""
        g = self.grad_w(x, z, t)
        return 0.5 * (g + np.swapaxes(g, -1, -2))

    def div_w(self, x: np.ndarray, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Trace of grad w; identically zero, kept for the machine-precision test."""
        g = self.grad_w(x, z, t)
        return g[..., 0, 0] + g[..., 1, 1]

    def dw_dt(self, x: np.ndarray, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Time derivative of w, shape (..., 2); only B(q) carries time."""
        u, s, q = self._usq(x, z, t)
        dBq = bump_d1(q) / self.r_t
        wx_t = -self.r_x * bump(u) * bump_d1(s) * dBq
        wz_t = +self.r_z * bump_d1(u) * bump(s) * dBq
        return np.stack([wx_t, wz_t], axis=-1)

    def in_support(self, x: np.ndarray, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        u, s, q = self._usq(x, z, t)
        return (np.abs(u) < 1.0) & (np.abs(s) < 1.0) & (np.abs(q) < 1.0)

    @property
    def t_window(self) -> tuple[float, float]:
        return (self.t_c - self.r_t, self.t_c + self.r_t)


def patch_rows(
    x_c: float,
    z_c: float,
    t_c: float,
    r_x: float,
    r_z: float,
    r_t: float,
    offset_frac: float = 0.25,
) -> list[BumpTestFunction]:
    """The 3 rows per patch: full scale, half scale centered, offset full-scale copy."""
    full = BumpTestFunction(x_c, z_c, t_c, r_x, r_z, r_t)
    half = BumpTestFunction(x_c, z_c, t_c, 0.5 * r_x, 0.5 * r_z, 0.5 * r_t)
    offset = BumpTestFunction(
        x_c + offset_frac * r_x, z_c + offset_frac * r_z, t_c, r_x, r_z, r_t
    )
    return [full, half, offset]
