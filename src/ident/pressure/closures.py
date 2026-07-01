"""P0 and P1 pressure closures (MATH_REFERENCE.md Section 5).

Signs are never written here; they live in common/conventions.py helpers:

  P1: p(x, z, t) = INT from z to h(x, t) of p1_integrand(rho, a_z) dz'
  P0: p(x, z, t) = hydrostatic_pressure_p0(rho, h, z)  (the a_z = 0 limit)

Column primitives operate on samples of one vertical column; the particle
field versions bin particles into x columns per frame.
"""

from __future__ import annotations

import numpy as np

from common.conventions import hydrostatic_pressure_p0, p1_integrand


def p0_column(z: np.ndarray, h: float, rho: np.ndarray | float) -> np.ndarray:
    """P0 at heights z below the free surface h."""
    return hydrostatic_pressure_p0(rho, h, np.asarray(z, dtype=float))


def p1_column(z: np.ndarray, a_z: np.ndarray, h: float, rho: np.ndarray | float) -> np.ndarray:
    """P1 at sample heights z of one column with vertical accelerations a_z.

    Integrates the conventions integrand downward from the free surface where
    p(h) = 0, by trapezoid on the sorted samples augmented with the surface
    point (a_z at the surface taken from the topmost sample).
    """
    z = np.asarray(z, dtype=float)
    a_z = np.asarray(a_z, dtype=float)
    rho_arr = np.broadcast_to(np.asarray(rho, dtype=float), z.shape).astype(float)
    order = np.argsort(z)
    zs = z[order]
    fs = p1_integrand(rho_arr[order], a_z[order])

    # append the free surface node
    z_aug = np.append(zs, h)
    f_aug = np.append(fs, fs[-1] if fs.size else 0.0)

    # cumulative trapezoid from the top: p(z_i) = INT_{z_i}^{h} f dz'
    p_aug = np.zeros_like(z_aug)
    for i in range(len(z_aug) - 2, -1, -1):
        dz = z_aug[i + 1] - z_aug[i]
        p_aug[i] = p_aug[i + 1] + 0.5 * (f_aug[i] + f_aug[i + 1]) * dz

    out = np.empty_like(z)
    out[order] = p_aug[:-1]
    return out


def surface_height_per_bin(
    x: np.ndarray, z: np.ndarray, bin_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Free-surface height h per x bin as the max particle height in the bin.

    Returns (bin_index_per_particle, bin_centers, h_per_bin).
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    if x.size == 0:
        return np.empty(0, dtype=int), np.empty(0), np.empty(0)
    x0 = x.min()
    bin_idx = np.floor((x - x0) / bin_width).astype(int)
    n_bins = bin_idx.max() + 1
    h = np.full(n_bins, -np.inf)
    np.maximum.at(h, bin_idx, z)
    centers = x0 + (np.arange(n_bins) + 0.5) * bin_width
    return bin_idx, centers, h


def p0_particles(
    x: np.ndarray, z: np.ndarray, rho: np.ndarray, bin_width: float
) -> np.ndarray:
    """P0 closure per particle, surface height from per-bin max z."""
    bin_idx, _, h = surface_height_per_bin(x, z, bin_width)
    return hydrostatic_pressure_p0(rho, h[bin_idx], z)


def p1_particles(
    x: np.ndarray,
    z: np.ndarray,
    a_z: np.ndarray,
    rho: np.ndarray,
    bin_width: float,
) -> np.ndarray:
    """P1 closure per particle, integrated within each x bin."""
    bin_idx, _, h = surface_height_per_bin(x, z, bin_width)
    out = np.zeros_like(np.asarray(z, dtype=float))
    rho_arr = np.broadcast_to(np.asarray(rho, dtype=float), out.shape)
    for b in np.unique(bin_idx):
        sel = bin_idx == b
        out[sel] = p1_column(z[sel], a_z[sel], float(h[b]), rho_arr[sel])
    return out
