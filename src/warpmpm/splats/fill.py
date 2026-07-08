"""Interior filling and per-particle volumes, reimplemented from the PhysGaussian
algorithm without Taichi.

A thin shell of surface splats is not a solid; the material point method needs particles
throughout the body. fill_interior finds empty interior cells by ray casting on an
opacity density grid and drops filler particles into them. particle_volumes assigns each
particle the cell volume shared among the particles in its cell.
"""
from __future__ import annotations

import warnings

import numpy as np

# axis-direction names to (axis, sign)
_DIRS = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1),
         "+z": (2, 1), "-z": (2, -1)}


def _cell_index(pos: np.ndarray, grid_dx: float, grid_n: int) -> np.ndarray:
    """Cell indices of points, clipped into the grid."""
    c = np.floor(np.asarray(pos, dtype=np.float64) / grid_dx).astype(np.int64)
    return np.clip(c, 0, grid_n - 1)


def _shift_toward_lower(a: np.ndarray, axis: int, fill) -> np.ndarray:
    """out[i] = a[i + 1] along axis; the last slice is set to fill."""
    a = np.moveaxis(a, axis, 0)
    out = np.empty_like(a)
    out[:-1] = a[1:]
    out[-1] = fill
    return np.moveaxis(out, 0, axis)


def _any_ahead(occ: np.ndarray, axis: int, sign: int) -> np.ndarray:
    """For each cell, whether any occupied cell lies strictly ahead along (axis, sign)."""
    if sign == 1:
        incl = np.flip(np.maximum.accumulate(np.flip(occ, axis), axis), axis)  # any at >= i
        return _shift_toward_lower(incl, axis, False)                          # any at > i
    incl = np.maximum.accumulate(occ, axis)                                    # any at <= i
    # shift toward higher index: behind[i] = incl[i-1]
    a = np.moveaxis(incl, axis, 0)
    out = np.empty_like(a)
    out[1:] = a[:-1]
    out[0] = False
    return np.moveaxis(out, 0, axis)


def _runs_ahead(occ: np.ndarray, axis: int, sign: int) -> np.ndarray:
    """Number of maximal occupied runs strictly ahead of each cell along (axis, sign).
    This is the even-odd ray-crossing count used for the point-in-solid parity test."""
    if sign == -1:
        return np.flip(_runs_ahead(np.flip(occ, axis), axis, 1), axis)

    a = np.moveaxis(occ, axis, 0).astype(bool)
    a_prev = np.empty_like(a)
    a_prev[1:] = a[:-1]
    a_prev[0] = False
    edge = a & ~a_prev                                  # a run starts here
    straddle = a & a_prev                               # a run continues into here
    # suffix sum of edges, inclusive: E[i] = sum_{p >= i} edge[p]
    E = np.flip(np.cumsum(np.flip(edge.astype(np.int64), 0), 0), 0)
    runs_suffix = E + straddle.astype(np.int64)         # runs in a[i:]
    out = np.empty_like(runs_suffix)
    out[:-1] = runs_suffix[1:]                           # runs strictly ahead: from i+1
    out[-1] = 0
    return np.moveaxis(out, 0, axis)


def fill_interior(pos: np.ndarray, opacity: np.ndarray, cov6: np.ndarray, grid_n: int,
                  grid_dx: float, density_thres: float = 2.0, search_thres: float = 1.0,
                  max_particles_per_cell: int = 1, max_samples: int = 200_000,
                  exclude_dirs=(), boundary=None, seed: int = 0) -> np.ndarray:
    """Interior filler positions for a splat cloud.

    Splats each Gaussian's opacity into its cell to build a density grid, marks cells with
    density > density_thres occupied, and calls a cell interior when every ray direction
    (the six axes minus exclude_dirs, e.g. ("+z",) for an open-top object) hits an occupied
    cell and an even-odd crossing count (threshold search_thres) along a non-excluded
    direction is odd. Samples up to max_particles_per_cell jittered points per interior
    cell, capped at max_samples. Returns (M, 3). pos and cov6 are in the same space as
    grid_dx (sim space when driven by SplatScene). cov6 is accepted for API symmetry with
    the reference; the density here uses opacity mass per cell.
    """
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    opac = np.asarray(opacity, dtype=np.float64).reshape(-1)
    ci = _cell_index(pos, grid_dx, grid_n)

    density = np.zeros((grid_n, grid_n, grid_n), dtype=np.float64)
    np.add.at(density, (ci[:, 0], ci[:, 1], ci[:, 2]), opac)

    inside = np.ones((grid_n, grid_n, grid_n), dtype=bool)
    if boundary is not None:
        i_lo, i_hi, j_lo, j_hi, k_lo, k_hi = (int(b) for b in boundary)
        inside[:] = False
        inside[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi] = True

    occ_hit = (density > density_thres) & inside
    occ_cross = (density > search_thres) & inside
    empty = (~(density > density_thres)) & inside

    include = [d for d in _DIRS if d not in exclude_dirs]
    if not include:
        raise ValueError("exclude_dirs excludes all six directions")

    all_hit = np.ones((grid_n, grid_n, grid_n), dtype=bool)
    for d in include:
        axis, sign = _DIRS[d]
        all_hit &= _any_ahead(occ_hit, axis, sign)

    cast_axis, cast_sign = _DIRS[include[0]]
    parity_odd = (_runs_ahead(occ_cross, cast_axis, cast_sign) % 2) == 1

    interior = empty & all_hit & parity_odd
    cells = np.argwhere(interior)
    if cells.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    rng = np.random.default_rng(seed)
    reps = max(1, int(max_particles_per_cell))
    cells = np.repeat(cells, reps, axis=0)
    if cells.shape[0] > max_samples:
        warnings.warn(f"interior fill produced {cells.shape[0]} candidates, capping at "
                      f"{max_samples}", RuntimeWarning, stacklevel=2)
        cells = cells[rng.choice(cells.shape[0], max_samples, replace=False)]
    jitter = rng.uniform(0.0, 1.0, size=cells.shape)
    fillers = (cells.astype(np.float64) + jitter) * grid_dx
    return fillers.astype(np.float32)


def particle_volumes(pos: np.ndarray, grid_n: int, grid_dx: float,
                     uniform: bool = False) -> np.ndarray:
    """Per-particle volume: the cell volume divided by the particle count in that cell.
    uniform=True returns the mean volume everywhere (the reference uses this for sand)."""
    ci = _cell_index(pos, grid_dx, grid_n)
    count = np.zeros((grid_n, grid_n, grid_n), dtype=np.int64)
    np.add.at(count, (ci[:, 0], ci[:, 1], ci[:, 2]), 1)
    per_cell = count[ci[:, 0], ci[:, 1], ci[:, 2]]
    vol = (grid_dx ** 3) / np.clip(per_cell, 1, None)
    vol = vol.astype(np.float32)
    if uniform:
        return np.full(pos.shape[0], float(vol.mean()), dtype=np.float32)
    return vol
