"""World-to-sim similarity transform for placing a splat scene in the MPM grid.

The transform is an isotropic scale about a world center mapped to a sim center. Positions
translate and scale; covariances scale by s^2; the polar rotation is scale-and-translation
invariant, so R needs no transform.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class SimTransform:
    s: float
    world_center: np.ndarray
    sim_center: np.ndarray

    def to_sim(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float32)
        return ((pos - self.world_center) * self.s + self.sim_center).astype(np.float32)

    def from_sim(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float32)
        return ((pos - self.sim_center) / self.s + self.world_center).astype(np.float32)

    def to_sim_cov(self, cov6: np.ndarray) -> np.ndarray:
        return (np.asarray(cov6, dtype=np.float32) * (self.s * self.s)).astype(np.float32)

    def from_sim_cov(self, cov6: np.ndarray) -> np.ndarray:
        return (np.asarray(cov6, dtype=np.float32) / (self.s * self.s)).astype(np.float32)


def _max_sigma_min(cov6: np.ndarray) -> float:
    """Smallest per-splat principal standard deviation over the cloud (sqrt of the
    smallest covariance eigenvalue), used to flag under-resolved splats."""
    cov6 = np.asarray(cov6, dtype=np.float64)
    idx = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    sigma = np.zeros((cov6.shape[0], 3, 3))
    for c, (i, j) in enumerate(idx):
        sigma[:, i, j] = cov6[:, c]
        sigma[:, j, i] = cov6[:, c]
    evals = np.linalg.eigvalsh(sigma)
    return float(np.sqrt(np.clip(evals.min(), 0.0, None)))


def fit_to_grid(cloud, grid, extent_frac: float = 0.35, floor_cells: int = 4
                ) -> SimTransform:
    """Center the cloud in x, y at the grid center, scale so its largest extent equals
    extent_frac * grid_lim, and place its bottom floor_cells * dx above z = 0. Warns (does
    not raise) when the smallest splat sigma would map below dx / 4 (under-resolved)."""
    lo = cloud.pos.min(axis=0).astype(np.float64)
    hi = cloud.pos.max(axis=0).astype(np.float64)
    extent = float(np.max(hi - lo))
    if extent <= 0.0:
        raise ValueError("cloud has zero extent; cannot fit to grid")
    s = extent_frac * grid.grid_lim / extent

    world_center = 0.5 * (lo + hi)
    dx = grid.dx
    bottom_sim = floor_cells * dx
    half_height_sim = 0.5 * s * (hi[2] - lo[2])
    sim_center = np.array([0.5 * grid.grid_lim, 0.5 * grid.grid_lim,
                           bottom_sim + half_height_sim], dtype=np.float64)

    tf = SimTransform(s=float(s), world_center=world_center.astype(np.float32),
                      sim_center=sim_center.astype(np.float32))

    sigma_min_sim = s * _max_sigma_min(cloud.cov)
    if sigma_min_sim < 0.25 * dx:
        warnings.warn(
            f"smallest splat sigma maps to {sigma_min_sim:.2e} m, below dx/4 = "
            f"{0.25 * dx:.2e} m; the scene is under-resolved at n_grid={grid.n_grid} "
            "(raise n_grid or extent_frac).", RuntimeWarning, stacklevel=2)
    return tf
