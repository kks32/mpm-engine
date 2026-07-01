"""Mode P dictionary: mu(I) = mu_s + sum_g c_g f_g(I), f_g(I) = I / (I + I_0g).

I_0g sits on a log grid in [1e-3, 1]. f_g(0) = 0 and f_g(inf) = 1, so mu_s
rides the constant feature (column 0) and sum_g c_g is the friction rise.
Nonnegativity c_g >= 0 applies to columns 1..K-1; L1 selection happens in the
solver, not here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ident.features.base import Dictionary

I0_GRID_MIN: float = 1.0e-3
I0_GRID_MAX: float = 1.0
N_I0_DEFAULT: int = 7


def default_I0_grid(n: int = N_I0_DEFAULT) -> np.ndarray:
    return np.logspace(np.log10(I0_GRID_MIN), np.log10(I0_GRID_MAX), n)


class PouliquenGridDict(Dictionary):
    """Column 0 is the constant feature; columns 1..n are f_g(I) = I/(I+I_0g)."""

    def __init__(self, I0_grid: np.ndarray | None = None) -> None:
        self.I0 = default_I0_grid() if I0_grid is None else np.asarray(I0_grid, dtype=float)
        if np.any(self.I0 <= 0.0):
            raise ValueError("I_0g must be positive")

    @property
    def K(self) -> int:
        return 1 + self.I0.shape[0]

    def phi(self, I: np.ndarray) -> np.ndarray:
        I = np.atleast_1d(np.asarray(I, dtype=float))
        out = np.empty((I.shape[0], self.K))
        out[:, 0] = 1.0
        # f_g(I) = I/(I+I0) written as 1/(1+I0/I): exact, and robust at the
        # limits f_g(0)=0, f_g(inf)=1 without inf/inf producing NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = self.I0[None, :] / I[:, None]
        out[:, 1:] = 1.0 / (1.0 + ratio)
        out[:, 1:] = np.where(I[:, None] <= 0.0, 0.0, out[:, 1:])
        return out

    def dphi_dI(self, I: np.ndarray) -> np.ndarray:
        I = np.atleast_1d(np.asarray(I, dtype=float))
        out = np.zeros((I.shape[0], self.K))
        out[:, 1:] = self.I0[None, :] / (I[:, None] + self.I0[None, :]) ** 2
        # f_g is clamped to a constant 0 for I <= 0 in phi(); its derivative must
        # therefore be 0 there too (FD consistency for the dictionary interface).
        out[:, 1:] = np.where(I[:, None] <= 0.0, 0.0, out[:, 1:])
        return out

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "P",
            "support": (0.0, np.inf),
            "nonnegativity": [False] + [True] * self.I0.shape[0],
            "I0_grid": self.I0.tolist(),
        }
