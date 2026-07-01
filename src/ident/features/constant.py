"""Mode C dictionary: phi_1 = 1, K = 1."""

from __future__ import annotations

from typing import Any

import numpy as np

from ident.features.base import Dictionary


class ConstantDict(Dictionary):
    """mu(I) = theta_1, rate independent."""

    @property
    def K(self) -> int:
        return 1

    def phi(self, I: np.ndarray) -> np.ndarray:
        I = np.atleast_1d(np.asarray(I, dtype=float))
        return np.ones((I.shape[0], 1))

    def dphi_dI(self, I: np.ndarray) -> np.ndarray:
        I = np.atleast_1d(np.asarray(I, dtype=float))
        return np.zeros((I.shape[0], 1))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "C",
            "support": (0.0, np.inf),
            "nonnegativity": [False],
        }
