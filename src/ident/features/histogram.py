"""Piecewise-constant (histogram) dictionary on log10 I.

A non-shipping helper used to build the per-bin design matrix C[j,b] such that
A[j,k] = sum_p coeff_{j,p} phi_k(I_p) becomes A_j = sum_b C[j,b] mu(I_b) when
phi_b is the indicator of I-bin b. This bridges the weak-form assembly to an
arbitrary pointwise mu(I) representation (e.g. an NN constitutive function in
the NN-EUCLID parallel track): the SAME assembly produces C, then either a
linear fit (mu_b = sum_k theta_k psi_k(I_b)) or a nonlinear fit (mu_b = NN(I_b))
minimizes ||C mu - b||. Each is linear vs nonlinear in its OWN parameters; the
weak form itself is always linear in the pointwise mu values.

Derivatives are zero a.e. (constant within a bin); this dictionary is only used
to assemble C (which calls phi and K), never the pressure sensitivity operator
or the monotonicity rows, so the zero derivatives are correct for its use.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ident.features.base import Dictionary


class HistogramDict(Dictionary):
    def __init__(self, s_edges: np.ndarray) -> None:
        """s_edges: (M+1,) ascending log10 I bin edges. K = M bins."""
        s_edges = np.asarray(s_edges, dtype=float)
        if s_edges.ndim != 1 or s_edges.size < 2 or np.any(np.diff(s_edges) <= 0):
            raise ValueError("s_edges must be ascending with >= 2 entries")
        self.s_edges = s_edges
        self.s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        self._K = s_edges.size - 1

    @property
    def K(self) -> int:
        return self._K

    @property
    def bin_centers_I(self) -> np.ndarray:
        return 10.0 ** self.s_centers

    def phi(self, I: np.ndarray) -> np.ndarray:
        I = np.atleast_1d(np.asarray(I, dtype=float))
        with np.errstate(divide="ignore"):
            s = np.log10(np.maximum(I, np.finfo(float).tiny))
        # clamp into the edge bins so out-of-range I still contributes
        idx = np.clip(np.digitize(s, self.s_edges) - 1, 0, self._K - 1)
        out = np.zeros((I.size, self._K))
        out[np.arange(I.size), idx] = 1.0
        return out

    def dphi_dI(self, I: np.ndarray) -> np.ndarray:
        I = np.atleast_1d(np.asarray(I, dtype=float))
        return np.zeros((I.size, self._K))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "histogram",
            "support": (float(10.0 ** self.s_edges[0]), float(10.0 ** self.s_edges[-1])),
            "nonnegativity": [True] * self._K,
            "n_bins": self._K,
        }
