"""Fisher-information design scores and greedy active probe selection.

The identification covariance is $\\sigma^2 (A^T A)^{-1}$. Before an experiment runs,
the design matrix $A(a)$ of a candidate action $a$ sets the information
$M(a)=A(a)^T A(a)/\\sigma^2$. This module scores candidate actions and greedily
selects the next one, turning the covariance from a diagnostic into an action
selector:

  D-optimal:  maximize  log det M             (overall information)
  A-optimal:  minimize  trace M^{-1}           (mean variance)
  c-optimal:  minimize  c^T M^{-1} c            (variance of the query-relevant c^T theta)

Selection is a discrete greedy search over a pool of candidate probes; no
simulator gradient is taken. Each probe contributes rows to A, so running it
adds its outer product to M. The scores are closed form in M.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg


def d_score(M: np.ndarray) -> float:
    """log det M (information volume); -inf if singular."""
    sign, logdet = np.linalg.slogdet(M)
    return float(logdet) if sign > 0 else -np.inf


def a_score(M: np.ndarray) -> float:
    """trace M^{-1} (sum of parameter variances)."""
    return float(np.trace(linalg.inv(M)))


def c_var(M: np.ndarray, c: np.ndarray) -> float:
    """c^T M^{-1} c, the variance of the query-relevant combination c^T theta."""
    c = np.asarray(c, dtype=float)
    return float(c @ linalg.solve(M, c, assume_a="pos"))


def greedy_select(pool, budget, objective="D", c=None, lam=1e-9):
    """Greedily pick `budget` probes from `pool` to optimize the design objective.

    pool: list of (label, row) where row is the (K,) effective design row a probe
        contributes (its outer product is added to M when the probe is run).
        Probes may be re-selected (running again adds information).
    objective: 'D' (maximize log det), 'A' (minimize trace M^{-1}),
        or 'c' (minimize c^T M^{-1} c, requires c).
    Returns (chosen_indices, trajectory) where trajectory[k] holds the variances
        after k+1 picks.
    """
    rows = [np.asarray(r, dtype=float) for _, r in pool]
    K = rows[0].shape[0]
    M = lam * np.eye(K)
    chosen, traj = [], []
    for _ in range(budget):
        best_i, best_val = None, None
        for i, r in enumerate(rows):
            Mi = M + np.outer(r, r)
            if objective == "D":
                val = d_score(Mi)
            elif objective == "A":
                val = -a_score(Mi)
            elif objective == "c":
                val = -c_var(Mi, c)
            else:
                raise ValueError(objective)
            if best_val is None or val > best_val:
                best_val, best_i = val, i
        M = M + np.outer(rows[best_i], rows[best_i])
        chosen.append(best_i)
        rec = {"logdet": d_score(M), "trace_inv": a_score(M)}
        if c is not None:
            rec["c_var"] = c_var(M, c)
        traj.append(rec)
    return chosen, traj
