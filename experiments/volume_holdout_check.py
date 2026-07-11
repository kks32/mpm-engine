"""Cross-volume held-out test for the real-data pipeline.

realdata_pipeline.run() identifies a law from each volume's own video+force and re-simulates
it at the same volume, so its 1x/1.5x numbers are per-volume self-consistency, not a held-out
generalization (the report caption overstated this). This evaluates the full 2x2 matrix: each
recovered law (identified on 1x, identified on 1.5x) re-simulated at both volumes. The diagonal
reproduces the self-rollout, and the off-diagonal measures transfer between volumes. The
re-simulation uses the warp-mpm kernel without CoTracker.

Run:  python experiments/volume_holdout_check.py
"""
from __future__ import annotations

import numpy as np

import realdata_pipeline as R

LAWS = {"1x": (276.7, 16.1), "1.5x": (101.6, 59.7)}   # recovered (tau_y, eta) from identify.json
SCALES = {"1x": 1.0, "1.5x": 1.5}


def run():
    # truth force series at each volume (computed once)
    truth = {}
    for v, sc in SCALES.items():
        st, F = R._slab_force_series(sc, *R.TRUTH)
        truth[v] = (st, F)
        print(f"truth @ {v} done ({len(st)} frames)")

    def rel(eval_vol, law):
        st, Ft = truth[eval_vol]
        _, Fr = R._slab_force_series(SCALES[eval_vol], *law)
        w = st > 5
        return float(np.linalg.norm((Fr - Ft)[w]) / max(np.linalg.norm(Ft[w]), 1e-9))

    print(f"\n  truth law = {R.TRUTH} (tau_y, eta)")
    print("  cross-volume force-rollout error (rows: law identified on; cols: evaluated at):")
    print(f"  {'':16s} eval@1x    eval@1.5x")
    M = {}
    for ident_vol, law in LAWS.items():
        row = {ev: rel(ev, law) for ev in SCALES}
        M[ident_vol] = row
        print(f"  id@{ident_vol:4s} {law!s:14s} "
              f"{row['1x']*100:6.1f}%    {row['1.5x']*100:6.1f}%")
    print("\n  diagonal = self-consistency (what the report currently calls the result);")
    print("  off-diagonal = TRUE held-out generalization (train one volume, predict the other):")
    print(f"    train 1x   -> predict 1.5x : {M['1x']['1.5x']*100:.1f}%")
    print(f"    train 1.5x -> predict 1x   : {M['1.5x']['1x']*100:.1f}%")
    return M


if __name__ == "__main__":
    run()
