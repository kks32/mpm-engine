"""Joint multi-aspect-ratio divergence-free inversion for G3.

A single collapse, identified by the divergence-free continuum weak form on a
reconstructed field, is patch-scale biased (mu rises with patch radius and
crosses the truth at an r/H-dependent scale). The joint solve stacks rows from collapses at
different aspect ratios (different column height H, hence different r/H and
different I coverage) into one linear system, so a single mu(I) must satisfy
all geometries at once, breaking the per-geometry scale degeneracy. Combined
with multi-scale patches (full/half/offset per patch) and I-stratified
placement.

The fields are reconstructed (streamfunction B-spline) so this works on
perceived or tracked kinematics. This implementation is first checked on
particle-reconstructed fields with simulator pressure to separate conditioning from
tracking noise. Pure ident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.conventions import EPS_GAMMA_DEFAULT
from ident.features.base import Dictionary
from ident.features.constant import ConstantDict
from ident.gates.g1_oracle import stratify_patches
from ident.gates.g3_field import _reconstruct_frames
from ident.io.schema import load_dump
from ident.solve.ridge import ridge_solve
from ident.weakform.assembly import assemble_system


class _Shim:
    pass


def _assemble_one(dump_path, dictionary, radii_cells, pressure_mode, n_time=4):
    """Reconstruct fields for one collapse, assemble multi-scale weak-form rows."""
    dump = load_dump(dump_path)
    frames, bundle, dx = _reconstruct_frames(dump, pressure_mode=pressure_mode)
    if not frames:
        return None
    shim = _Shim(); shim.frames = frames
    shim.times = np.array([f.t for f in frames])
    r_t = 30.0 * dump.meta.frame_dt
    A_parts, b_parts = [], []
    for rc in radii_cells:
        rows, _ = stratify_patches(shim, rc * dx, rc * dx, r_t,
                                   n_time=n_time, min_particles=40)
        if not rows:
            continue
        sysm = assemble_system(frames, rows, dictionary, EPS_GAMMA_DEFAULT)
        A_parts.append(sysm.A); b_parts.append(sysm.b_tw)  # time-weak (perception)
    if not A_parts:
        return None
    A = np.vstack(A_parts); b = np.concatenate(b_parts)
    keep = np.linalg.norm(A, axis=1) > 1e-12 * np.linalg.norm(A, axis=1).max()
    return A[keep], b[keep], bundle.I_observed


def joint_invert(dump_paths, dictionary: Dictionary | None = None,
                 radii_cells=(4, 6, 8, 10, 12), pressure_mode="true", lam=1e-8):
    dic = dictionary if dictionary is not None else ConstantDict()
    per_aspect = {}
    A_all, b_all, A_norm = [], [], []
    for dp in dump_paths:
        res = _assemble_one(dp, dic, radii_cells, pressure_mode)
        if res is None:
            continue
        A, b, Iobs = res
        # per-aspect normalization: scale rows so each collapse contributes
        # equally to the joint normal equations (otherwise the largest /
        # most-energetic collapse dominates and reimposes its own scale bias).
        s = np.linalg.norm(A) + 1e-30
        A_all.append(A); b_all.append(b); A_norm.append((A / s, b / s))
        if dic.K == 1:
            a = A[:, 0]
            per_aspect[Path(dp).stem] = float((a @ b) / (a @ a))
        else:
            per_aspect[Path(dp).stem] = ridge_solve(A, b, lam=lam).theta.tolist()

    # naive (row-count weighted) joint
    Aj = np.vstack(A_all); bj = np.concatenate(b_all)
    # equal-weighted joint
    Ane = np.vstack([a for a, _ in A_norm]); bne = np.concatenate([b for _, b in A_norm])
    if dic.K == 1:
        joint_naive = float((Aj[:, 0] @ bj) / (Aj[:, 0] @ Aj[:, 0]))
        joint_eq = float((Ane[:, 0] @ bne) / (Ane[:, 0] @ Ane[:, 0]))
    else:
        joint_naive = ridge_solve(Aj, bj, lam=lam).theta.tolist()
        joint_eq = ridge_solve(Ane, bne, lam=lam).theta.tolist()
    return {"joint_equal_weighted": joint_eq, "joint_naive": joint_naive,
            "per_aspect": per_aspect, "n_rows_joint": int(Aj.shape[0])}


if __name__ == "__main__":
    import sys
    dumps = sys.argv[1:] or [
        "out/dumps/column_constant_a1.npz",
        "out/dumps/column_constant_a2.npz",
        "out/dumps/column_constant_a4.npz",
    ]
    dumps = [d for d in dumps if Path(d).exists()]
    res = joint_invert(dumps, ConstantDict())
    print(json.dumps(res, indent=2, default=float))
    print("truth mu = 0.38")
