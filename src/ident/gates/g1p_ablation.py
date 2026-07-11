"""G1P pressure-closure ablation: the four-curve figure of the project.

For one oracle dump, identify mu(I) three ways that differ only in the
pressure fed to the identifier:
  - true_p: the true MPM 3D stress-trace pressure (the oracle path)
  - P1:     the depth-integrated closure with measured a_z (default candidate)
  - P0:     the hydrostatic closure rho g (h - z)

Uses the grid-consistent (Bubnov-Galerkin) assembler. The flowing-at-yield
node set is fixed from the true pressure and reused for all three closures, so
the only thing that changes is the pressure channel (the multiplicative p
factor in A and b, and the inertial-number argument). This isolates the bias
introduced by the closures (MATH_REFERENCE.md Section 5).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.conventions import (
    EPS_GAMMA_DEFAULT,
    equivalent_shear_rate,
    inertial_number,
    pressure_from_cauchy_3d_trace,
    sym,
)
from ident.features.base import Dictionary
from ident.features.constant import ConstantDict
from ident.features.pouliquen_grid import PouliquenGridDict
from ident.gates.plotting import plot_mu_curves
from ident.io.schema import load_dump
from ident.masks.flowing import flowing_mask
from ident.pressure.closures import p0_particles, p1_particles
from ident.solve.ridge import ridge_solve
from ident.weakform.grid_assembly import assemble_grid_consistent

RESULTS_SCHEMA_VERSION = "g1p-2.0"
FLOW_FRAC_MIN = 0.90


def _closure_pressures(dump):
    """Per-frame (F,P) pressure fields for true, P1, P0, plus rho and a_z."""
    meta = dump.meta
    ax, az = meta.in_plane_axes
    d, rho_s = meta.grain_diameter, meta.rho_s
    bin_w = 4.0 * d
    cfg = meta.extra.get("config", {})
    if "grid_lim" in cfg and "n_grid" in cfg:
        bin_w = max(bin_w, 2.0 * cfg["grid_lim"] / cfg["n_grid"])

    x = dump.x[..., ax]
    z = dump.x[..., az]
    v_ip = dump.v[..., [ax, az]]
    rho = dump.mass[None, :] / np.maximum(dump.volume, 1e-30)
    frame_dt = meta.frame_dt
    F = meta.n_frames
    a_z = np.zeros((F, dump.meta.n_particles))
    if F >= 3:
        a_z[1:-1] = (v_ip[2:, :, 1] - v_ip[:-2, :, 1]) / (2.0 * frame_dt)
    a_z[0] = (v_ip[1, :, 1] - v_ip[0, :, 1]) / frame_dt
    a_z[-1] = (v_ip[-1, :, 1] - v_ip[-2, :, 1]) / frame_dt

    p_true = pressure_from_cauchy_3d_trace(dump.stress)
    p0 = np.zeros((F, dump.meta.n_particles))
    p1 = np.zeros((F, dump.meta.n_particles))
    for f in range(F):
        m = dump.active[f]
        if not np.any(m):
            continue
        p0[f, m] = p0_particles(x[f, m], z[f, m], rho[f, m], bin_w)
        p1[f, m] = p1_particles(x[f, m], z[f, m], a_z[f, m], rho[f, m], bin_w)
    return {"true_p": p_true, "P1": p1, "P0": p0}


def run_gate(dump_path, out_dir="out/g1p", dictionary=None, lam=1.0e-8):
    dump_path = Path(dump_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = load_dump(dump_path)
    meta = dump.meta
    dic = dictionary if dictionary is not None else ConstantDict()
    mode = dic.metadata["mode"]

    # fixed flowing-at-yield set from TRUE pressure
    ax, az = meta.in_plane_axes
    D_ip = sym(dump.L[:, :, [ax, az]][:, :, :, [ax, az]])
    gd = equivalent_shear_rate(D_ip, EPS_GAMMA_DEFAULT)
    p_true = pressure_from_cauchy_3d_trace(dump.stress)
    I_true = inertial_number(gd, p_true, meta.grain_diameter, meta.rho_s)
    flow_true = flowing_mask(gd, I_true)

    pressures = _closure_pressures(dump)
    I_obs = (0.0, 0.0)
    curves, recoveries = {}, {}
    for name, pfield in pressures.items():
        gs = assemble_grid_consistent(
            dump, dic, EPS_GAMMA_DEFAULT, flow_frac_min=FLOW_FRAC_MIN,
            pressure=pfield, flowing_override=flow_true,
        )
        if gs.n_rows == 0:
            continue
        res = ridge_solve(gs.A, gs.b, lam=lam)
        I_obs = gs.I_observed
        I_grid = np.logspace(np.log10(max(gs.I_observed[0], 1e-4)),
                             np.log10(max(gs.I_observed[1], 1e-3)), 100)
        curves[name] = dic.phi(I_grid) @ res.theta
        rec = {"theta_hat": res.theta.tolist(), "rows": gs.n_rows}
        if mode == "C" and meta.law == "constant":
            mu_s = meta.law_params["mu_s"]
            rec["mu_hat"] = float(res.theta[0])
            rec["mu_relative_error"] = abs(res.theta[0] - mu_s) / mu_s
        recoveries[name] = rec
        last_I_grid = I_grid

    # truth curve
    if mode == "C" and meta.law == "constant":
        curves["truth"] = np.full_like(last_I_grid, meta.law_params["mu_s"])
    elif meta.law == "pouliquen":
        lp = meta.law_params
        curves["truth"] = lp["mu_s"] + lp["delta_mu"] * last_I_grid / (last_I_grid + lp["I0"])

    fig = plot_mu_curves(
        last_I_grid, curves, out_dir / f"g1p_{mode}_fourcurve.png",
        observed_I=I_obs,
        title=f"G1P {mode}  a={meta.extra.get('aspect')}  pressure-closure ablation (grid-consistent)",
    )
    result = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "dump": str(dump_path), "mode": mode, "law": meta.law,
        "aspect": meta.extra.get("aspect"), "observed_I": list(I_obs),
        "flow_frac_min": FLOW_FRAC_MIN, "recoveries": recoveries,
        "paths_to_figures": [str(fig)],
    }
    with open(out_dir / "results.json", "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    return result


if __name__ == "__main__":
    import sys as _sys
    path = _sys.argv[1] if len(_sys.argv) > 1 else "out/dumps/column_constant_a2.npz"
    mode = _sys.argv[2] if len(_sys.argv) > 2 else "C"
    dic = ConstantDict() if mode == "C" else PouliquenGridDict()
    res = run_gate(path, out_dir=f"out/g1p_{mode}", dictionary=dic)
    print(json.dumps(res, indent=2, default=float))
