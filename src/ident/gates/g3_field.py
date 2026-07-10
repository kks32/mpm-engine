"""Field-based identification: the G3 rendered-video closure engine.

For each frame, reconstruct a smooth divergence-free velocity field from
scattered (x, v) (StreamFunctionField, MATH_REFERENCE 6.5) and a smooth
pressure field, then evaluate the weak form by DENSE regular quadrature on the
smooth fields (the G0 regime: smooth integrands + regular quadrature, which
avoids the particle-scatter bias). The divergence-free bump test functions
make the result insensitive to the pressure GRADIENT, so a pressure CLOSURE
(P1/P0) can stand in for the unavailable true pressure with only the clean
multiplicative bias of MATH_REFERENCE Section 5.

This engine is agnostic to where (x, v) come from: oracle particles (this
validation) or CoTracker3 tracks (G3 with real perception). Pure ident.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

from common.conventions import (
    EPS_GAMMA_DEFAULT,
    equivalent_shear_rate,
    inertial_number,
    pressure_from_cauchy_3d_trace,
)
from ident.features.constant import ConstantDict
from ident.gates.g1_oracle import stratify_patches
from ident.gates.plotting import plot_mu_curves
from ident.io.schema import load_dump
from ident.masks.flowing import flowing_mask
from ident.pressure.closures import p0_particles, p1_particles
from ident.solve.ridge import ridge_solve
from ident.weakform.assembly import FrameData, assemble_system
from ident.weakform.field_reconstruction import StreamFunctionField
from ident.weakform.from_dump import build_inplane_frames


def _reconstruct_frames(dump, pressure_mode="true", quad_dx_factor=1.0,
                        n_knots=(20, 10), lam_smooth=1e-4, frame_stride=2):
    """Build dense-quadrature FrameData from reconstructed fields per frame."""
    meta = dump.meta
    ax, az = meta.in_plane_axes
    cfg = meta.extra.get("config", {})
    dx = cfg["grid_lim"] / cfg["n_grid"]
    d, rho_s = meta.grain_diameter, meta.rho_s
    bundle = build_inplane_frames(dump)
    h = quad_dx_factor * dx

    # global flowing bbox over the admissible window
    xs_all, zs_all = [], []
    for fr in bundle.frames:
        if fr.mask is not None and np.any(fr.mask):
            xs_all.append(fr.x[fr.mask, 0]); zs_all.append(fr.x[fr.mask, 1])
    xc = np.concatenate(xs_all); zc = np.concatenate(zs_all)
    gx = np.arange(xc.min(), xc.max() + h, h)
    gz = np.arange(zc.min(), zc.max() + h, h)
    GX, GZ = np.meshgrid(gx, gz, indexing="ij")
    grid = np.stack([GX.ravel(), GZ.ravel()], axis=-1)
    vol = np.full(grid.shape[0], h * h)
    rho_bulk = meta.rho_bulk

    records = []
    for fi in range(0, meta.n_frames, frame_stride):
        fr = bundle.frames[fi]
        if fr.mask is None or fr.mask.sum() < 200:
            continue
        m = fr.mask
        xp = fr.x[m]; vp = fr.v[m]
        fld = StreamFunctionField(xp, vp, n_knots=n_knots, lam_smooth=lam_smooth)
        v_g = fld.velocity(grid)
        L_g = fld.grad_v(grid)
        D_g = fld.D(grid)
        p_g = None
        if pressure_mode == "true":
            p_part = pressure_from_cauchy_3d_trace(dump.stress[fi, np.where(dump.active[fi])[0]])
            xa = dump.x[fi, dump.active[fi]][:, [ax, az]]
            p_g = griddata(xa, p_part, grid, method="linear", fill_value=0.0)
        from scipy.spatial import cKDTree
        tree = cKDTree(xp)
        dist, _ = tree.query(grid, k=1)
        near = dist < 1.5 * h
        records.append({"t": fr.t, "v": v_g, "L": L_g, "D": D_g,
                        "near": near, "p": p_g})

    if not records:
        return [], bundle, dx

    field_times = np.array([rec["t"] for rec in records], dtype=float)
    velocity = np.stack([rec["v"] for rec in records])
    if len(records) > 1:
        dvdt = np.gradient(velocity, field_times, axis=0,
                           edge_order=2 if len(records) > 2 else 1)
    else:
        dvdt = np.zeros_like(velocity)

    frames_out = []
    for ri, rec in enumerate(records):
        material_accel = dvdt[ri] + np.einsum("nij,nj->ni", rec["L"], rec["v"])
        p_g = rec["p"]
        if pressure_mode == "p1":
            p_g = p1_particles(grid[:, 0], grid[:, 1], material_accel[:, 1],
                               np.full(grid.shape[0], rho_bulk), 4 * dx)
        elif pressure_mode == "p0":
            p_g = p0_particles(grid[:, 0], grid[:, 1],
                               np.full(grid.shape[0], rho_bulk), 4 * dx)
        elif pressure_mode != "true":
            raise ValueError(
                f"pressure_mode must be 'true', 'p1', or 'p0', got {pressure_mode!r}")
        gd = equivalent_shear_rate(rec["D"], EPS_GAMMA_DEFAULT)
        I_g = inertial_number(gd, np.maximum(p_g, 1e-9), d, rho_s)
        # mask: inside the data region (near flowing particles), flowing, p>0
        near = rec["near"]
        gmask = near & (p_g > 0) & flowing_mask(gd, I_g) & np.isfinite(I_g)
        if gmask.sum() < 50:
            continue
        frames_out.append(FrameData(
            t=rec["t"], x=grid, v=rec["v"], D=rec["D"],
            a=material_accel,
            p=p_g, I=I_g, vol=vol, rho=np.full(grid.shape[0], rho_bulk),
            mask=gmask,
        ))
    return frames_out, bundle, dx


def run_gate(dump_path, out_dir="out/g3", dictionary=None, pressure_mode="true",
             lam=1e-8, patch_radius_cells=6.0):
    dump_path = Path(dump_path); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = load_dump(dump_path)
    dic = dictionary if dictionary is not None else ConstantDict()
    mode = dic.metadata["mode"]

    frames, bundle, dx = _reconstruct_frames(dump, pressure_mode=pressure_mode)
    if not frames:
        return {"status": "NO_FRAMES", "passed": False}

    r = patch_radius_cells * dx
    r_t = 30.0 * dump.meta.frame_dt
    # patch stratification expects a bundle-like object; reuse stratify_patches
    class _B:  # minimal shim
        pass
    shim = _B(); shim.frames = frames
    shim.times = np.array([f.t for f in frames])
    rows, pmeta = stratify_patches(shim, r, r, r_t, min_particles=40)
    if not rows:
        return {"status": "NO_PATCHES", "passed": False}

    sysm = assemble_system(frames, rows, dic, EPS_GAMMA_DEFAULT)
    A = sysm.A
    keep = np.linalg.norm(A, axis=1) > 1e-12 * np.linalg.norm(A, axis=1).max()
    # time-weak load is the perception-faithful one (no acceleration)
    res_tw = ridge_solve(A[keep], sysm.b_tw[keep], lam=lam)

    observed_values = np.concatenate([fr.I[fr.mask] for fr in frames if np.any(fr.mask)])
    observed_I = (float(np.percentile(observed_values, 5)),
                  float(np.percentile(observed_values, 90)))
    result = {
        "dump": str(dump_path), "mode": mode, "pressure_mode": pressure_mode,
        "n_frames": len(frames), "n_rows": int(keep.sum()),
        "observed_I": list(observed_I),
        "theta_hat_timeweak": res_tw.theta.tolist(),
        "cond": res_tw.cond_AtA,
    }
    meta = dump.meta
    Igrid = np.logspace(np.log10(max(observed_I[0], 1e-3)),
                        np.log10(max(observed_I[1], 1e-2)), 80)
    mu_hat = dic.phi(Igrid) @ res_tw.theta
    curves = {f"field_{pressure_mode}": mu_hat}
    if mode == "C" and meta.law == "constant":
        mu_s = meta.law_params["mu_s"]
        curves["truth"] = np.full_like(Igrid, mu_s)
        result["mu_hat"] = float(res_tw.theta[0])
        result["mu_relative_error"] = float(abs(res_tw.theta[0] - mu_s) / mu_s)
        result["passed"] = bool(result["mu_relative_error"] < 0.05)
    elif meta.law == "pouliquen":
        lp = meta.law_params
        mu_true = lp["mu_s"] + lp["delta_mu"] * Igrid / (Igrid + lp["I0"])
        curves["truth"] = mu_true
        result["mu_curve_relative_L2"] = float(
            np.sqrt(np.mean((mu_hat - mu_true) ** 2)) / np.sqrt(np.mean(mu_true ** 2)))
        result["passed"] = bool(result["mu_curve_relative_L2"] < 0.15)
    fig = plot_mu_curves(Igrid, curves, out_dir / f"g3_{mode}_{pressure_mode}.png",
                         observed_I=observed_I,
                         title=f"G3 field-based {mode} (pressure={pressure_mode})")
    result["figure"] = str(fig)
    with open(out_dir / f"results_{mode}_{pressure_mode}.json", "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "out/dumps/column_constant_a2.npz"
    pm = sys.argv[2] if len(sys.argv) > 2 else "true"
    res = run_gate(path, pressure_mode=pm)
    print(json.dumps(res, indent=2, default=float))
