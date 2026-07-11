"""G1 oracle gate: identify mu(I) from a warp-mpm dump using simulator pressure.

This Phase 1 gate runs a constant-mu solve on oracle dumps with pressure computed from
the MPM stress. Later stages use the same pipeline with Mode P
(``PouliquenGridDict``); only the dictionary changes. Later stages require this gate
to pass.

Pipeline (MATH_REFERENCE.md Sections 4, 2.3, 8):
  1. load + validate dump, build in-plane FrameData with flowing + validity
     masks and the gate-transient time cut.
  2. stratify space-time patches over the valid region (I-decile balanced),
     three test-function rows each.
  3. momentum-closure diagnostic before any regression.
  4. assemble A, b_acc (trajectory acceleration) and b_tw (time-weak); solve
     ridge; report mu_hat, cond(A^T A), effective rank, row survival, and the
     acceleration-vs-time-weak agreement.
  5. write results.json and the mu(I) figure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from common.conventions import SEED_DEFAULT, git_rev
from ident.features.base import Dictionary
from ident.features.constant import ConstantDict
from ident.features.pouliquen_grid import PouliquenGridDict
from ident.gates.plotting import plot_mu_curves
from ident.io.schema import load_dump
from ident.solve.ridge import ridge_solve
from ident.solve.qp import constrained_solve
from ident.weakform.assembly import assemble_system
from ident.weakform.closure import closure_diagnostic
from ident.weakform.from_dump import build_inplane_frames, in_plane_sigma_frames
from ident.weakform.grid_assembly import assemble_grid_consistent
from ident.weakform.test_functions import patch_rows

RESULTS_SCHEMA_VERSION = "g1-2.0"
FLOW_FRAC_MIN = 0.90  # node-purity threshold (essentially all-flowing material)


def stratify_patches(
    bundle,
    r_x: float,
    r_z: float,
    r_t: float,
    overlap: float = 0.5,
    n_time: int = 5,
    min_particles: int = 40,
):
    """Tile space-time patches over the valid region, keep well-populated ones.

    Returns (rows, patch_meta). Patches are accepted when their space-time
    support contains at least min_particles valid particle-quadrature points
    summed over the frames in the time window; each accepted patch emits the
    standard three rows (full, half, offset).
    """
    frames = bundle.frames
    times = bundle.times
    # spatial extent of the valid region
    xs_all, zs_all, t_all = [], [], []
    for fr in frames:
        if fr.mask is None or not np.any(fr.mask):
            continue
        xs_all.append(fr.x[fr.mask, 0])
        zs_all.append(fr.x[fr.mask, 1])
        t_all.append(fr.t)
    if not xs_all:
        return [], []
    xcat = np.concatenate(xs_all)
    zcat = np.concatenate(zs_all)
    x0, x1 = np.percentile(xcat, 1), np.percentile(xcat, 99)
    z0, z1 = np.percentile(zcat, 1), np.percentile(zcat, 99)
    t0, t1 = min(t_all), max(t_all)

    step_x = max(r_x * 2 * (1 - overlap), r_x)
    step_z = max(r_z * 2 * (1 - overlap), r_z)
    x_centers = np.arange(x0 + r_x, x1 - r_x + 1e-9, step_x)
    z_centers = np.arange(z0 + r_z, z1 - r_z + 1e-9, step_z)
    if len(x_centers) == 0:
        x_centers = np.array([(x0 + x1) / 2])
    if len(z_centers) == 0:
        z_centers = np.array([(z0 + z1) / 2])
    t_centers = np.linspace(t0 + r_t, t1 - r_t, n_time) if t1 - t0 > 2 * r_t else \
        np.array([(t0 + t1) / 2])

    rows, patch_meta = [], []
    for tc in t_centers:
        # frames within this time window
        fidx = [i for i, fr in enumerate(frames) if abs(fr.t - tc) < r_t]
        for xc in x_centers:
            for zc in z_centers:
                count = 0
                I_in = []
                for i in fidx:
                    fr = frames[i]
                    if fr.mask is None:
                        continue
                    sel = fr.mask & (np.abs(fr.x[:, 0] - xc) < r_x) & \
                        (np.abs(fr.x[:, 1] - zc) < r_z)
                    count += int(sel.sum())
                    if np.any(sel):
                        I_in.append(fr.I[sel])
                if count >= min_particles:
                    rows.extend(patch_rows(xc, zc, tc, r_x, r_z, r_t))
                    patch_meta.append(
                        dict(xc=float(xc), zc=float(zc), tc=float(tc),
                             count=count,
                             I_median=float(np.median(np.concatenate(I_in))))
                    )
    return rows, patch_meta


def run_gate(
    dump_path: str | Path,
    out_dir: str | Path = "out/g1",
    dictionary: Dictionary | None = None,
    lam: float = 1.0e-8,
    patch_radius_cells: float = 4.0,
) -> dict:
    dump_path = Path(dump_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dump = load_dump(dump_path)
    meta = dump.meta
    dic = dictionary if dictionary is not None else ConstantDict()
    mode = dic.metadata["mode"]

    bundle = build_inplane_frames(dump)
    frames = bundle.frames

    # patch radii scaled to the grid; r_t spans a few dump frames
    dx = float(meta.extra.get("config", {}).get("dx", meta.grain_diameter * 4))
    if "config" in meta.extra:
        cfg = meta.extra["config"]
        dx = cfg["grid_lim"] / cfg["n_grid"]
    r = patch_radius_cells * dx
    r_t = 30.0 * meta.frame_dt
    rows, patch_meta = stratify_patches(bundle, r, r, r_t)

    # ---- momentum-closure diagnostic before any regression ----
    sigma_frames = in_plane_sigma_frames(dump)
    closure_patches = [(pm["xc"], pm["zc"], pm["tc"], r, r, r_t) for pm in patch_meta[:40]]
    closure = closure_diagnostic(frames, sigma_frames, closure_patches)

    # ---- assemble + solve ----
    result: dict = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "git_rev": git_rev(),
        "random_seed": SEED_DEFAULT,
        "mode": mode,
        "pressure_source": meta.pressure_source,
        "dictionary_mode": mode,
        "regularization_lambda": lam,
        "dump": str(dump_path),
        "aspect": meta.extra.get("aspect"),
        "plane_strain_residual": bundle.plane_strain_residual,
        "gate_transient_cut": bundle.gate_cut,
        "valid_particle_frames": bundle.n_rows_valid_total,
        "n_patches": len(patch_meta),
        "observed_I_min": bundle.I_observed[0],
        "observed_I_max": bundle.I_observed[1],
        "closure_error_summary": {
            "worst_full": closure.worst_full,
            "worst_admissible": closure.worst_admissible,
            "median_admissible": float(np.median(closure.rel_error_admissible))
            if closure.rel_error_admissible else None,
        },
    }

    # ---- primary recovery: grid-consistent (Bubnov-Galerkin) assembly ----
    # EUCLID lesson: test functions in the simulator's own discrete (grid
    # B-spline) space with its grad-N operator remove the operator-mismatch
    # bias that independent analytic bump functions sampled at particles incur
    # (docs/MATH_REFERENCE.md Section 2.5). The bump-based system is retained
    # only as the momentum-closure diagnostic above.
    gs = assemble_grid_consistent(dump, dic, bundle.eps_gamma,
                                  flow_frac_min=FLOW_FRAC_MIN,
                                  gate_clearance_time=0.0)
    if gs.n_rows == 0:
        result["status"] = "NO_GRID_ROWS"
        result["passed"] = False
        with open(out_dir / "results.json", "w") as fh:
            json.dump(result, fh, indent=2, default=float)
        return result

    # Mode C: closed-form ridge. Mode P / F: constrained QP (nonnegativity +
    # admissibility + monotonicity) because the basis is collinear and the
    # unconstrained solve returns unphysical coefficients.
    I_lo = max(bundle.I_observed[0], 1e-3)
    I_hi = max(min(bundle.I_observed[1], 1.0), I_lo * 10)
    if mode == "C":
        res = ridge_solve(gs.A, gs.b, lam=lam)
        theta_hat = res.theta
        cond, eff_rank, res_rel = res.cond_AtA, res.effective_rank, res.residual_rel
        sigma_theta = res.Sigma_theta
        qp_status = None
    else:
        G = dic.gram((np.logspace(-4, 0, 257), np.ones(257)))
        I_con = np.logspace(np.log10(I_lo), np.log10(I_hi), 40)
        qp = constrained_solve(gs.A, gs.b, dic, lam=max(lam, 1e-6), G=G,
                               mu_min=0.05, I_constraint_grid=I_con,
                               nonnegativity=True, monotonic=True)
        theta_hat = qp.theta
        cond, eff_rank, res_rel = qp.cond_AtA, qp.effective_rank, qp.residual_rel
        sigma_theta = np.zeros((dic.K, dic.K))  # constrained posterior TODO
        qp_status = qp.status

    res = type("R", (), {"theta": theta_hat, "cond_AtA": cond,
                         "effective_rank": eff_rank, "residual_rel": res_rel,
                         "sigma2": float("nan"), "Sigma_theta": sigma_theta})()

    # threshold-sensitivity of mu_hat (the residual particle-vs-grid
    # acceleration gap shows up as a few-percent dependence on node purity)
    sens = {}
    for ff in (0.85, 0.90, 0.95):
        g2 = assemble_grid_consistent(dump, dic, bundle.eps_gamma, flow_frac_min=ff)
        if g2.n_rows:
            if mode == "C":
                t2 = ridge_solve(g2.A, g2.b, lam=lam).theta
                sens[f"{ff:.2f}"] = float(t2[0])
            else:
                G2 = dic.gram((np.logspace(-4, 0, 257), np.ones(257)))
                I_con = np.logspace(np.log10(I_lo), np.log10(I_hi), 40)
                t2 = constrained_solve(g2.A, g2.b, dic, lam=max(lam, 1e-6), G=G2,
                                       mu_min=0.05, I_constraint_grid=I_con,
                                       nonnegativity=True, monotonic=True).theta
                sens[f"{ff:.2f}"] = float((dic.phi(np.array([0.1])) @ t2)[0])

    mu_s_true = meta.law_params.get("mu_s") if meta.law == "constant" else None
    theta_true = meta.theta_true

    I_grid = np.logspace(
        np.log10(max(bundle.I_observed[0], 1e-4)),
        np.log10(max(bundle.I_observed[1], 1e-3)), 100,
    )
    mu_hat_curve = dic.phi(I_grid) @ res.theta
    curves = {"true_p": mu_hat_curve}
    if mode == "C" and mu_s_true is not None:
        curves["truth"] = np.full_like(I_grid, mu_s_true)
    elif meta.law == "pouliquen":
        lp = meta.law_params
        curves["truth"] = lp["mu_s"] + lp["delta_mu"] * I_grid / (I_grid + lp["I0"])
    elif theta_true is not None and len(theta_true) == dic.K:
        curves["truth"] = dic.phi(I_grid) @ theta_true
    band = np.sqrt(np.maximum(
        np.einsum("nk,kl,nl->n", dic.phi(I_grid), res.Sigma_theta, dic.phi(I_grid)), 0.0))
    fig = plot_mu_curves(
        I_grid, curves, out_dir / f"g1_{mode}_mu.png", bands={"true_p": band},
        observed_I=bundle.I_observed,
        title=f"G1 oracle {mode}  a={meta.extra.get('aspect')}  true pressure (grid-consistent)",
    )

    result.update({
        "assembly": "grid_consistent_bubnov_galerkin",
        "flow_frac_min": FLOW_FRAC_MIN,
        "grid_rows": gs.n_rows,
        "condition_number": res.cond_AtA,
        "effective_rank": res.effective_rank,
        "theta_hat": res.theta.tolist(),
        "posterior_summary": {
            "sigma2": res.sigma2,
            "theta_std": np.sqrt(np.diag(res.Sigma_theta)).tolist(),
            "residual_rel": res.residual_rel,
        },
        "mu_threshold_sensitivity": sens,
        "paths_to_figures": [str(fig)],
    })

    result["qp_status"] = qp_status

    if mode == "C" and mu_s_true is not None:
        mu_hat = float(res.theta[0])
        rel_err = abs(mu_hat - mu_s_true) / mu_s_true
        result["mu_hat"] = mu_hat
        result["mu_s_true"] = mu_s_true
        result["mu_relative_error"] = rel_err
        result["passed"] = bool(rel_err < 0.02)  # within 2 percent (def of done)
    elif meta.law == "pouliquen":
        lp = meta.law_params
        mu_true_curve = lp["mu_s"] + lp["delta_mu"] * I_grid / (I_grid + lp["I0"])
        relL2 = float(np.sqrt(np.mean((mu_hat_curve - mu_true_curve) ** 2))
                      / np.sqrt(np.mean(mu_true_curve ** 2)))
        result["mu_curve_relative_L2"] = relL2
        result["theta_true_law"] = {"mu_s": lp["mu_s"], "delta_mu": lp["delta_mu"],
                                    "I0": lp["I0"]}
        result["mu_at_sample_I"] = {
            f"{Iv:.3f}": float((dic.phi(np.atleast_1d(Iv)) @ res.theta)[0])
            for Iv in (0.003, 0.01, 0.03, 0.1, 0.3)
        }
        if mode == "P":
            # mu_s and friction rise are meaningful only for the Pouliquen
            # dictionary (column 0 = mu_s, columns 1.. = the rise); for the
            # learned Mode F basis theta has no such interpretation
            result["mu_s_hat"] = float(res.theta[0])
            result["friction_rise_hat"] = float(res.theta[1:].sum())
        result["passed"] = bool(relL2 < 0.10)  # curve within 10 percent L2

    result["config_hash"] = hashlib.sha256(
        json.dumps({"dump": str(dump_path), "mode": mode, "lam": lam}, sort_keys=True).encode()
    ).hexdigest()[:16]

    with open(out_dir / "results.json", "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    return result


if __name__ == "__main__":
    import sys as _sys

    path = _sys.argv[1] if len(_sys.argv) > 1 else "out/dumps/column_constant_a2.npz"
    mode = _sys.argv[2] if len(_sys.argv) > 2 else "C"
    if mode == "C":
        dic = ConstantDict()
    elif mode == "P":
        dic = PouliquenGridDict()
    else:  # F: the learned function-encoder basis (frozen tabulation)
        from ident.features.function_encoder import FunctionEncoderDict
        fe = np.load("mpm_engine/fe-weights/granular_mu_i.npz")
        dic = FunctionEncoderDict(fe["s_grid"], fe["table"])
    res = run_gate(path, out_dir=f"out/g1_{mode}", dictionary=dic)
    print(json.dumps({k: v for k, v in res.items() if k != "paths_to_figures"}, indent=2, default=float))
    print("G1", "PASSED" if res.get("passed") else "CHECK", "mu_hat=", res.get("mu_hat"))
