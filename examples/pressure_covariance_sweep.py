"""Force/pressure sweep for viscoplastic dough-press identifiability.

This is a headless experiment for the paper's covariance claim. Each run presses the same
Bingham-like dough with a constant-speed plate and stops at a different target load. The measured
motion builds the weak-form power-balance design matrix for theta = [tau_y, eta]:

    P_dev = tau_y * INT(q / gamma_dot) dV + eta * INT(q) dV,
    q = 2 dev(D):dev(D), gamma_dot = sqrt(q + eps^2).

For each target pressure we report the recovered parameters, the information matrix A^T A,
and cov(theta_hat) = sigma_power^2 (A^T A)^-1 under a fixed power-noise assumption.

Important caveat: this is the scalar power-balance diagnostic, not the paper's full
divergence-free tensor weak-form estimator. The paper-style estimator for the press needs
discrete virtual work of the contact load, i.e. contact/grid impulse distribution weighted by
the same divergence-free test fields used in A. A total plate force is enough for scalar power,
but not enough for arbitrary tensor weak-form rows.

Run:
  python examples/pressure_covariance_sweep.py --device cuda:0
  python examples/pressure_covariance_sweep.py --device cuda:0 --seed 0  # quick single-seed run
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, block, newtonian
from warpmpm.coupling.backend import WarpMPMBackend

OUT = Path(__file__).resolve().parents[1] / "out" / "pressure_covariance_sweep"
G_MAG = 9.81
EPS_GAMMA = 0.05


def equivalent_shear_rate(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (gamma_dot_eps, q), q = 2 dev(D):dev(D), D = sym(L)."""
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    tr = (D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2]) / 3.0
    dev = D - tr[..., None, None] * np.eye(3)
    q = 2.0 * np.einsum("...ij,...ij->...", dev, dev)
    return np.sqrt(q + EPS_GAMMA * EPS_GAMMA), q


def _information_stats_from_AtA(AtA: np.ndarray, sigma_power: float, theta_ref: tuple[float, float]) -> dict:
    eig_info = np.linalg.eigvalsh(AtA)
    rank = int(np.linalg.matrix_rank(AtA, tol=max(float(eig_info[-1]), 1.0) * 1e-10))
    cond = float(np.linalg.cond(AtA)) if rank == 2 else float("inf")
    if rank == 2:
        cov = sigma_power * sigma_power * np.linalg.inv(AtA)
        eig_cov = np.linalg.eigvalsh(cov)
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        corr = float(cov[0, 1] / max(std[0] * std[1], 1e-30))
        major_std = float(np.sqrt(max(eig_cov[-1], 0.0)))
        minor_std = float(np.sqrt(max(eig_cov[0], 0.0)))
    else:
        cov = np.full((2, 2), np.nan)
        eig_cov = np.array([np.nan, np.inf])
        std = np.array([np.inf, np.inf])
        corr = float("nan")
        major_std = float("inf")
        minor_std = float("nan")
    tau_y, eta = theta_ref
    return {
        "AtA": AtA,
        "eig_info": eig_info,
        "rank": rank,
        "cond": cond,
        "cov": cov,
        "eig_cov": eig_cov,
        "std_tau_y": float(std[0]),
        "std_eta": float(std[1]),
        "rel_std_tau_y": float(std[0] / max(abs(tau_y), 1e-30)),
        "rel_std_eta": float(std[1] / max(abs(eta), 1e-30)),
        "corr_tau_y_eta": corr,
        "major_axis_std": major_std,
        "minor_axis_std": minor_std,
    }


def _information_stats(A: np.ndarray, sigma_power: float, theta_ref: tuple[float, float]) -> dict:
    return _information_stats_from_AtA(A.T @ A, sigma_power=sigma_power, theta_ref=theta_ref)


def _fit_design_matrix(A: np.ndarray, b: np.ndarray, sigma_power: float, theta_ref: tuple[float, float]) -> dict:
    if len(b) < 2:
        return {"ok": False, "reason": "not enough rows"}
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ theta
    resid = b - pred
    dof = max(len(b) - 2, 1)
    sigma_fit = float(np.sqrt(np.dot(resid, resid) / dof))
    relres = float(np.linalg.norm(resid) / max(np.linalg.norm(b), 1e-30))
    stats = _information_stats(A, sigma_power=sigma_power, theta_ref=theta_ref)
    fit_stats = _information_stats(A, sigma_power=sigma_fit, theta_ref=theta_ref)
    return {
        "ok": True,
        "A": A,
        "b": b,
        "theta": theta,
        "pred": pred,
        "resid": resid,
        "sigma_fit": sigma_fit,
        "fit_relres": relres,
        "n_rows": int(len(b)),
        "fixed_noise": stats,
        "fit_noise": fit_stats,
    }


def _fit_power_balance(records: list[dict], sigma_power: float, theta_ref: tuple[float, float]) -> dict:
    if len(records) < 4:
        return {"ok": False, "reason": "not enough active rows"}

    t = np.array([r["t"] for r in records])
    KE = np.array([r["KE"] for r in records])
    A_rows = []
    b_rows = []
    used = []
    for i, r in enumerate(records):
        i0 = max(i - 1, 0)
        i1 = min(i + 1, len(records) - 1)
        dKE = (KE[i1] - KE[i0]) / max(t[i1] - t[i0], 1e-30)
        p_plate = r["F_grid"] * r["down_speed"]
        diss = p_plate + r["P_grav"] - dKE
        if r["X1"] <= 1e-14 or r["X2"] <= 1e-14:
            continue
        A_rows.append((r["X1"], r["X2"]))
        b_rows.append(diss)
        used.append(r)

    if len(A_rows) < 4:
        return {"ok": False, "reason": "not enough nonzero rows"}

    A = np.asarray(A_rows)
    b = np.asarray(b_rows)
    fit = _fit_design_matrix(A, b, sigma_power=sigma_power, theta_ref=theta_ref)
    fit["records"] = used
    return fit


def run_probe(
    f_target: float,
    *,
    device: str,
    n_grid: int,
    ticks: int,
    substeps: int,
    dt: float,
    v_max: float,
    tau_y: float,
    eta: float,
    density: float,
    bulk: float,
    dough_size: tuple[float, float, float],
    sigma_power: float,
    seed: int,
) -> dict:
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    pos, vol0, floor = block(grid, size=dough_size, ppc=2, seed=seed)
    solver = Solver(grid=grid, device=device).load_particles(pos, vol0)
    solver.set_material(newtonian(eta=eta, density=density, bulk_modulus=bulk).with_yield(tau_y))
    solver.add_plane((0, 0, floor), (0, 0, 1), "sticky")

    cx = cy = grid.grid_lim * 0.5
    col_w, col_d, col_h = dough_size
    plate_half = (0.5 * col_w + 0.015, 0.5 * col_d + 0.015, 0.6 * grid.dx)
    plate_area = 4.0 * plate_half[0] * plate_half[1]
    dough_top = floor + col_h
    dt_ctrl = dt * substeps
    backend = WarpMPMBackend(solver=solver)

    z = dough_top + plate_half[2]
    z_floor = floor + plate_half[2] + 0.003
    tool = backend.attach_tool((cx, cy, z), plate_half, velocity=(0, 0, 0))

    active_records = []
    all_records = []
    for tick in range(ticks + 1):
        v_down_cmd = 0.0 if tick == 0 else v_max
        z_new = max(z - v_down_cmd * dt_ctrl, z_floor)
        vz = (z_new - z) / dt_ctrl
        down_speed = max(0.0, -vz)
        if tick > 0:
            backend.set_tool_kinematics(tool, center=(cx, cy, z), velocity=(0, 0, vz))
            backend.reset_tool_force(tool)
            backend.step(dt, substeps)
        z = z_new

        plate_bottom = z - plate_half[2]
        f_grid = max(float(backend.get_tool_reaction(tool, dt_ctrl)[2]), 0.0) if tick > 0 else 0.0

        v = solver.v()
        L = solver.L()
        vol = solver.vol()
        gd, q = equivalent_shear_rate(L)
        X1 = float(np.sum((q / np.maximum(gd, 1e-12)) * vol))
        X2 = float(np.sum(q * vol))
        KE = float(0.5 * density * np.sum(vol * np.sum(v * v, axis=1)))
        P_grav = float(np.sum(density * (-G_MAG) * v[:, 2] * vol))
        strain = (col_h - (plate_bottom - floor)) / col_h
        rec = {
            "tick": tick,
            "t": tick * dt_ctrl,
            "z": z,
            "down_speed": down_speed,
            "F_grid": f_grid,
            "F_static": f_grid,
            "F_filt": f_grid,
            "X1": X1,
            "X2": X2,
            "P_grav": P_grav,
            "KE": KE,
            "strain": strain,
            "depth_mm": max(0.0, col_h - (plate_bottom - floor)) * 1e3,
            "gap_mm": (plate_bottom - floor) * 1e3,
            "n_contact": int(f_grid > 0.0),
        }
        all_records.append(rec)
        if tick > 0 and down_speed > 1e-5 and f_grid > 0.01 * f_target and np.isfinite(X2) and X2 > 1e-14:
            active_records.append(rec)

        if tick > 0 and f_grid >= f_target:
            break

    fit = _fit_power_balance(active_records, sigma_power=sigma_power, theta_ref=(tau_y, eta))
    peak_grid = max(r["F_grid"] for r in all_records)
    summary = {
        "seed": seed,
        "fit_method": "power_balance",
        "f_target": f_target,
        "target_pressure_pa": f_target / plate_area,
        "target_pressure_kpa": f_target / plate_area / 1000.0,
        "tail_pressure_pa": peak_grid / plate_area,
        "tail_pressure_kpa": peak_grid / plate_area / 1000.0,
        "plate_area_m2": plate_area,
        "peak_static_force": peak_grid,
        "peak_grid_force": peak_grid,
        "tail_static_force": peak_grid,
        "final_depth_mm": all_records[-1]["depth_mm"],
        "final_strain_pct": 100.0 * all_records[-1]["strain"],
        "ticks_run": len(all_records) - 1,
        "active_rows": len(active_records),
        "hit_floor": bool(all_records[-1]["gap_mm"] <= 3.1),
    }
    if fit["ok"]:
        fixed = fit["fixed_noise"]
        fit_noise = fit["fit_noise"]
        summary.update({
            "tau_y_hat": float(fit["theta"][0]),
            "eta_hat": float(fit["theta"][1]),
            "tau_y_err": abs(float(fit["theta"][0]) / tau_y - 1.0),
            "eta_err": abs(float(fit["theta"][1]) / eta - 1.0),
            "fit_relres": fit["fit_relres"],
            "sigma_fit_w": fit["sigma_fit"],
            "rank": fixed["rank"],
            "cond": fixed["cond"],
            "std_tau_y_fixed": fixed["std_tau_y"],
            "std_eta_fixed": fixed["std_eta"],
            "rel_std_tau_y_fixed": fixed["rel_std_tau_y"],
            "rel_std_eta_fixed": fixed["rel_std_eta"],
            "corr_tau_y_eta_fixed": fixed["corr_tau_y_eta"],
            "major_axis_std_fixed": fixed["major_axis_std"],
            "std_tau_y_fit": fit_noise["std_tau_y"],
            "std_eta_fit": fit_noise["std_eta"],
            "rel_std_tau_y_fit": fit_noise["rel_std_tau_y"],
            "rel_std_eta_fit": fit_noise["rel_std_eta"],
            "cov_fixed": fixed["cov"].tolist(),
            "AtA": fixed["AtA"].tolist(),
            "eig_info": fixed["eig_info"].tolist(),
        })
    else:
        summary.update({
            "tau_y_hat": float("nan"),
            "eta_hat": float("nan"),
            "tau_y_err": float("nan"),
            "eta_err": float("nan"),
            "fit_relres": float("nan"),
            "sigma_fit_w": float("nan"),
            "rank": 0,
            "cond": float("inf"),
            "std_tau_y_fixed": float("inf"),
            "std_eta_fixed": float("inf"),
            "rel_std_tau_y_fixed": float("inf"),
            "rel_std_eta_fixed": float("inf"),
            "corr_tau_y_eta_fixed": float("nan"),
            "major_axis_std_fixed": float("inf"),
            "std_tau_y_fit": float("inf"),
            "std_eta_fit": float("inf"),
            "rel_std_tau_y_fit": float("inf"),
            "rel_std_eta_fit": float("inf"),
            "cov_fixed": [[float("nan"), float("nan")], [float("nan"), float("nan")]],
            "AtA": [[float("nan"), float("nan")], [float("nan"), float("nan")]],
            "eig_info": [float("nan"), float("nan")],
            "reason": fit["reason"],
        })
    return {"summary": summary, "records": all_records, "active_records": active_records, "fit": fit}


def _target_force_from_pressure_kpa(pressure_kpa: float, dough_size: tuple[float, float, float]) -> float:
    plate_area = (dough_size[0] + 0.03) * (dough_size[1] + 0.03)
    return pressure_kpa * 1000.0 * plate_area


def _format_yield_uncertainty_axis(ax):
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.yaxis.get_offset_text().set_size(11)


def _target_key(value: float) -> float:
    return round(float(value), 8)


def _ordered_summary_keys(summaries: list[dict]) -> list[str]:
    keys = []
    seen = set()
    for summary in summaries:
        for key in summary:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _finite_percentiles(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    p25 = np.full(samples.shape[1], np.nan)
    median = np.full(samples.shape[1], np.nan)
    p75 = np.full(samples.shape[1], np.nan)
    for i in range(samples.shape[1]):
        col = samples[:, i]
        col = col[np.isfinite(col)]
        if len(col) > 0:
            p25[i], median[i], p75[i] = np.percentile(col, [25.0, 50.0, 75.0])
    return p25, median, p75


def _combined_rows_for_seed(
    results_by_force: dict[float, dict],
    *,
    seed: int,
    sigma_power: float,
    tau_y: float,
    eta: float,
    combined_pressures_kpa: list[float],
    dough_size: tuple[float, float, float],
) -> list[dict]:
    combined_rows = []
    A_parts = []
    b_parts = []
    for pressure_kpa in sorted(combined_pressures_kpa):
        f_target = _target_force_from_pressure_kpa(pressure_kpa, dough_size)
        result = results_by_force[_target_key(f_target)]
        fit = result["fit"]
        if not fit["ok"]:
            combined_rows.append({
                "seed": seed,
                "max_pressure_kpa": pressure_kpa,
                "n_experiments": len(combined_rows) + 1,
                "tau_y_hat": float("nan"),
                "eta_hat": float("nan"),
                "tau_y_err": float("nan"),
                "eta_err": float("nan"),
                "rel_std_tau_y": float("inf"),
                "rel_std_eta": float("inf"),
                "rank": 0,
                "cond": float("inf"),
            })
            continue
        A_parts.append(fit["A"])
        b_parts.append(fit["b"])
        A = np.vstack(A_parts)
        b = np.concatenate(b_parts)
        combined_fit = _fit_design_matrix(A, b, sigma_power=sigma_power, theta_ref=(tau_y, eta))
        stats = combined_fit["fixed_noise"]
        tau_y_hat = float(combined_fit["theta"][0])
        eta_hat = float(combined_fit["theta"][1])
        combined_rows.append({
            "seed": seed,
            "max_pressure_kpa": pressure_kpa,
            "n_experiments": len(combined_rows) + 1,
            "tau_y_hat": tau_y_hat,
            "eta_hat": eta_hat,
            "tau_y_err": abs(tau_y_hat / tau_y - 1.0),
            "eta_err": abs(eta_hat / eta - 1.0),
            "rel_std_tau_y": stats["rel_std_tau_y"],
            "rel_std_eta": stats["rel_std_eta"],
            "rank": stats["rank"],
            "cond": stats["cond"],
        })
    return combined_rows


def write_outputs(
    results: list[dict],
    out_dir: Path,
    sigma_power: float,
    tau_y: float,
    eta: float,
    single_targets: list[float],
    combined_pressures_kpa: list[float],
    dough_size: tuple[float, float, float],
    seeds: list[int],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in out_dir.glob("*.png"):
        old_plot.unlink()
    for old_record in out_dir.glob("target_*_records.npz"):
        old_record.unlink()
    for old_record in out_dir.glob("seed_*_target_*_records.npz"):
        old_record.unlink()

    summaries = [r["summary"] for r in results]
    summaries_by_seed_force = {
        seed: {_target_key(s["f_target"]): s for s in summaries if int(s["seed"]) == seed}
        for seed in seeds
    }
    results_by_seed_force = {
        seed: {_target_key(r["summary"]["f_target"]): r for r in results if int(r["summary"]["seed"]) == seed}
        for seed in seeds
    }
    keys = _ordered_summary_keys(summaries)
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    (out_dir / "results.json").write_text(json.dumps(summaries, indent=2, default=float))
    for r in results:
        tag = f"seed_{r['summary']['seed']}_target_{r['summary']['f_target']:.1f}N".replace(".", "p")
        np.savez_compressed(
            out_dir / f"{tag}_records.npz",
            all=np.array([[row[k] for k in row] for row in r["records"]], dtype=float),
            all_keys=np.array(list(r["records"][0].keys())),
        )

    first_seed_summaries = summaries_by_seed_force[seeds[0]]
    ordered_single_targets = sorted(
        single_targets,
        key=lambda f: first_seed_summaries[_target_key(f)]["target_pressure_kpa"],
    )
    P = np.array([
        first_seed_summaries[_target_key(f)]["target_pressure_kpa"] for f in ordered_single_targets
    ])
    rel_tau_by_seed = np.array([
        [
            summaries_by_seed_force[seed][_target_key(f)]["rel_std_tau_y_fixed"]
            for f in ordered_single_targets
        ]
        for seed in seeds
    ])
    rel_tau_p25, rel_tau_median, rel_tau_p75 = _finite_percentiles(rel_tau_by_seed)
    finite = np.isfinite(P) & np.isfinite(rel_tau_median)
    single_pressure_rows = []
    for i, pressure_kpa in enumerate(P):
        single_pressure_rows.append({
            "target_pressure_kpa": float(pressure_kpa),
            "n_seeds": len(seeds),
            "rel_std_tau_y": rel_tau_median[i],
            "rel_std_tau_y_p25": rel_tau_p25[i],
            "rel_std_tau_y_p75": rel_tau_p75[i],
        })
    (out_dir / "single_pressure_results.json").write_text(
        json.dumps(single_pressure_rows, indent=2, default=float)
    )
    with open(out_dir / "single_pressure_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(single_pressure_rows[0].keys()))
        writer.writeheader()
        writer.writerows(single_pressure_rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(P[finite], rel_tau_p25[finite], rel_tau_p75[finite],
                    color="tab:blue", alpha=0.16, linewidth=0)
    ax.plot(P[finite], rel_tau_median[finite], "o-", lw=2.2,
            label="median across seeds")
    ax.set_xlim(0.0, max(float(P.max()) * 1.05, 1.0))
    ax.set_ylim(bottom=0.0)
    _format_yield_uncertainty_axis(ax)
    ax.set_xlabel("target plate pressure  (kPa)")
    ax.set_ylabel(r"yield-stress uncertainty  $\sqrt{C_{\tau_y\tau_y}} / \tau_y$")
    ax.set_title("Yield Stress Uncertainty")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "relative_uncertainty_vs_pressure.png", dpi=160)
    plt.close(fig)

    combined_rows_by_seed = []
    combined_rel_by_seed = []
    combined_eta_rel_by_seed = []
    combined_tau_hat_by_seed = []
    combined_eta_hat_by_seed = []
    single_combined_rel_by_seed = []
    single_combined_eta_rel_by_seed = []
    single_combined_tau_hat_by_seed = []
    single_combined_eta_hat_by_seed = []
    for seed in seeds:
        rows = _combined_rows_for_seed(
            results_by_seed_force[seed],
            seed=seed,
            sigma_power=sigma_power,
            tau_y=tau_y,
            eta=eta,
            combined_pressures_kpa=combined_pressures_kpa,
            dough_size=dough_size,
        )
        combined_rows_by_seed.extend(rows)
        combined_rel_by_seed.append([r["rel_std_tau_y"] for r in rows])
        combined_eta_rel_by_seed.append([r["rel_std_eta"] for r in rows])
        combined_tau_hat_by_seed.append([r["tau_y_hat"] for r in rows])
        combined_eta_hat_by_seed.append([r["eta_hat"] for r in rows])
        single_combined_rel_by_seed.append([
            summaries_by_seed_force[seed][
                _target_key(_target_force_from_pressure_kpa(r["max_pressure_kpa"], dough_size))
            ]["rel_std_tau_y_fixed"]
            for r in rows
        ])
        single_combined_eta_rel_by_seed.append([
            summaries_by_seed_force[seed][
                _target_key(_target_force_from_pressure_kpa(r["max_pressure_kpa"], dough_size))
            ]["rel_std_eta_fixed"]
            for r in rows
        ])
        single_combined_tau_hat_by_seed.append([
            summaries_by_seed_force[seed][
                _target_key(_target_force_from_pressure_kpa(r["max_pressure_kpa"], dough_size))
            ]["tau_y_hat"]
            for r in rows
        ])
        single_combined_eta_hat_by_seed.append([
            summaries_by_seed_force[seed][
                _target_key(_target_force_from_pressure_kpa(r["max_pressure_kpa"], dough_size))
            ]["eta_hat"]
            for r in rows
        ])

    (out_dir / "combined_results_by_seed.json").write_text(
        json.dumps(combined_rows_by_seed, indent=2, default=float)
    )
    with open(out_dir / "combined_results_by_seed.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined_rows_by_seed[0].keys()))
        writer.writeheader()
        writer.writerows(combined_rows_by_seed)

    Pc = np.array([r["max_pressure_kpa"] for r in combined_rows_by_seed[:len(combined_pressures_kpa)]])
    combined_rel_by_seed = np.asarray(combined_rel_by_seed, dtype=float)
    combined_eta_rel_by_seed = np.asarray(combined_eta_rel_by_seed, dtype=float)
    combined_tau_hat_by_seed = np.asarray(combined_tau_hat_by_seed, dtype=float)
    combined_eta_hat_by_seed = np.asarray(combined_eta_hat_by_seed, dtype=float)
    single_combined_rel_by_seed = np.asarray(single_combined_rel_by_seed, dtype=float)
    single_combined_eta_rel_by_seed = np.asarray(single_combined_eta_rel_by_seed, dtype=float)
    single_combined_tau_hat_by_seed = np.asarray(single_combined_tau_hat_by_seed, dtype=float)
    single_combined_eta_hat_by_seed = np.asarray(single_combined_eta_hat_by_seed, dtype=float)
    rel_tau_c_p25, rel_tau_c_median, rel_tau_c_p75 = _finite_percentiles(combined_rel_by_seed)
    rel_eta_c_p25, rel_eta_c_median, rel_eta_c_p75 = _finite_percentiles(combined_eta_rel_by_seed)
    tau_hat_c_p25, tau_hat_c_median, tau_hat_c_p75 = _finite_percentiles(combined_tau_hat_by_seed)
    eta_hat_c_p25, eta_hat_c_median, eta_hat_c_p75 = _finite_percentiles(combined_eta_hat_by_seed)
    rel_tau_single_p25, rel_tau_single_median, rel_tau_single_p75 = _finite_percentiles(
        single_combined_rel_by_seed
    )
    rel_eta_single_p25, rel_eta_single_median, rel_eta_single_p75 = _finite_percentiles(
        single_combined_eta_rel_by_seed
    )
    tau_hat_single_p25, tau_hat_single_median, tau_hat_single_p75 = _finite_percentiles(
        single_combined_tau_hat_by_seed
    )
    eta_hat_single_p25, eta_hat_single_median, eta_hat_single_p75 = _finite_percentiles(
        single_combined_eta_hat_by_seed
    )
    combined_rows = []
    for i, pressure_kpa in enumerate(Pc):
        combined_rows.append({
            "max_pressure_kpa": float(pressure_kpa),
            "n_experiments": i + 1,
            "n_seeds": len(seeds),
            "rel_std_tau_y": rel_tau_c_median[i],
            "rel_std_tau_y_p25": rel_tau_c_p25[i],
            "rel_std_tau_y_p75": rel_tau_c_p75[i],
            "single_rel_std_tau_y": rel_tau_single_median[i],
            "single_rel_std_tau_y_p25": rel_tau_single_p25[i],
            "single_rel_std_tau_y_p75": rel_tau_single_p75[i],
            "rel_std_eta": rel_eta_c_median[i],
            "rel_std_eta_p25": rel_eta_c_p25[i],
            "rel_std_eta_p75": rel_eta_c_p75[i],
            "single_rel_std_eta": rel_eta_single_median[i],
            "single_rel_std_eta_p25": rel_eta_single_p25[i],
            "single_rel_std_eta_p75": rel_eta_single_p75[i],
            "tau_y_hat": tau_hat_c_median[i],
            "tau_y_hat_p25": tau_hat_c_p25[i],
            "tau_y_hat_p75": tau_hat_c_p75[i],
            "single_tau_y_hat": tau_hat_single_median[i],
            "single_tau_y_hat_p25": tau_hat_single_p25[i],
            "single_tau_y_hat_p75": tau_hat_single_p75[i],
            "eta_hat": eta_hat_c_median[i],
            "eta_hat_p25": eta_hat_c_p25[i],
            "eta_hat_p75": eta_hat_c_p75[i],
            "single_eta_hat": eta_hat_single_median[i],
            "single_eta_hat_p25": eta_hat_single_p25[i],
            "single_eta_hat_p75": eta_hat_single_p75[i],
        })
    (out_dir / "combined_results.json").write_text(json.dumps(combined_rows, indent=2, default=float))
    with open(out_dir / "combined_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
        writer.writeheader()
        writer.writerows(combined_rows)

    finite_c = np.isfinite(Pc) & np.isfinite(rel_tau_c_median)
    finite_single_c = np.isfinite(Pc) & np.isfinite(rel_tau_single_median)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(Pc[finite_single_c], rel_tau_single_p25[finite_single_c],
                    rel_tau_single_p75[finite_single_c],
                    color="tab:blue", alpha=0.14, linewidth=0)
    ax.plot(Pc[finite_single_c], rel_tau_single_median[finite_single_c], "s--", lw=1.9,
            label="single pressure")
    ax.fill_between(Pc[finite_c], rel_tau_c_p25[finite_c], rel_tau_c_p75[finite_c],
                    color="tab:orange", alpha=0.18, linewidth=0)
    ax.plot(Pc[finite_c], rel_tau_c_median[finite_c], "o-", lw=2.2,
            label="combined pressures")
    ax.set_xlim(0.0, max(float(Pc.max()) * 1.05, 1.0))
    ax.set_ylim(bottom=0.0)
    _format_yield_uncertainty_axis(ax)
    ax.set_xlabel("highest included pressure  (kPa)")
    ax.set_ylabel(r"yield-stress uncertainty  $\sqrt{C_{\tau_y\tau_y}} / \tau_y$")
    ax.set_title("Yield Stress Uncertainty: Single vs Combined Pressures")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "combined_relative_uncertainty_vs_pressure.png", dpi=160)
    plt.close(fig)

    finite_eta_c = np.isfinite(Pc) & np.isfinite(rel_eta_c_median)
    finite_eta_single_c = np.isfinite(Pc) & np.isfinite(rel_eta_single_median)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(Pc[finite_eta_single_c], rel_eta_single_p25[finite_eta_single_c],
                    rel_eta_single_p75[finite_eta_single_c],
                    color="tab:blue", alpha=0.14, linewidth=0)
    ax.plot(Pc[finite_eta_single_c], rel_eta_single_median[finite_eta_single_c], "s--", lw=1.9,
            label="single pressure")
    ax.fill_between(Pc[finite_eta_c], rel_eta_c_p25[finite_eta_c], rel_eta_c_p75[finite_eta_c],
                    color="tab:orange", alpha=0.18, linewidth=0)
    ax.plot(Pc[finite_eta_c], rel_eta_c_median[finite_eta_c], "o-", lw=2.2,
            label="combined pressures")
    ax.set_xlim(0.0, max(float(Pc.max()) * 1.05, 1.0))
    ax.set_ylim(bottom=0.0)
    _format_yield_uncertainty_axis(ax)
    ax.set_xlabel("highest included pressure  (kPa)")
    ax.set_ylabel(r"viscosity uncertainty  $\sqrt{C_{\eta\eta}} / \eta$")
    ax.set_title("Viscosity Uncertainty: Single vs Combined Pressures")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "combined_relative_viscosity_uncertainty_vs_pressure.png", dpi=160)
    plt.close(fig)

    def plot_recovery(
        single_median: np.ndarray,
        single_p25: np.ndarray,
        single_p75: np.ndarray,
        combined_median: np.ndarray,
        combined_p25: np.ndarray,
        combined_p75: np.ndarray,
        true_value: float,
        ylabel: str,
        title: str,
        filename: str,
    ) -> None:
        finite_single = np.isfinite(Pc) & np.isfinite(single_median)
        finite_combined = np.isfinite(Pc) & np.isfinite(combined_median)
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.fill_between(Pc[finite_single], single_p25[finite_single], single_p75[finite_single],
                        color="tab:blue", alpha=0.14, linewidth=0)
        ax.plot(Pc[finite_single], single_median[finite_single], "s--", lw=1.9,
                label="single pressure")
        ax.fill_between(Pc[finite_combined], combined_p25[finite_combined], combined_p75[finite_combined],
                        color="tab:orange", alpha=0.18, linewidth=0)
        ax.plot(Pc[finite_combined], combined_median[finite_combined], "o-", lw=2.2,
                label="combined pressures")
        ax.axhline(true_value, color="black", linestyle=":", lw=1.8, label="true")
        ax.set_xlim(0.0, max(float(Pc.max()) * 1.05, 1.0))
        ax.margins(y=0.08)
        ax.set_xlabel("highest included pressure  (kPa)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    plot_recovery(
        tau_hat_single_median,
        tau_hat_single_p25,
        tau_hat_single_p75,
        tau_hat_c_median,
        tau_hat_c_p25,
        tau_hat_c_p75,
        tau_y,
        r"recovered yield stress  $\tau_y$ (Pa)",
        "Yield Stress Recovery: Single vs Combined Pressures",
        "combined_yield_recovery_vs_pressure.png",
    )
    plot_recovery(
        eta_hat_single_median,
        eta_hat_single_p25,
        eta_hat_single_p75,
        eta_hat_c_median,
        eta_hat_c_p25,
        eta_hat_c_p75,
        eta,
        r"recovered viscosity  $\eta$ (Pa s)",
        "Viscosity Recovery: Single vs Combined Pressures",
        "combined_viscosity_recovery_vs_pressure.png",
    )

    print(f"\nwrote sweep outputs to {out_dir}")
    print(f"main plot: {out_dir / 'relative_uncertainty_vs_pressure.png'}")
    print(f"combined plot: {out_dir / 'combined_relative_uncertainty_vs_pressure.png'}")
    print(
        "combined viscosity plot: "
        f"{out_dir / 'combined_relative_viscosity_uncertainty_vs_pressure.png'}"
    )
    print(f"yield recovery plot: {out_dir / 'combined_yield_recovery_vs_pressure.png'}")
    print(f"viscosity recovery plot: {out_dir / 'combined_viscosity_recovery_vs_pressure.png'}")
    print(f"target pressure range: {P.min():.3f} to {P.max():.3f} kPa")


def run(args) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    dough_size = tuple(args.dough_size)
    seeds = [args.seed] if args.seed is not None else list(args.seeds)
    seeds = list(dict.fromkeys(int(seed) for seed in seeds))
    combined_targets = [
        _target_force_from_pressure_kpa(p, dough_size) for p in args.combined_pressures_kpa
    ]
    targets = []
    seen = set()
    for target in list(args.targets) + combined_targets:
        key = round(float(target), 8)
        if key not in seen:
            targets.append(float(target))
            seen.add(key)

    results = []
    for seed in seeds:
        print(f"\n######## seed {seed} ########", flush=True)
        for f_target in targets:
            print(f"\n=== target force {f_target:.1f} N ===", flush=True)
            result = run_probe(
                f_target,
                device=args.device,
                n_grid=args.n_grid,
                ticks=args.ticks,
                substeps=args.substeps,
                dt=args.dt,
                v_max=args.v_max,
                tau_y=args.tau_y,
                eta=args.eta,
                density=args.density,
                bulk=args.bulk,
                dough_size=dough_size,
                sigma_power=args.sigma_power,
                seed=seed,
            )
            s = result["summary"]
            print(
                f"pressure={s['target_pressure_kpa']:.3f} kPa, "
                f"rows={s['active_rows']}, depth={s['final_depth_mm']:.1f} mm, "
                f"theta=({s['tau_y_hat']:.1f}, {s['eta_hat']:.1f}), "
                f"rel_std=({s['rel_std_tau_y_fixed']:.3g}, {s['rel_std_eta_fixed']:.3g}), "
                f"cond={s['cond']:.2g}",
                flush=True,
            )
            results.append(result)
    write_outputs(
        results,
        OUT,
        sigma_power=args.sigma_power,
        tau_y=args.tau_y,
        eta=args.eta,
        single_targets=list(args.targets),
        combined_pressures_kpa=list(args.combined_pressures_kpa),
        dough_size=dough_size,
        seeds=seeds,
    )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep applied pressure and plot parameter covariance.")
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    parser.add_argument("--targets", nargs="+", type=float,
                        default=[
                            225.0, 180.0, 146.25, 101.25, 72.0, 56.25, 45.0,
                            36.0, 29.25, 24.75, 20.25, 15.75, 11.25,
                        ],
                        help="Target plate forces in N. Default spans about 0.5-10 kPa.")
    parser.add_argument("--combined-pressures-kpa", nargs="+", type=float,
                        default=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
                        help="Pressure levels to combine cumulatively in the second plot.")
    parser.add_argument("--n-grid", type=int, default=48)
    parser.add_argument("--ticks", type=int, default=220)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--v-max", type=float, default=0.08, help="Downward press speed in m/s.")
    parser.add_argument("--sigma-power", type=float, default=0.05,
                        help="Assumed std of power-balance noise b, in Watts, for covariance plots.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Particle initialization seeds for median and seed-band plots.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run one particle initialization seed. Overrides --seeds.")
    parser.add_argument("--tau-y", type=float, default=200.0)
    parser.add_argument("--eta", type=float, default=40.0)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--bulk", type=float, default=9.0e5)
    parser.add_argument("--dough-size", nargs=3, type=float, default=(0.12, 0.12, 0.06),
                        metavar=("W", "D", "H"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
