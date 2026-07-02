"""Arm-mounted plate squeeze: cross-validate the 2D squeeze-flow rheology in 3D.

We reproduce the validated quasi-2D plate squeeze (sim/squeeze_scene.py +
perception/squeeze_force_identify.py) but now the flat plate is MOUNTED ON THE FRANKA and
presses a full 3D dough blob. The material is identical (newtonian eta=40 Pa.s, tau_y=200 Pa,
rho=1000, bulk=9e5) and the plate descends at the same constant v_plate=0.08 m/s. We run the
SAME identification -- the mechanical power balance

    INT tau:D dV = P_plate + P_gravity - dKE/dt ,
    INT tau:D dV = tau_y INT|gd| dV + eta INT|gd|^2 dV   (pressure does no work),

so regressing the measured dissipation (from the plate reaction force, gravity power, and the
kinetic-energy rate) against (INT|gd|, INT|gd|^2) over the squeeze recovers (tau_y, eta) from
the boundary force alone. We then compare the recovered law to ground truth and to the 2D
result. |gd| = sqrt(2 dev(D):dev(D) + eps^2), D = sym(L), eps = 0.05 (matching the kernel).

Run:  ../.venv/bin/python examples/squeeze_plate_franka.py
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import argparse
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, block, newtonian
from warpmpm.coupling.backend import WarpMPMBackend

OUT = Path(__file__).resolve().parents[1] / "out"
G_MAG = 9.81
EPS_GAMMA = 0.05   # match the warp-mpm kernel's shear-rate regularization
# the validated 2D quasi-plane-strain result (perception/squeeze_force.json), for reference
REF_2D = {"tau_y_hat": 272.0, "eta_hat": 50.3, "tau_y_true": 200.0, "eta_true": 40.0}


def equivalent_shear_rate(L: np.ndarray) -> np.ndarray:
    """|gd|_eps = sqrt(2 dev(D):dev(D) + eps^2), D = sym(L); matches the kernel exactly."""
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    tr = (D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2]) / 3.0
    Dd = D - tr[..., None, None] * np.eye(3)
    dd = np.einsum("...ij,...ij->...", Dd, Dd)
    return np.sqrt(2.0 * dd + EPS_GAMMA * EPS_GAMMA)


def power_balance_identify(rec: dict, v_plate: float, frame_dt: float,
                           t_lo: float, t_hi: float, correct_eos: bool = False) -> dict:
    """Regress measured dissipation against the exact deviatoric-power columns
    (INT q/gd dV, INT q dV), q = 2 dev(D):dev(D), to recover (tau_y, eta).

    The 2D test assumes incompressibility (INT p div v = 0). The MPM dough is weakly
    compressible, so with correct_eos=True we add back the volumetric work term
    INT p div(v) dV. IMPORTANT: that p is the ORACLE Cauchy-trace pressure (s.cauchy()),
    so correct_eos is an ORACLE-STRESS correction, NOT an EOS-from-kinematics one -- a
    real load-cell/video pipeline does not have it unless pressure or density is separately
    observed or inferred. It is reported here as the upper bound on what removing the EOS
    bias buys; the deployable number is the correct_eos=False column."""
    times = np.asarray(rec["t"]); Fp = np.asarray(rec["F_plate"])
    Pg = np.asarray(rec["P_grav"]); KE = np.asarray(rec["KE"])
    X1 = np.asarray(rec["X1"]); X2 = np.asarray(rec["X2"]); Pvol = np.asarray(rec["Pvol"])
    n = len(times)
    rows = []
    for f in range(n):
        if not (t_lo <= times[f] <= t_hi):
            continue
        P_plate = v_plate * Fp[f]                  # plate power on dough; Fp = +Fz reaction
        dKE = (KE[min(f + 1, n - 1)] - KE[max(f - 1, 0)]) / (2 * frame_dt)
        diss = P_plate + Pg[f] - dKE + (Pvol[f] if correct_eos else 0.0)
        rows.append((X1[f], X2[f], diss, times[f], P_plate))
    R = np.array(rows)
    A = R[:, :2]; b = R[:, 2]
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    relres = float(np.linalg.norm(A @ theta - b) / max(np.linalg.norm(b), 1e-30))
    return {"tau_y_hat": float(theta[0]), "eta_hat": float(theta[1]),
            "n_times": int(R.shape[0]), "cond": float(np.linalg.cond(A.T @ A)),
            "fit_relres": relres, "window_t": [t_lo, t_hi],
            "_diss": b.tolist(), "_pred": (A @ theta).tolist(),
            "_t": R[:, 3].tolist()}


def run(n_grid=48, v_plate=0.08, eta=40.0, tau_y=200.0, density=1000.0, bulk=9.0e5,
        press_strain=0.5, dt=1.0e-4, substeps=20, render=True, render_every=3,
        out_name="squeeze_plate_franka.mp4", device="auto"):
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    col_w, col_d, col_h = 0.12, 0.12, 0.06
    pos, vol0, floor = block(grid, size=(col_w, col_d, col_h), ppc=2)
    s = Solver(grid=grid, device=device).load_particles(pos, vol0)
    s.set_material(newtonian(eta=eta, density=density, bulk_modulus=bulk).with_yield(tau_y))
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    cx = cy = grid.grid_lim * 0.5
    dough_top = floor + col_h
    plate_hx = 0.5 * col_w + 0.015
    plate_hy = 0.5 * col_d + 0.015
    plate_hz = 0.6 * grid.dx
    box_half = (plate_hx, plate_hy, plate_hz)
    T_layer = 3.0 * float(np.mean(vol0) ** (1.0 / 3.0))

    backend = WarpMPMBackend(solver=s)
    z = dough_top + plate_hz                          # plate bottom touching the dough top
    tool = backend.attach_tool((cx, cy, z), box_half, velocity=(0, 0, 0))

    frame_dt = dt * substeps
    t_press = press_strain * col_h / v_plate
    n_frames = round(t_press / frame_dt)

    # --- render setup (EE inversion so the gripper tip tracks the plate top) ----------
    arm = a_grid = ee_z = None
    ex0 = ey0 = z_off = 0.0
    if render:
        import matplotlib

        from warpmpm.adapters.mujoco_adapter import FrankaArm
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        arm = FrankaArm(height=620, width=820)
        a_grid = np.linspace(0.0, 1.0, 80)
        ee = np.array([arm.set_descent(float(a), frame_dt)["pos"] for a in a_grid])
        arm._prev_ee = None
        ee_z = ee[:, 2]
        ex0, ey0 = float(ee[len(ee) // 2, 0]), float(ee[len(ee) // 2, 1])
        z_off = float(np.interp(0.28, a_grid, ee_z)) - (z + plate_hz)
        table_z = floor + z_off
        arm.cam.lookat[:] = [ex0, ey0, table_z + 0.04]
        arm.cam.distance = 0.8
        arm.cam.azimuth = 138
        arm.cam.elevation = -14
        tmp = Path(tempfile.mkdtemp())

        def a_of(box_top_mpm):
            return float(np.interp(box_top_mpm + z_off, ee_z[::-1], a_grid[::-1]))

        def to_world(p):
            out = np.empty_like(p)
            out[:, 0] = ex0 - cx + p[:, 0]; out[:, 1] = ey0 - cy + p[:, 1]
            out[:, 2] = p[:, 2] + z_off
            return out

    rec = {k: [] for k in ("t", "F_plate", "F_stress", "P_grav", "KE", "X1", "X2", "Pvol",
                           "strain", "gap_mm")}
    prev_z = z
    rframe = 0
    for f in range(n_frames + 1):
        z_new = z - v_plate * frame_dt if f > 0 else z
        vz = (z_new - prev_z) / frame_dt
        if f > 0:
            backend.set_tool_kinematics(tool, center=(cx, cy, prev_z), velocity=(0, 0, vz))
            backend.reset_tool_force(tool)             # accumulate the impulse over this frame
            backend.step(dt, substeps)
        z = z_new; prev_z = z_new
        # per-frame scalars for the power-balance identification
        x = s.x(); v = s.v(); L = s.L(); cau = s.cauchy(); vol = s.vol()
        # PRIMARY plate reaction = the Newton-EXACT grid impulse the collider imposes:
        # F = sum_substeps sum_nodes m*(v_free - v_imposed) / frame_dt. No contact band, no
        # T_layer, no gating -- the calibrated force (this is what MuJoCo's wrist sensor would
        # read back). +Fz in compression, so P_plate = +v_plate*F_react.
        F_react = float(backend.get_tool_reaction(tool, frame_dt)[2]) if f > 0 else 0.0
        # diagnostic: the stress-integral surface-band estimator (the 2D method) for comparison
        m_xy = (np.abs(x[:, 0] - cx) < plate_hx) & (np.abs(x[:, 1] - cy) < plate_hy)
        z_top = float(np.percentile(x[m_xy, 2], 98)) if m_xy.any() else 0.0
        band = m_xy & (x[:, 2] > z_top - T_layer)
        szz = cau[:, 2, 2]
        F_stress = float(-np.sum(szz[band] * vol[band]) / T_layer) if band.sum() >= 5 else 0.0
        n_contact = int(band.sum())
        gd = equivalent_shear_rate(L)
        # EXACT kernel deviatoric power columns. The HB Kirchhoff stress is 2 eta_app dev(D) with
        # eta_app = eta + tau_y/gd, so the deviatoric stress power density is
        #   stress:dev(D) = 2 eta_app (dev(D):dev(D)) = eta_app * q,  q = 2 dev(D):dev(D) = gd^2 - eps^2.
        # Hence power = tau_y * (q/gd) + eta * q, i.e. the tau_y column is INT (q/gd) dV and the eta
        # column is INT q dV. (Using gd and gd^2 directly is only the eps->0 limit and slightly
        # over-counts the regularization floor at low shear.)
        q = np.maximum(gd ** 2 - EPS_GAMMA ** 2, 0.0)               # 2 dev(D):dev(D)
        p = -(cau[:, 0, 0] + cau[:, 1, 1] + cau[:, 2, 2]) / 3.0      # 3D-trace pressure (ORACLE stress)
        div_v = L[:, 0, 0] + L[:, 1, 1] + L[:, 2, 2]                 # tr(L) = div(v)
        rec["t"].append(f * frame_dt)
        rec["F_plate"].append(F_react)
        rec["F_stress"].append(F_stress)
        rec["P_grav"].append(float(np.sum(density * (-G_MAG) * v[:, 2] * vol)))
        rec["KE"].append(float(0.5 * density * np.sum(vol * np.sum(v ** 2, axis=1))))
        rec["X1"].append(float(np.sum((q / np.maximum(gd, 1e-12)) * vol)))   # tau_y column: INT (q/gd) dV
        rec["X2"].append(float(np.sum(q * vol)))                             # eta column:   INT q dV
        # NOTE: Pvol uses the ORACLE Cauchy-trace pressure p; the correct_eos path below is therefore
        # an ORACLE-STRESS correction, NOT an EOS-from-kinematics one. A load-cell pipeline cannot
        # form INT p div(v) unless pressure/density is separately observed or inferred (see report).
        rec["Pvol"].append(float(np.sum(p * div_v * vol)))           # INT p div(v) dV (oracle p)
        plate_bottom = z - plate_hz
        rec["strain"].append((col_h - (plate_bottom - floor)) / col_h)
        rec["gap_mm"].append((plate_bottom - floor) * 1e3)
        if render and (f % render_every == 0):
            arm.set_descent(a_of(z + plate_hz), frame_dt, track_camera=False)
            spd = np.linalg.norm(v, axis=1)
            col = colormaps["YlOrBr_r"](np.clip(spd / 0.25, 0, 1)); col[:, 3] = 1.0
            plate_world = (ex0, ey0, z + z_off)
            plate_box = (plate_world, (plate_hx, plate_hy, plate_hz),
                         (0.7, 0.72, 0.78, 1.0))
            rgb = arm.render_with_particles(to_world(x), col, radius=0.0033,
                                            table=(ex0, ey0, table_z, 0.26),
                                            boxes=[plate_box])
            fig = plt.figure(figsize=(8.2, 6.2), facecolor="black")
            ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(rgb); ax.axis("off")
            ax.text(0.02, 0.97, "Franka plate squeeze  (newtonian dough: "
                    f"eta={eta:.0f} Pa.s, tau_y={tau_y:.0f} Pa)", color="w",
                    fontsize=11, transform=ax.transAxes, va="top")
            ax.text(0.02, 0.92, f"v_plate = {v_plate*1e3:.0f} mm/s   strain = "
                    f"{rec['strain'][-1]*100:4.1f}%   F_plate = {abs(F_react):5.1f} N",
                    color="#ffd27f", fontsize=11, transform=ax.transAxes, va="top")
            fig.savefig(tmp / f"f_{rframe:04d}.png", dpi=110, facecolor="black")
            plt.close(fig); rframe += 1
        if f % 10 == 0:
            print(f"frame {f:3d}/{n_frames} strain={rec['strain'][-1]*100:4.1f}% "
                  f"F_grid={abs(F_react):6.1f}N (stress {abs(F_stress):6.1f}N) "
                  f"gap={rec['gap_mm'][-1]:5.1f}mm nc={n_contact}")

    # --- identification + comparison --------------------------------------------------
    t_lo, t_hi = 0.12 * t_press, 0.92 * t_press
    ident = power_balance_identify(rec, v_plate, frame_dt, t_lo, t_hi, correct_eos=False)
    ident_eos = power_balance_identify(rec, v_plate, frame_dt, t_lo, t_hi, correct_eos=True)
    res = {
        "tau_y_true": tau_y, "eta_true": eta, "v_plate": v_plate, "n_grid": n_grid,
        "device": device,
        # matched to the 2D method (incompressible assumption)
        "tau_y_hat": ident["tau_y_hat"], "eta_hat": ident["eta_hat"],
        "tau_y_err": abs(ident["tau_y_hat"] - tau_y) / tau_y,
        "eta_err": abs(ident["eta_hat"] - eta) / eta,
        # EOS-corrected (adds INT p div v): the volumetric work the 2D method drops
        "tau_y_hat_eos": ident_eos["tau_y_hat"], "eta_hat_eos": ident_eos["eta_hat"],
        "tau_y_err_eos": abs(ident_eos["tau_y_hat"] - tau_y) / tau_y,
        "eta_err_eos": abs(ident_eos["eta_hat"] - eta) / eta,
        "n_times": ident["n_times"], "cond": ident["cond"],
        "fit_relres": ident["fit_relres"], "fit_relres_eos": ident_eos["fit_relres"],
        "ref_2d": REF_2D,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "squeeze_plate_franka.json").write_text(json.dumps(res, indent=2, default=float))

    if render:
        _figure(rec, ident, res, OUT / "squeeze_plate_franka_ident.png")
        mp4 = OUT / out_name
        subprocess.run(["ffmpeg", "-y", "-framerate", "16", "-i", str(tmp / "f_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)],
                       check=True, capture_output=True)
        res["mp4"] = str(mp4); res["fig"] = str(OUT / "squeeze_plate_franka_ident.png")
        print("wrote", mp4)

    print(f"\n[arm-plate squeeze]   truth (tau_y, eta) = ({tau_y:.0f}, {eta:.0f})   "
          f"2D ref ({REF_2D['tau_y_hat']:.0f}, {REF_2D['eta_hat']:.0f})")
    print(f"  matched-2D method  (incompressible): ({res['tau_y_hat']:.0f}, "
          f"{res['eta_hat']:.1f})   err ({res['tau_y_err']*100:.0f}%, {res['eta_err']*100:.0f}%)")
    print(f"  EOS-corrected (+INT p div v)       : ({res['tau_y_hat_eos']:.0f}, "
          f"{res['eta_hat_eos']:.1f})   err ({res['tau_y_err_eos']*100:.0f}%, "
          f"{res['eta_err_eos']*100:.0f}%)")
    print(f"  n={res['n_times']}  cond={res['cond']:.0f}  fit_relres={res['fit_relres']:.2f}")
    return res


def _figure(rec, ident, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
    st = np.array(rec["strain"]) * 100
    a0.plot(st, np.abs(rec["F_stress"]), color="#adb5bd", lw=1, ls="--",
            label="stress-integral (2D method)")
    a0.plot(st, np.abs(rec["F_plate"]), color="#d9480f", lw=2,
            label="grid-impulse (Newton-exact)")
    a0.set_xlabel("engineering strain  (%)"); a0.set_ylabel("plate reaction force  |F_z|  (N)")
    a0.set_title("Arm-mounted plate: measured squeeze force")
    a0.legend(fontsize=8, loc="upper left"); a0.grid(alpha=0.3)
    a1.scatter(ident["_diss"], ident["_pred"], s=10, color="#1c7ed6", alpha=0.7)
    lim = [min(ident["_diss"]), max(ident["_diss"])]
    a1.plot(lim, lim, "k--", lw=1)
    a1.set_xlabel("measured dissipation  (W)"); a1.set_ylabel("fit  tau_y*X1 + eta*X2  (W)")
    a1.set_title(f"power-balance fit  (relres {ident['fit_relres']:.2f})"); a1.grid(alpha=0.3)
    txt = (f"recovered (tau_y, eta) = ({res['tau_y_hat']:.0f}, {res['eta_hat']:.1f})\n"
           f"ground truth          = ({res['tau_y_true']:.0f}, {res['eta_true']:.0f})\n"
           f"2D test recovered     = ({res['ref_2d']['tau_y_hat']:.0f}, "
           f"{res['ref_2d']['eta_hat']:.0f})\n"
           f"err = ({res['tau_y_err']*100:.0f}%, {res['eta_err']*100:.0f}%)")
    a1.text(0.03, 0.97, txt, transform=a1.transAxes, va="top", fontsize=9,
            family="monospace", bbox=dict(boxstyle="round", fc="#fff3bf", ec="0.6"))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    parser.add_argument("--no-render", action="store_true", help="skip video/figure rendering")
    args = parser.parse_args()
    run(device=args.device, render=not args.no_render)
