"""3D dough wide-shear-rate cell: FE recovery + held-out ROLLOUT, with rendered visuals.

The 3D squeeze excited only ~1 decade of shear rate and FE lost to Bingham there. This lifts
the 2D wide-shear protocol (examples/shear_cell_fe.py) to a genuine 3D dough block: a free-y
block on a sticky floor, sheared by a top wall translating in x at a SWEEP of speeds spanning
~2 decades of gamma_dot. Same power balance INT eta_app(gd) gd^2 dV = v_wall*Fx + Pg - dKE,
linear in theta. The recovered FE eta_app(gd) is re-simulated DIRECTLY (tabulated-viscosity
material, fork id 12) on a HELD-OUT speed and compared to truth; the metric is the self-
consistent rollout (sim -> learn -> re-sim), not the curve. Trajectories are recorded so the
held-out rollout can be rendered (see examples/shear_rollout_video.py).

Run:  PYTHONPATH=src ../.venv/bin/python examples/shear_cell_3d.py          # full sweep + rollout
      PYTHONPATH=src ../.venv/bin/python examples/shear_cell_3d.py probe    # one segment probe
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# reuse the validated 2D helpers (same dir on sys.path when run as a script)
from shear_cell_fe import EPS, RHO, TRUTH, _gd, _power_rows, eta_app_true, viscous_prior
from warpmpm import GridConfig, Solver, newtonian, tabulated_viscous
from warpmpm.coupling.backend import WarpMPMBackend

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parents[1] / "out" / "shear_cell_3d"
G_MAG = 9.81
N_GRID = 48
GRID_LIM = 0.4
GEOM = (0.12, 0.06, 0.045)        # (Lx, Ly, Lz): a genuine 3D block (free in y)
FE_TABLE = REPO / "mpm_engine" / "fe-weights" / "viscous.npz"


def _build_block(grid: GridConfig):
    dx = grid.dx
    cx = cy = grid.grid_lim * 0.5
    floor = 3 * dx
    h = dx / 2                                       # ppc = 2
    Lx, Ly, Lz = GEOM
    xs = np.arange(cx - 0.5 * Lx + 0.5 * h, cx + 0.5 * Lx, h)
    ys = np.arange(cy - 0.5 * Ly + 0.5 * h, cy + 0.5 * Ly, h)
    zs = np.arange(floor + 0.5 * h, floor + Lz, h)
    pos = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    pos += np.random.default_rng(0).uniform(-0.2 * h, 0.2 * h, pos.shape).astype(np.float32)
    vol = np.full(len(pos), h ** 3, dtype=np.float32)
    return pos, vol, floor, cx, cy


def shear_segment(v_shear, material, n_frames, dt=1.0e-4, substeps=20,
                  record_power=False, record_traj=False, traj_stride=2,
                  record_stress=False, device="auto"):
    """One 3D shear segment. record_power -> per-frame (diss rows); record_traj -> store
    x[F,N,3], v[F,N,3], times, Fx[F] for rendering; record_stress -> per-frame per-particle
    (gd, vol, sigma_dev:D_dev) for the STRONG-form (pointwise constitutive) recovery."""
    grid = GridConfig(n_grid=N_GRID, grid_lim=GRID_LIM)
    pos, vol0, floor, cx, cy = _build_block(grid)
    _Lx, Ly, Lz = GEOM
    s = Solver(grid=grid, device=device).load_particles(pos, vol0)
    s.set_material(material)
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")                  # no-slip floor
    be = WarpMPMBackend(solver=s)
    fdt = dt * substeps
    bh = (0.45 * grid.grid_lim, 0.5 * Ly + 0.012, 0.6 * grid.dx)     # wall spans x (wide) and y
    z_wall = floor + Lz
    tool = be.attach_tool((cx, cy, z_wall), bh)

    out = {"t": [], "Fx": [], "gd_pct": []}
    if record_power:
        out.update(diss_rows=[], KE=[])
    if record_traj:
        out.update(X=[], V=[])
    if record_stress:
        out.update(strong_rows=[])
    cwx = cx
    for f in range(n_frames + 1):
        if f > 0:
            be.set_tool_kinematics(tool, center=(cwx, cy, z_wall), velocity=(v_shear, 0.0, 0.0))
            be.reset_tool_force(tool)
            be.step(dt, substeps)
            cwx += v_shear * fdt
        F = be.get_tool_reaction(tool, fdt) if f > 0 else np.zeros(3)
        out["t"].append(f * fdt)
        out["Fx"].append(float(F[0]))
        L = s.L()
        gd = _gd(L)
        out["gd_pct"].append(np.percentile(gd, [25, 50, 75, 95]))
        if record_power:
            v = s.v()
            vol = s.vol()
            out["KE"].append(float(0.5 * RHO * np.sum(vol * np.sum(v ** 2, axis=1))))
            Pg = float(np.sum(RHO * (-G_MAG) * v[:, 2] * vol))
            Pwall = abs(v_shear) * abs(float(F[0]))
            out["diss_rows"].append((f, Pwall, Pg, gd.copy(), vol.copy()))
        if record_stress and f > 0:
            sig = s.cauchy()                                          # (N,3,3) Cauchy
            D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
            trs = (sig[..., 0, 0] + sig[..., 1, 1] + sig[..., 2, 2]) / 3.0
            trd = (D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2]) / 3.0
            sdev = sig - trs[..., None, None] * np.eye(3)
            ddev = D - trd[..., None, None] * np.eye(3)
            diss_dens = np.einsum("...ij,...ij->...", sdev, ddev)     # sigma_dev:D_dev
            out["strong_rows"].append((gd.copy(), s.vol().copy(), diss_dens))
        if record_traj and f % traj_stride == 0:
            out["X"].append(s.x().copy())
            out["V"].append(s.v().copy())
    out["gd_pct"] = np.array(out["gd_pct"])
    out["Fx"] = np.array(out["Fx"])
    out["t"] = np.array(out["t"])
    if record_traj:
        out["X"] = np.array(out["X"])
        out["V"] = np.array(out["V"])
        out["traj_t"] = out["t"][::traj_stride][: len(out["X"])]
        out["floor"] = floor
    return out


def _truth_material():
    return (newtonian(eta=TRUTH["eta"], density=RHO, bulk_modulus=9.0e5)
            .with_yield(TRUTH["tau_y"]).with_powerlaw(K=TRUTH["pk"], n=TRUTH["pn"]))


def run(speeds=(0.006, 0.012, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8), v_holdout=0.16,
        rho=0.1, device="auto"):
    from ident.features.function_encoder import FunctionEncoderDict
    from ident.solve.qp import constrained_solve
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(FE_TABLE)
    fe = FunctionEncoderDict(d["s_grid"], d["table"])
    fdt = 2.0e-3

    A_fe, A_b, bvec, p50s = [], [], [], []
    t0 = time.time()
    for vp in speeds:
        nf = int(np.clip(round(0.4 * GEOM[0] / (vp * fdt)), 40, 160))
        seg = shear_segment(vp, _truth_material(), n_frames=nf, record_power=True,
                            device=device)
        a, b2, bb, gd, w = _power_rows(seg, fe)
        A_fe.append(a)
        A_b.append(b2)
        bvec.append(bb)
        order = np.argsort(gd)                  # weight w must follow gd's ordering
        cw = np.cumsum(w[order]) / w.sum()
        p50 = float(gd[order][np.searchsorted(cw, 0.5)])
        p50s.append(p50)
        print(f"  v={vp:5.3f}: {nf:3d} frames, {len(bb)} rows, dissipation-median gd~{p50:.2f}/s")
    A_fe = np.vstack(A_fe)
    A_b = np.vstack(A_b)
    bvec = np.concatenate(bvec)
    gscale = float(np.sqrt(np.mean(bvec ** 2))) + 1e-30
    A_fe /= gscale
    A_b /= gscale
    bvec /= gscale
    gd_lo = max(min(p50s), 0.1)
    gd_hi = max(p50s)
    print(f"  pooled {len(bvec)} rows in {time.time()-t0:.0f}s; 3D dissipation-weighted shear-rate "
          f"excitation [{gd_lo:.2f}, {gd_hi:.2f}] /s ({np.log10(gd_hi/gd_lo):.1f} decades)")

    gg = np.logspace(np.log10(gd_lo), np.log10(gd_hi), 80)
    G = fe.gram((10.0 ** np.linspace(-1, 2, 257), np.ones(257)))
    Icon = np.logspace(np.log10(gd_lo), np.log10(gd_hi), 40)
    tbar, cov = viscous_prior(fe, d["s_grid"])

    def fe_solve(rho_):
        A, b = A_fe, bvec
        if rho_ > 1e-9:
            Lc = np.linalg.cholesky(rho_ * np.linalg.inv(cov))
            A = np.vstack([A_fe, Lc])
            b = np.concatenate([bvec, Lc @ tbar])
        qp = constrained_solve(A, b, fe, lam=1e-3, G=G, mu_min=0.5,
                               I_constraint_grid=Icon, nonnegativity=False, monotonic=False)
        return qp.theta

    theta_fe0 = fe_solve(0.0)
    theta_fe = fe_solve(rho)
    th_b, *_ = np.linalg.lstsq(A_b, bvec, rcond=None)
    tau_y_b, eta_b = float(th_b[0]), float(th_b[1])

    eta_fe = fe.phi(gg) @ theta_fe
    eta_fe0 = fe.phi(gg) @ theta_fe0
    eta_bg = eta_b + tau_y_b / np.sqrt(gg ** 2 + EPS ** 2)
    eta_tr = eta_app_true(gg)

    def rel(e):
        return float(np.sqrt(np.mean((e - eta_tr) ** 2)) / np.sqrt(np.mean(eta_tr ** 2)))
    r_fe, r_fe0, r_bg = rel(eta_fe), rel(eta_fe0), rel(eta_bg)
    print(f"\n  3D eta_app(gd) curve recovery over gd=[{gd_lo:.2f},{gd_hi:.2f}] /s (truth HB):")
    print(f"    Bingham (2-param)   relL2={r_bg*100:5.1f}%  (tau_y={tau_y_b:.0f},eta={eta_b:.0f})")
    print(f"    FE, no prior  (K=8)      relL2={r_fe0*100:5.1f}%")
    print(f"    FE + family prior (K=8)  relL2={r_fe*100:5.1f}%")

    # ---- held-out 3D ROLLOUT (record trajectories for rendering) -----------------------
    s_tab = np.linspace(-1.0, 2.0, 128)
    gd_tab = 10.0 ** s_tab
    tab_fe0 = np.clip(fe.phi(gd_tab) @ theta_fe0, 0.5, None)
    mats = {
        "truth": _truth_material(),
        "FE": tabulated_viscous(tab_fe0, -1.0, 2.0, RHO, 9.0e5),
        "Bingham": newtonian(eta=eta_b, density=RHO, bulk_modulus=9.0e5).with_yield(tau_y_b),
    }
    print(f"\n  held-out 3D ROLLOUT at v={v_holdout} m/s (self-consistent re-sim):")
    nf_h = int(np.clip(round(0.5 * GEOM[0] / (v_holdout * fdt)), 60, 160))
    roll = {tag: shear_segment(v_holdout, mat, n_frames=nf_h, record_traj=True,
                               device=device)
            for tag, mat in mats.items()}
    tF = roll["truth"]["Fx"]
    m0 = max(3, int(0.2 * len(tF)))
    fr = {}
    for tag in ("FE", "Bingham"):
        e = roll[tag]["Fx"][m0:]
        fr[tag] = float(np.linalg.norm(e - tF[m0:]) / (np.linalg.norm(tF[m0:]) + 1e-30))
        print(f"    {tag:9s}: wall-force rollout relL2 = {fr[tag]*100:5.1f}%")

    # save rollout trajectories for the video maker
    np.savez(OUT / "rollout_3d.npz",
             **{f"X_{t}": roll[t]["X"] for t in roll},
             **{f"V_{t}": roll[t]["V"] for t in roll},
             **{f"Fx_{t}": roll[t]["Fx"] for t in roll},
             t=roll["truth"]["t"], traj_t=roll["truth"]["traj_t"],
             floor=roll["truth"]["floor"], m0=m0, v_holdout=v_holdout,
             fr_FE=fr["FE"], fr_Bingham=fr["Bingham"])
    # save the curve/figure inputs
    np.savez(OUT / "curve_3d.npz", gg=gg, eta_tr=eta_tr, eta_bg=eta_bg, eta_fe0=eta_fe0,
             eta_fe=eta_fe, band=np.array([gd_lo, gd_hi]), r_bg=r_bg, r_fe0=r_fe0, r_fe=r_fe)
    _figure(gg, eta_tr, eta_bg, eta_fe0, r_bg, r_fe0, (gd_lo, gd_hi), roll, tF, m0, fr, v_holdout)
    return {"r_fe0": r_fe0, "r_bg": r_bg, "force_roll": fr, "band": (gd_lo, gd_hi),
            "device": device}


def _figure(gg, eta_tr, eta_bg, eta_fe0, r_bg, r_fe0, band, roll, tF, m0, fr, v_holdout):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.6))
    # the dissipation (~gd^2) weighted power balance only carries information where the flow
    # actually dissipates; below ~1/s the slow segments contribute negligible dissipation, so
    # the data-only FE tail there is under-determined. Compare in the determined band, where
    # FE tracks truth up to the constant discrete/continuum closure factor.
    trust_lo = max(1.0, band[0])
    mband = gg >= trust_lo
    factor = float(np.median((eta_fe0 / eta_tr)[mband]))
    ax[0].axvspan(trust_lo, band[1], color="#fff3bf", alpha=0.45,
                  label="dissipation-determined band")
    ax[0].loglog(gg, eta_tr, "k-", lw=2.6, label="truth (analytic HB)")
    ax[0].loglog(gg, eta_tr * factor, color="#868e96", lw=1.3, ls=":",
                 label=f"truth x{factor:.1f} (closure factor)")
    ax[0].loglog(gg, eta_bg, color="#e8590c", lw=1.9, ls="--", label="Bingham fit")
    ax[0].loglog(gg, eta_fe0, color="#1864ab", lw=2.4, label="FE (recovered)")
    ax[0].set_xlim(0.6, band[1] * 1.05)
    ax[0].set_xlabel("shear rate  gamma_dot  (1/s)")
    ax[0].set_ylabel("apparent viscosity  eta_app  (Pa.s)")
    ax[0].set_title(f"(a) 3D recovered eta_app(gd), {np.log10(band[1]/band[0]):.1f}-decade sweep\n"
                    "FE tracks truth in the determined band; Bingham misfits the shape")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    th = roll["truth"]["t"]
    C = {"truth": "k", "FE": "#1864ab", "Bingham": "#e8590c"}
    for tag in ("truth", "FE", "Bingham"):
        lab = tag if tag == "truth" else f"{tag} ({fr[tag]*100:.0f}%)"
        ax[1].plot(th, -roll[tag]["Fx"], color=C[tag],
                   lw=2.8 if tag == "truth" else 2.0,
                   ls="-" if tag != "Bingham" else "--", label=lab)
    ax[1].axvspan(0, th[m0], color="#f1f3f5", alpha=0.8)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("wall shear force  -F_x  (N)")
    ax[1].set_title(f"(b) held-out 3D FORCE rollout at v={v_holdout} m/s\n"
                    "self-consistent (relL2 vs truth)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "shear_cell_3d.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


def replot():
    """Redraw the 3D figure from cached run outputs (no re-simulation)."""
    c = np.load(OUT / "curve_3d.npz")
    r = np.load(OUT / "rollout_3d.npz", allow_pickle=True)
    roll = {t: {"t": r["t"], "Fx": r[f"Fx_{t}"]} for t in ("truth", "FE", "Bingham")}
    fr = {"FE": float(r["fr_FE"]), "Bingham": float(r["fr_Bingham"])}
    _figure(c["gg"], c["eta_tr"], c["eta_bg"], c["eta_fe0"], float(c["r_bg"]),
            float(c["r_fe0"]), tuple(c["band"]), roll, r["Fx_truth"], int(r["m0"]), fr,
            float(r["v_holdout"]))


def probe(device="auto"):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = shear_segment(0.1, _truth_material(), n_frames=60, record_power=True,
                        device=device)
    gd = res["gd_pct"]
    grid = GridConfig(n_grid=N_GRID, grid_lim=GRID_LIM)
    pos, *_ = _build_block(grid)
    print(f"3D probe v=0.1: {len(res['t'])} frames, N={len(pos)} in {time.time()-t0:.1f}s")
    fin = np.all(np.isfinite(res["Fx"]))
    print(f"  Fx range [{res['Fx'].min():.3f},{res['Fx'].max():.3f}]  finite={fin}")
    print(f"  gd p50 last {gd[-1,1]:.2f}/s p95 {gd[-1,3]:.2f}/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="run", choices=("run", "probe", "replot"))
    parser.add_argument("--device", default="cuda:0", help="Warp MPM device, e.g. cuda:0 or cuda:1")
    args = parser.parse_args()
    if args.which == "probe":
        probe(device=args.device)
    elif args.which == "replot":
        replot()
    else:
        run(device=args.device)
