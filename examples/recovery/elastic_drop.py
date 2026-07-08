"""Hyperelastic gravity-drop: learn the elastic stiffness from a bounce, no backpropagation.

NCLaw learns an elastic law by differentiating an MPM rollout (BPTT on positions). We do the
elastic analogue of our granular weak-form recovery: drop a fixed-corotated (neo-Hookean-style)
blob, observe its bounce (positions, velocities, the deformation gradient F per particle per
frame), and recover the elastic moduli (mu, lambda), the stiffness, by a convex,
linear-in-theta weak-form momentum-residual solve. The simulator is never differentiated.

Key idea (docs: the elastic case of the weak form):
  the first Piola-Kirchhoff stress of the FCR model is linear in the moduli,
      P = mu * [2 (F - R)] + lambda * [J (J-1) F^{-T}]  ==  mu P_mu(F) + lambda P_lambda(F),
  with R the rotation from polar(F). Plugged into the dynamic weak form (reference config),
      sum_p V0_p P : grad_X w_j  =  - sum_p V0_p rho0 (a_p - g) . w_j     (interior bumps),
  the inertia term rho0*a (known from the observed trajectory) supplies the absolute force
  scale, so the moduli are pinned without any force sensor; the bounce is the load cell.
  Two unknowns, many (test function x frame) rows -> over-determined convex least squares.

Run:  python -m examples.recovery.elastic_drop            # truth drop -> recover -> re-sim -> compare
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

wp.config.quiet = True
wp.init()
DEVICE = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"  # auto, like Solver(device="auto")
OUT = ROOT / "out" / "elastic_drop"
TRUTH = dict(E=2.0e5, nu=0.30, rho=1000.0)


def _raw_points(shape, size, h):
    """Particle cloud for a shape, centred near the origin (placed later). `size` is the
    characteristic half-extent. Shapes: sphere, box (rectangular blob), star (5-point star in
    the x-z plane, extruded thin in y; a qualitatively different, non-convex geometry)."""
    g = np.arange(-size - h, size + h, h)
    P = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3).astype(np.float64)
    if shape == "sphere":
        keep = np.linalg.norm(P, axis=1) <= size
    elif shape == "box":
        a = np.array([size, size, 0.85 * size])           # a rectangular (non-cubic) blob
        keep = np.all(np.abs(P) <= a, axis=1)
    elif shape == "star":
        from matplotlib.path import Path as MplPath
        R, rin, ty = size, 0.42 * size, 0.55 * size       # outer/inner radius, y half-thickness
        ang = np.pi / 2 + np.arange(10) * np.pi / 5
        rad = np.where(np.arange(10) % 2 == 0, R, rin)
        verts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])   # star in (x,z)
        inside = MplPath(verts).contains_points(P[:, [0, 2]])
        keep = inside & (np.abs(P[:, 1]) <= ty)
    else:
        raise ValueError(shape)
    return P[keep]


def run_drop(E, nu, rho, out_path, shape="sphere", size=0.14, drop_gap=0.18,
             n_grid=48, grid_lim=1.0, t_end=0.9, frame_dt=2.0e-3, floor_bc="slip", log=print):
    import warp as wp
    wp.config.quiet = True
    wp.init()
    from warpmpm.kernels import MPM_Simulator_WARP
    import torch

    dx = grid_lim / n_grid
    h = dx / 2
    floor = 3 * dx
    cx = cy = grid_lim * 0.5
    pos = _raw_points(shape, size, h)
    pos += np.random.default_rng(0).uniform(-0.15 * h, 0.15 * h, pos.shape)
    pos[:, 0] += cx - pos[:, 0].mean()                   # centre x,y; base drop_gap above floor
    pos[:, 1] += cy - pos[:, 1].mean()
    pos[:, 2] += (floor + drop_gap) - pos[:, 2].min()
    pos = pos.astype(np.float32)
    vol = np.full(len(pos), h ** 3, dtype=np.float32)
    X_ref = pos.copy()                                   # reference (undeformed) config

    c_el = float(np.sqrt(E / rho))
    substeps = int(np.ceil(frame_dt / (0.3 * dx / c_el)))
    dt = frame_dt / substeps
    n_frames = int(round(t_end / frame_dt))
    log(f"[elastic] {shape} N={len(pos)} grid={n_grid}^3 dx={dx*1e3:.1f}mm c={c_el:.0f}m/s "
        f"dt={dt:.1e} sub={substeps} frames={n_frames} E={E:.2e} nu={nu}")

    s = MPM_Simulator_WARP(len(pos), device=DEVICE)
    s.load_initial_data_from_torch(
        torch.from_numpy(np.ascontiguousarray(pos)),
        torch.from_numpy(np.ascontiguousarray(vol)),
        n_grid=n_grid, grid_lim=grid_lim, device=DEVICE)
    s.set_parameters_dict(
        {"material": "jelly", "E": E, "nu": nu, "density": rho, "g": [0.0, 0.0, -9.81]},
        device=DEVICE)
    s.finalize_mu_lam(device=DEVICE)
    s.add_surface_collider((0.0, 0.0, floor), (0.0, 0.0, 1.0), floor_bc)

    X, V, F = [], [], []
    t0 = time.time()
    step = 0
    for frame in range(n_frames + 1):
        x = s.export_particle_x_to_torch().detach().cpu().numpy().copy()
        v = s.export_particle_v_to_torch().detach().cpu().numpy().copy()
        f = s.export_particle_F_to_torch().detach().cpu().numpy().copy()
        if not np.isfinite(x).all():
            log(f"[elastic] NaN at frame {frame}"); break
        X.append(x); V.append(v); F.append(f)
        if frame == n_frames:
            break
        for _ in range(substeps):
            s.p2g2p(step, dt, device=DEVICE); step += 1
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, x=np.asarray(X, np.float32), v=np.asarray(V, np.float32),
             F=np.asarray(F, np.float32), X_ref=X_ref.astype(np.float32),
             vol=vol, frame_dt=frame_dt, rho=rho, g=np.array([0.0, 0.0, -9.81]),
             floor=floor, E=E, nu=nu, mu=mu, lam=lam, shape=shape, size=size, grid_lim=grid_lim)
    log(f"[elastic] wrote {out_path} ({len(X)} frames, {time.time()-t0:.0f}s)  mu={mu:.3e} lam={lam:.3e}")
    return out_path


def _polar_R(F):
    """Rotation from polar decomposition of a batch of F (M,3,3), det>0 enforced."""
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    bad = np.linalg.det(R) < 0
    if bad.any():
        U[bad, :, -1] *= -1.0
        R = U @ Vt
    return R


def _moduli_to_E_nu(mu, lam):
    nu = lam / (2.0 * (lam + mu))
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    return float(E), float(nu)


def recover(dump_path, n_modes=4, log=print):
    """Convex weak-form recovery of the elastic moduli (mu, lambda) from the bounce.

    P(F) = mu * 2(F-R) + lambda * J(J-1) F^{-T}  is linear in (mu, lambda). Per frame the
    dynamic momentum balance  sum_p V0 P:grad_X w_j = - sum_p V0 rho0 (a_p - g).w_j  gives two
    columns and one row per (test function, frame); we stack and least-squares for (mu, lambda).
    Interior reference bumps (zero on the surface) make boundary tractions vanish; the inertia
    term sets the absolute scale. No simulator differentiation.
    """
    d = np.load(dump_path)
    X = d["x"].astype(np.float64)                                  # (T,N,3) current positions
    v = d["v"].astype(np.float64)
    F = d["F"].astype(np.float64).reshape(X.shape[0], X.shape[1], 3, 3)
    Xr = d["X_ref"].astype(np.float64)                             # (N,3) reference config
    V0 = d["vol"].astype(np.float64)                               # (N,) reference volume
    rho0 = float(d["rho"]); g = d["g"].astype(np.float64)
    fdt = float(d["frame_dt"]); T = X.shape[0]
    mu_t, lam_t = float(d["mu"]), float(d["lam"])

    # central-difference material acceleration, frames 1..T-2
    a = (v[2:] - v[:-2]) / (2.0 * fdt)                             # (T-2,N,3)
    Frng = np.arange(1, T - 1)

    # interior reference bumps: radial window (zero on the sphere surface) x low-order modes
    c = Xr.mean(0); r = np.linalg.norm(Xr - c, axis=1); Rb = r.max() * 1.02
    u = r / Rb
    W = np.clip(1.0 - u ** 2, 0.0, None) ** 2
    dWdr = -4.0 * u / Rb * np.clip(1.0 - u ** 2, 0.0, None)        # dW/dr
    rsafe = np.where(r > 1e-9, r, 1.0)
    gradW = (dWdr / rsafe)[:, None] * (Xr - c)                     # (N,3)
    dX = (Xr - c) / Rb
    modes = [np.ones(len(Xr))] + [dX[:, i] for i in range(3)][: max(0, n_modes - 1)]
    gmodes = [np.zeros((len(Xr), 3))] + [np.eye(3)[i][None, :] / Rb * np.ones((len(Xr), 1))
                                         for i in range(3)][: max(0, n_modes - 1)]
    phis = [W * m for m in modes]                                  # (N,) each
    gphis = [gradW * m[:, None] + W[:, None] * gm for m, gm in zip(modes, gmodes)]  # (N,3) each

    rows_A, rows_b = [], []
    cov = []
    for ti, t in enumerate(Frng):
        Ft = F[t]
        # rotation-invariant validity filter: Hencky strain ||log(sig)|| not ||F - I||
        # (frame-objective; robust to the benign SVD-branch rotation that can differ
        # between MPM backends when storing F).
        sig_t = np.clip(np.linalg.svd(Ft, compute_uv=False), 1e-9, None)
        J = sig_t.prod(-1); devF = np.linalg.norm(np.log(sig_t), axis=1)
        m = (J > 0.3) & (J < 2.0) & (devF < 0.8) & np.isfinite(J)   # drop pathological surface pts
        if m.sum() < 50:
            continue
        cov.append(devF[m])
        Fm = Ft[m]; Jm = J[m]
        R = _polar_R(Fm)
        Pmu = 2.0 * (Fm - R)                                       # (Mm,3,3)
        Finvt = np.transpose(np.linalg.inv(Fm), (0, 2, 1))
        Plam = (Jm * (Jm - 1.0))[:, None, None] * Finvt
        Vm = V0[m]
        at = a[ti][m]                                              # (Mm,3)
        for dirn in range(3):
            for phi, gphi in zip(phis, gphis):
                gp = gphi[m]; ph = phi[m]
                a_mu = np.sum(Vm * np.einsum("pj,pj->p", Pmu[:, dirn, :], gp))
                a_lam = np.sum(Vm * np.einsum("pj,pj->p", Plam[:, dirn, :], gp))
                bb = -np.sum(Vm * rho0 * (at[:, dirn] - g[dirn]) * ph)
                rows_A.append([a_mu, a_lam]); rows_b.append(bb)
    A = np.asarray(rows_A); b = np.asarray(rows_b)
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    mu_h, lam_h = float(theta[0]), float(theta[1])
    E_h, nu_h = _moduli_to_E_nu(mu_h, lam_h)
    E_t, nu_t = _moduli_to_E_nu(mu_t, lam_t)
    cond = float(np.linalg.cond(A.T @ A))
    covv = np.concatenate(cov)
    log(f"[recover] rows={A.shape[0]}  cond(A^T A)={cond:.1f}  strain coverage ||F-I|| "
        f"[{covv.min():.3f}, {np.percentile(covv,99):.3f}]")
    log(f"[recover] mu:  {mu_h:.4e}  (truth {mu_t:.4e},  {100*abs(mu_h/mu_t-1):.1f}% err)")
    log(f"[recover] lam: {lam_h:.4e}  (truth {lam_t:.4e},  {100*abs(lam_h/lam_t-1):.1f}% err)")
    log(f"[recover] E:   {E_h:.4e}  (truth {E_t:.4e},  {100*abs(E_h/E_t-1):.1f}% err)   "
        f"nu: {nu_h:.3f} (truth {nu_t:.3f})")
    return dict(mu=mu_h, lam=lam_h, E=E_h, nu=nu_h, E_err=abs(E_h / E_t - 1),
                mu_err=abs(mu_h / mu_t - 1), cond=cond)


def _pos_err(tp, rp):
    t = np.load(tp); r = np.load(rp)
    nf = min(t["x"].shape[0], r["x"].shape[0]); n = min(t["x"].shape[1], r["x"].shape[1])
    rmse_mm = float(np.sqrt(((t["x"][:nf, :n] - r["x"][:nf, :n]) ** 2).sum(-1).mean()) * 1e3)
    gl = float(t["grid_lim"])
    mse_box = float((((t["x"][:nf, :n] - r["x"][:nf, :n]) / gl) ** 2).sum(-1).mean())
    return rmse_mm, mse_box, nf


def shape_generalization():
    """Learn the elastic law on a rectangular blob drop, then predict a star neo-Hookean blob
    (held-out, qualitatively different geometry), the elastic analogue of the sand
    bunny/dragon generalization. No backprop; recovery is a convex solve."""
    OUT.mkdir(parents=True, exist_ok=True)
    E, nu, rho = TRUTH["E"], TRUTH["nu"], TRUTH["rho"]

    # 1) learn on the rectangular blob
    box = OUT / "box_truth.npz"
    run_drop(E, nu, rho, box, shape="box", size=0.16, drop_gap=0.18)
    rec = recover(box)

    # 2) predict the star: truth-law vs recovered-law re-sim on the same star cloud (1:1 particles)
    star_t = run_drop(E, nu, rho, OUT / "star_truth.npz", shape="star", size=0.17)
    star_p = run_drop(rec["E"], rec["nu"], rho, OUT / "star_pred.npz", shape="star", size=0.17)
    rmse_mm, mse_box, nf = _pos_err(star_t, star_p)
    print(f"\n[shape-gen] learn on RECTANGLE -> predict STAR (held-out geometry):")
    print(f"  recovered E={rec['E']:.3e} ({100*rec['E_err']:.1f}% err), "
          f"mu={rec['mu']:.3e} ({100*rec['mu_err']:.1f}% err)")
    print(f"  star: truth-law vs recovered-law re-sim = {rmse_mm:.2f} mm RMS, "
          f"box-norm MSE {mse_box:.2e} over {nf} frames")
    return rec, rmse_mm, mse_box


def report_errors():
    """Elastic reconstruction vs generalization position error (NCLaw metric family).
    Reconstruction = re-sim the recovered law on the training geometry (the rectangle);
    generalization = the held-out star. Mirrors the sand Table 5 rows."""
    OUT.mkdir(parents=True, exist_ok=True)
    E, nu, rho = TRUTH["E"], TRUTH["nu"], TRUTH["rho"]
    box = OUT / "box_truth.npz"
    if not box.exists():
        run_drop(E, nu, rho, box, shape="box", size=0.16, drop_gap=0.18)
    rec = recover(box)
    # reconstruction: re-sim the rectangle with the recovered moduli (same geometry, 1:1 particles)
    boxp = run_drop(rec["E"], rec["nu"], rho, OUT / "box_pred.npz", shape="box", size=0.16, drop_gap=0.18)
    rec_rmse, rec_mse, _ = _pos_err(box, boxp)
    # generalization: held-out star (reuse if present)
    st, sp = OUT / "star_truth.npz", OUT / "star_pred.npz"
    if not (st.exists() and sp.exists()):
        run_drop(E, nu, rho, st, shape="star", size=0.17)
        run_drop(rec["E"], rec["nu"], rho, sp, shape="star", size=0.17)
    gen_rmse, gen_mse, _ = _pos_err(st, sp)
    print("\n[elastic: reconstruction vs generalization] per-particle position error (NCLaw metric)")
    print(f"  recovered E={rec['E']:.3e} ({100*rec['E_err']:.1f}% err), mu err {100*rec['mu_err']:.1f}%")
    print(f"  RECONSTRUCTION (rectangle, training geom): {rec_rmse:.2f} mm RMS, box-norm MSE {rec_mse:.2e}")
    print(f"  GENERALIZATION (star, held-out geom):      {gen_rmse:.2f} mm RMS, box-norm MSE {gen_mse:.2e}")
    if (OUT / "truth.npz").exists() and (OUT / "recovered.npz").exists():
        s_rmse, s_mse, _ = _pos_err(OUT / "truth.npz", OUT / "recovered.npz")
        print(f"  (sphere reconstruction, for reference:     {s_rmse:.2f} mm RMS, box-norm MSE {s_mse:.2e})")
    return dict(recon_rmse=rec_rmse, recon_mse=rec_mse, gen_rmse=gen_rmse, gen_mse=gen_mse, rec=rec)


if __name__ == "__main__":
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "errors":
        report_errors()
    elif len(sys.argv) > 1 and sys.argv[1] == "shape":
        shape_generalization()
    else:
        tp = OUT / "truth.npz"
        if not tp.exists():
            run_drop(TRUTH["E"], TRUTH["nu"], TRUTH["rho"], tp, shape="sphere", size=0.14)
        rec = recover(tp)
        rp = run_drop(rec["E"], rec["nu"], TRUTH["rho"], OUT / "recovered.npz", shape="sphere", size=0.14)
        rmse_mm, mse_box, nf = _pos_err(tp, rp)
        print(f"[learned bounce] truth vs recovered-law re-sim: {rmse_mm:.2f} mm RMS, "
              f"box-norm MSE {mse_box:.2e}  over {nf} frames")
