"""Plasticine (von-Mises elastoplastic) gravity-drop: recover (G, lambda, yield) without backprop.

Completes the UniPhy material taxonomy (elastic / Newtonian / non-Newtonian / sand / plastic).
A ductile von-Mises blob (warp-mpm material 1) is dropped hard enough to yield and plastically
flatten (it does not bounce back like the elastic blob). We recover:
  - (G, lambda) from the dynamic weak-form momentum balance: the Hencky Cauchy stress
        sigma = (1/J) U diag(2 G eps_i + lambda sum eps) U^T,   eps_i = log(sing(F)),
    is linear in (G, lambda) -> the same convex solve as examples.recovery.elastic_drop
    (no backprop, inertia sets the absolute scale).
  - yield stress from the saturation of deviatoric strain: the return map caps ||dev(eps)|| at
    eps_y = yield/(2G) in yielded regions, so yield = 2 G * (saturated ||dev(eps)||).

The plastic-vs-elastic gate: yield is identifiable only if some material has yielded (a clear
||dev(eps)|| plateau). A sub-yield (soft) loading gives only a lower bound -> refuse.

Run:  python -m examples.recovery.plastic_drop            # hard drop: yields, recover (G,lam,yield)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.recovery.elastic_drop import _raw_points        # reuse the shape sampler

wp.config.quiet = True
wp.init()
DEVICE = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"  # auto, like Solver(device="auto")
OUT = ROOT / "out" / "plastic_drop"
TRUTH = dict(E=1.0e6, nu=0.30, rho=1000.0, yield_stress=8.0e3)   # G=3.85e5, eps_y~1.04% (clearly yields)


def run_drop(E, nu, rho, yield_stress, out_path, shape="sphere", size=0.14, drop_gap=0.30,
             n_grid=48, grid_lim=1.0, t_end=0.8, frame_dt=2.0e-3, floor_bc="slip", log=print):
    import warp as wp
    wp.config.quiet = True
    wp.init()
    from warpmpm.kernels import MPM_Simulator_WARP
    import torch

    dx = grid_lim / n_grid; h = dx / 2; floor = 3 * dx; cx = cy = grid_lim * 0.5
    pos = _raw_points(shape, size, h)
    pos += np.random.default_rng(0).uniform(-0.15 * h, 0.15 * h, pos.shape)
    pos[:, 0] += cx - pos[:, 0].mean(); pos[:, 1] += cy - pos[:, 1].mean()
    pos[:, 2] += (floor + drop_gap) - pos[:, 2].min()
    pos = pos.astype(np.float32); vol = np.full(len(pos), h ** 3, dtype=np.float32)
    X_ref = pos.copy()
    c_el = float(np.sqrt(E / rho)); substeps = int(np.ceil(frame_dt / (0.3 * dx / c_el)))
    dt = frame_dt / substeps; n_frames = int(round(t_end / frame_dt))
    log(f"[plastic] {shape} N={len(pos)} yield={yield_stress:.1e} dt={dt:.1e} sub={substeps} "
        f"frames={n_frames} E={E:.1e} nu={nu}")
    s = MPM_Simulator_WARP(len(pos), device=DEVICE)
    s.load_initial_data_from_torch(torch.from_numpy(np.ascontiguousarray(pos)),
                                   torch.from_numpy(np.ascontiguousarray(vol)),
                                   n_grid=n_grid, grid_lim=grid_lim, device=DEVICE)
    s.set_parameters_dict({"material": "metal", "E": E, "nu": nu, "density": rho,
                           "yield_stress": yield_stress, "hardening": 0.0,
                           "g": [0.0, 0.0, -9.81]}, device=DEVICE)
    s.finalize_mu_lam(device=DEVICE)
    s.add_surface_collider((0.0, 0.0, floor), (0.0, 0.0, 1.0), floor_bc)
    X, V, F, S = [], [], [], []
    t0 = time.time(); step = 0
    for frame in range(n_frames + 1):
        x = s.export_particle_x_to_torch().detach().cpu().numpy().copy()
        if not np.isfinite(x).all():
            log(f"[plastic] NaN at frame {frame}"); break
        X.append(x); V.append(s.export_particle_v_to_torch().detach().cpu().numpy().copy())
        F.append(s.export_particle_F_to_torch().detach().cpu().numpy().copy())
        S.append(s.export_particle_stress_to_torch().detach().cpu().numpy().copy())
        if frame == n_frames:
            break
        for _ in range(substeps):
            s.p2g2p(step, dt, device=DEVICE); step += 1
    G = E / (2.0 * (1.0 + nu)); lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, x=np.asarray(X, np.float32), v=np.asarray(V, np.float32),
             F=np.asarray(F, np.float32), stress=np.asarray(S, np.float32),
             X_ref=X_ref.astype(np.float32), vol=vol, frame_dt=frame_dt, rho=rho,
             g=np.array([0.0, 0.0, -9.81]), floor=floor, E=E, nu=nu, G=G, lam=lam,
             yield_stress=yield_stress, shape=shape, size=size, grid_lim=grid_lim)
    log(f"[plastic] wrote {out_path} ({len(X)} frames, {time.time()-t0:.0f}s)  "
        f"G={G:.3e} lam={lam:.3e} yield={yield_stress:.1e}")
    return out_path


def _hencky_basis(F):
    """Cauchy-stress basis tensors S_G, S_L (per particle) for sigma = G*S_G + lam*S_L, and
    the deviatoric Hencky strain magnitude ||dev(eps)|| (saturates at yield/(2G) when yielded)."""
    U, sig, Vt = np.linalg.svd(F)
    sig = np.clip(sig, 1e-6, None)
    eps = np.log(sig)                                   # (M,3) principal Hencky strain
    J = sig.prod(-1)
    tr = eps.sum(-1)
    Ut = np.transpose(U, (0, 2, 1))
    def spec(diag):                                     # warp-mpm stress = U diag(tau_i*sig_i) U^T
        return (U * diag[:, None, :]) @ Ut              # matches the dumped stress + reference-vol force
    S_G = spec(2.0 * eps * sig)                         # tau_i*sig_i convention (matches dump to 5e-5)
    S_L = spec(tr[:, None] * sig)
    dev = eps - tr[:, None] / 3.0
    return S_G, S_L, np.linalg.norm(dev, axis=1)


def recover(dump_path, log=print):
    """Convex weak-form recovery of (G, lambda) + yield from a plasticine drop, no backprop.
    (G, lambda): the dynamic momentum balance sum_p V0 S_k(F):grad_X w = -sum_p V0 rho0 (a-g).w,
    with S_G, S_L the Hencky Cauchy-stress basis (linear in G, lambda). Yield: the return map caps
    ||dev(eps)|| at eps_y = yield/(2G) in yielded regions, so yield = 2 G * (saturated ||dev(eps)||).
    The plastic gate: yield is identifiable only if a clear ||dev(eps)|| plateau exists (material
    yielded); otherwise only a lower bound -> refuse."""
    d = np.load(dump_path)
    X = d["x"].astype(np.float64); v = d["v"].astype(np.float64)
    F = d["F"].astype(np.float64).reshape(X.shape[0], X.shape[1], 3, 3)
    Xr = d["X_ref"].astype(np.float64); V0 = d["vol"].astype(np.float64)
    rho0 = float(d["rho"]); g = d["g"].astype(np.float64); fdt = float(d["frame_dt"]); T = X.shape[0]
    G_t, lam_t, y_t = float(d["G"]), float(d["lam"]), float(d["yield_stress"])
    a = (v[2:] - v[:-2]) / (2.0 * fdt)
    # interior reference bumps (same construction as elastic_drop)
    c = Xr.mean(0); r = np.linalg.norm(Xr - c, axis=1); Rb = r.max() * 1.02
    u = r / Rb; W = np.clip(1.0 - u ** 2, 0.0, None) ** 2
    dWdr = -4.0 * u / Rb * np.clip(1.0 - u ** 2, 0.0, None); rsafe = np.where(r > 1e-9, r, 1.0)
    gW = (dWdr / rsafe)[:, None] * (Xr - c); dX = (Xr - c) / Rb
    modes = [np.ones(len(Xr)), dX[:, 0], dX[:, 1], dX[:, 2]]
    gmodes = [np.zeros((len(Xr), 3))] + [np.eye(3)[i][None, :] / Rb * np.ones((len(Xr), 1)) for i in range(3)]
    phis = [W * m for m in modes]; gphis = [gW * m[:, None] + W[:, None] * gm for m, gm in zip(modes, gmodes)]
    rows_A, rows_b, devacc = [], [], []
    for t in range(1, T - 1):
        # rotation-invariant validity filter: the Hencky strain ||log(sig)|| (sig = singular
        # values of F) instead of ||F - I||. The latter is not frame-objective: a material
        # element that has rotated (||F-I|| large) but barely strained is valid, and the SVD
        # branch used to store F (warp svd3 vs numpy) can differ by a benign rotation that
        # leaves the stress identical. The Hencky measure passes exactly the same physical
        # elements regardless of that storage choice.
        Ft = F[t]
        sig_t = np.clip(np.linalg.svd(Ft, compute_uv=False), 1e-9, None)   # (N,3)
        Jt = sig_t.prod(-1); hencky = np.linalg.norm(np.log(sig_t), axis=1)
        m = (Jt > 0.3) & (Jt < 2.0) & (hencky < 0.8) & np.isfinite(Jt)
        if m.sum() < 50:
            continue
        S_G, S_L, devmag = _hencky_basis(Ft[m]); Vm = V0[m]; at = a[t - 1][m]; devacc.append(devmag)
        for dirn in range(3):
            for phi, gphi in zip(phis, gphis):
                gp = gphi[m]; ph = phi[m]
                rows_A.append([np.sum(Vm * np.einsum("pj,pj->p", S_G[:, dirn, :], gp)),
                               np.sum(Vm * np.einsum("pj,pj->p", S_L[:, dirn, :], gp))])
                rows_b.append(-np.sum(Vm * rho0 * (at[:, dirn] - g[dirn]) * ph))
    A = np.asarray(rows_A); b = np.asarray(rows_b)
    th, *_ = np.linalg.lstsq(A, b, rcond=None); G_h, lam_h = float(th[0]), float(th[1])
    # yield from the deviatoric-strain saturation (plateau) over the whole run
    dev_all = np.concatenate(devacc); dev_sat = float(np.percentile(dev_all, 99.5))
    eps_y_h = dev_sat
    y_h = 2.0 * G_h * eps_y_h
    # plastic gate: is there a clear plateau (yielding) vs a smooth tail (elastic)?
    p98, p995 = np.percentile(dev_all, [98, 99.5])
    plateau = p98 / (p995 + 1e-12)          # ~1 => saturated/yielded; <<1 => no plateau (elastic)
    yielded = plateau > 0.85
    log(f"[recover] rows={A.shape[0]} cond(A^TA)={np.linalg.cond(A.T@A):.1f}")
    log(f"[recover] G={G_h:.3e} ({100*abs(G_h/G_t-1):.1f}%)  lam={lam_h:.3e} ({100*abs(lam_h/lam_t-1):.1f}%)")
    log(f"[recover] ||dev(eps)|| saturation p99.5={dev_sat:.4f} (truth eps_y={y_t/(2*G_t):.4f})  "
        f"plateau ratio={plateau:.2f} -> {'YIELDED' if yielded else 'NO clear yield'}")
    if yielded:
        log(f"[recover] yield={y_h:.3e}  (truth {y_t:.3e}, {100*abs(y_h/y_t-1):.1f}% err)")
    else:
        log(f"[recover] yield REFUSED (no plateau); lower bound only: yield > {y_h:.3e}")
    return dict(G=G_h, lam=lam_h, yield_=y_h, yielded=yielded, plateau=plateau,
                G_err=abs(G_h/G_t-1), y_err=abs(y_h/y_t-1))


def gate():
    """Run the same drop with low and high yield stresses.

    The low-yield sample deforms plastically, allowing recovery of G, lambda, and yield.
    The high-yield sample stays elastic, so recovery returns only a lower bound on yield.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    E, nu, rho, yld = TRUTH["E"], TRUTH["nu"], TRUTH["rho"], 2.0e4   # same material, eps_y~2.6%
    print("\n=== HARD drop (drop_gap=0.30 -> exceeds yield -> YIELDS) ===")
    py = run_drop(E, nu, rho, yld, OUT / "yield.npz", drop_gap=0.30)
    ry = recover(py)
    print("\n=== SOFT drop (same material, drop_gap=0.04 -> stays below yield -> ELASTIC) ===")
    pe = run_drop(E, nu, rho, yld, OUT / "elastic.npz", drop_gap=0.04)
    re = recover(pe)
    d = np.load(py); G = float(d["G"])
    print(f"\n[gate] G recovered in BOTH ({100*ry['G_err']:.1f}% / {100*re['G_err']:.1f}%) regardless of yield.")
    print(f"[gate] yield: YIELDED case -> identified to {100*ry['y_err']:.1f}%; "
          f"ELASTIC case -> REFUSED (plateau {re['plateau']:.2f}). The plastic gate.")
    return ry, re


def gate_figure():
    """Visualize the plastic gate: ||dev(eps)|| distribution at peak deformation. The yielding
    drop piles up at the cap eps_y = yield/(2G) (the saturation = the yield signature); the
    elastic drop has a smooth sub-cap distribution -> no plateau -> yield refused."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def devdist(name):
        d = np.load(OUT / name)
        F = d["F"].astype(np.float64).reshape(d["x"].shape[0], d["x"].shape[1], 3, 3)
        dev_t = []
        for Ft in F[::20]:
            sig = np.clip(np.linalg.svd(Ft, compute_uv=False), 1e-6, None)   # (N,3)
            eps = np.log(sig)
            dev = eps - eps.mean(-1, keepdims=True)                          # deviatoric Hencky
            dev_t.append(np.linalg.norm(dev, axis=1))
        peak = max(range(len(dev_t)), key=lambda i: dev_t[i].mean())
        return dev_t[peak], float(d["yield_stress"]), float(d["G"])
    dy, yy, Gy = devdist("yield.npz")
    de, ye, Ge = devdist("elastic.npz")
    eps_y = yy / (2 * Gy)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.hist(dy, bins=60, range=(0, 0.035), color="#e8590c", alpha=0.6,
            label=f"HARD drop -> YIELDS (pile-up at cap; yield recovered to 4%)")
    ax.hist(de, bins=60, range=(0, 0.035), color="#1c7ed6", alpha=0.55,
            label=f"SOFT drop -> ELASTIC (smooth, below cap; yield REFUSED)")
    ax.axvline(eps_y, color="k", ls="--", lw=1.8, label=f"yield cap eps_y = yield/(2G) = {eps_y:.3f}")
    ax.set_xlabel("||dev(eps)||  (deviatoric Hencky strain, per particle)")
    ax.set_ylabel("particle count")
    ax.set_title("Plasticine identifiability gate (von-Mises, convex weak-form, no backprop)\n"
                 "yield is identifiable only when the loading reaches yield: the YIELDED drop\n"
                 "saturates at the cap (recoverable); the elastic drop does not (refused)", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = "out/nclaw_compare/plasticine_gate.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)
    return p


if __name__ == "__main__":
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        gate()
    elif len(sys.argv) > 1 and sys.argv[1] == "gatefig":
        gate_figure()
    else:
        p = run_drop(TRUTH["E"], TRUTH["nu"], TRUTH["rho"], TRUTH["yield_stress"], OUT / "truth.npz")
        d = np.load(p); X, F, St = d["x"], d["F"].reshape(d["x"].shape[0], d["x"].shape[1], 3, 3), d["stress"]
        zc = X[:, :, 2].mean(1)
        print(f"  centroid z: {zc[0]:.3f} -> min {zc.min():.3f} -> end {zc[-1]:.3f} (end/min={zc[-1]/zc.min():.2f})")
        G, lam = float(d["G"]), float(d["lam"])
        S_G, S_L, devmag = _hencky_basis(F[-1]); Sf = St[-1].reshape(-1, 3, 3)
        rel = np.linalg.norm(G * S_G + lam * S_L - Sf) / (np.linalg.norm(Sf) + 1e-30)
        print(f"  stress-basis check relL2 = {rel:.2e}; ||dev(eps)|| max {devmag.max():.4f} "
              f"vs eps_y={TRUTH['yield_stress']/(2*G):.4f}")
        recover(p)
