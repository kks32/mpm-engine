"""Function-encoder recovery of a 3D dough's apparent viscosity eta_app(gamma_dot).

Tests whether the learned VISCOUS basis beats a fixed 2-parameter Bingham fit when the dough
is genuinely shear-thinning (Herschel-Bulkley, n<1) -- a curve a Bingham model structurally
cannot represent. We press a 3D dough blob at THREE plate speeds (to excite a range of shear
rate), read the Newton-exact grid-impulse FORCE each frame, and use the mechanical power
balance

    INT eta_app(gd) gd^2 dV = P_plate + P_grav - dKE ,   P_plate = v_plate * F (measured),

which is LINEAR in the basis coefficients for eta_app(gd) = sum_k theta_k g_k(gd). We pool the
three speeds, solve with the FE viscous basis (nonneg + Gram-smooth) and, separately, with the
Bingham basis {1/gd, 1}, and compare both recovered eta_app(gd) curves to truth over the
realized band. Force data is essential: it supplies P_plate. Run:
  ../.venv/bin/python examples/dough_fe_viscous.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, block, newtonian
from warpmpm.coupling.backend import WarpMPMBackend

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parents[1] / "out" / "fe_viscous"
G_MAG = 9.81
EPS = 0.05                       # the newtonian kernel's shear-rate regularization
# shear-thinning Herschel-Bulkley dough: eta_app(gd) = eta + tau_y/gd + pk*gd^(pn-1)
TRUTH = dict(eta=8.0, tau_y=100.0, pk=50.0, pn=0.4)


def eta_app_true(gd):
    g = np.sqrt(gd ** 2 + EPS ** 2)
    return TRUTH["eta"] + TRUTH["tau_y"] / g + TRUTH["pk"] * g ** (TRUTH["pn"] - 1.0)


def _gd(L):
    # match the kernel exactly: |gd|_eps = sqrt(2 dev(D):dev(D) + eps^2), eps=0.05
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    tr = (D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2]) / 3.0
    Dd = D - tr[..., None, None] * np.eye(3)
    return np.sqrt(2.0 * np.einsum("...ij,...ij->...", Dd, Dd) + EPS ** 2)


def squeeze(v_plate, fe, n_grid=48, geom=(0.12, 0.12, 0.06), press_strain=0.5,
            dt=1.0e-4, substeps=20, frame_stride=3, device="auto"):
    """One 3D squeeze; per frame return (diss, X_fe[K], X_bing[2], gd-percentiles)."""
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    cw, cd, ch = geom
    pos, vol0, floor = block(grid, size=geom, ppc=2)
    s = Solver(grid=grid, device=device).load_particles(pos, vol0)
    s.set_material(newtonian(eta=TRUTH["eta"], density=1000.0, bulk_modulus=9.0e5)
                   .with_yield(TRUTH["tau_y"]).with_powerlaw(K=TRUTH["pk"], n=TRUTH["pn"]))
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    cx = cy = grid.grid_lim * 0.5
    bh = (0.5 * cw + 0.012, 0.5 * cd + 0.012, 0.6 * grid.dx)
    be = WarpMPMBackend(solver=s); z = floor + ch + bh[2]; tool = be.attach_tool((cx, cy, z), bh)
    fdt = dt * substeps; nf = round(press_strain * ch / v_plate / fdt)
    prev = z; rows = []; KE = []; gd_all = []
    for f in range(nf + 1):
        zn = z - v_plate * fdt if f > 0 else z; vz = (zn - prev) / fdt
        if f > 0:
            be.set_tool_kinematics(tool, center=(cx, cy, prev), velocity=(0, 0, vz))
            be.reset_tool_force(tool); be.step(dt, substeps)
        z = zn; prev = zn
        v = s.v(); L = s.L(); vol = s.vol()
        F = abs(float(be.get_tool_reaction(tool, fdt)[2])) if f > 0 else 0.0
        gd = _gd(L)
        ge = fe.phi(gd)                                   # (N, K) basis at each particle gd
        w = gd ** 2 * vol                                 # eta_app * gd^2 dV weighting
        X_fe = (ge * w[:, None]).sum(0)                   # (K,)
        X_b = np.array([(1.0 / np.sqrt(gd ** 2 + EPS ** 2) * w).sum(), w.sum()])  # tau_y, eta
        Pg = float(np.sum(1000.0 * (-G_MAG) * v[:, 2] * vol))
        KE.append(float(0.5 * 1000.0 * np.sum(vol * np.sum(v ** 2, axis=1))))
        rows.append([f * fdt, v_plate * F, Pg, X_fe, X_b])
        gd_all.append(np.percentile(gd, [50, 95]))
    return rows, np.array(KE), np.array(gd_all)


def viscous_prior(fe, s_grid, n=800, seed=0):
    """Scale-aware family-coefficient prior: encode physically-scaled dough eta_app(gd)
    curves (the corpus is unit-normalized, so we rebuild at real magnitudes) into the FE
    basis -> Gaussian prior (theta_bar, Sigma) over plausible dough laws."""
    rng = np.random.default_rng(seed)
    gd = 10.0 ** s_grid
    g = np.sqrt(gd ** 2 + EPS ** 2)
    E = []
    for _ in range(n):
        kind = rng.choice(["newtonian", "powerlaw", "carreau", "bingham", "herschel"])
        if kind == "newtonian":
            e = np.full_like(gd, rng.uniform(1, 50))
        elif kind == "powerlaw":
            e = rng.uniform(10, 200) * g ** (rng.uniform(0.2, 1.0) - 1.0)
        elif kind == "carreau":
            e0, ei = rng.uniform(50, 400), rng.uniform(1, 20)
            lam, nn = rng.uniform(0.1, 5), rng.uniform(0.2, 0.9)
            e = ei + (e0 - ei) * (1 + (lam * gd) ** 2) ** ((nn - 1) / 2)
        elif kind == "bingham":
            e = rng.uniform(1, 30) + rng.uniform(20, 400) / g
        else:
            e = (rng.uniform(1, 30) + rng.uniform(20, 400) / g
                 + rng.uniform(10, 150) * g ** (rng.uniform(0.2, 1.0) - 1.0))
        E.append(e)
    E = np.array(E)
    Phi = fe.phi(gd)                                      # (256, K)
    W = np.gradient(s_grid)                               # uniform-in-s weight
    PtW = Phi.T * W; Gm = PtW @ Phi; K = Phi.shape[1]
    Gm = Gm + 1e-6 * (np.trace(Gm) / K) * np.eye(K)
    Th = np.linalg.solve(Gm, PtW @ E.T).T                 # (n, K) physical-scale coefficients
    tbar = Th.mean(0); cov = np.cov(Th, rowvar=False)
    cov = cov + 1e-3 * (np.trace(cov) / K) * np.eye(K)
    return tbar, cov


def run(device="auto"):
    from ident.features.function_encoder import FunctionEncoderDict
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(REPO / "mpm_engine/fe-weights/viscous.npz")
    fe = FunctionEncoderDict(d["s_grid"], d["table"])     # eta_app basis on s=log10 gd
    speeds = (0.04, 0.08, 0.16)
    A_fe, A_b, bvec, gdp = [], [], [], []
    for vp in speeds:
        rows, KE, gd_all = squeeze(vp, fe, device=device)
        n = len(rows)
        for f in range(n):
            _t, Pplate, Pg, X_fe, X_b = rows[f]
            if not (0.15 * (n - 1) <= f <= 0.92 * (n - 1)):
                continue
            dKE = (KE[min(f + 1, n - 1)] - KE[max(f - 1, 0)]) / (2 * (rows[1][0] - rows[0][0]))
            diss = Pplate + Pg - dKE
            A_fe.append(X_fe); A_b.append(X_b); bvec.append(diss); gdp.append(gd_all[f])
        print(f"v_plate={vp}: {n} frames, gd_med~{np.median([g[0] for g in gd_all]):.2f}/s")
    A_fe = np.array(A_fe); A_b = np.array(A_b); bvec = np.array(bvec); gdp = np.array(gdp)
    gd_lo = max(np.percentile(gdp[:, 0], 5), 0.2); gd_hi = np.percentile(gdp[:, 1], 95)
    gg = np.logspace(np.log10(gd_lo), np.log10(gd_hi), 80)

    from ident.solve.qp import constrained_solve
    G = fe.gram((10.0 ** np.linspace(-1, 2, 257), np.ones(257)))
    Icon = np.logspace(np.log10(gd_lo), np.log10(gd_hi), 40)

    def fe_solve(rho):
        A, b = A_fe, bvec
        if rho > 1e-9:
            M = rho * np.linalg.inv(cov); Lc = np.linalg.cholesky(M)
            A = np.vstack([A_fe, Lc]); b = np.concatenate([bvec, Lc @ tbar])
        qp = constrained_solve(A, b, fe, lam=1e-3, G=G, mu_min=1.0,
                               I_constraint_grid=Icon, nonnegativity=False, monotonic=False)
        return fe.phi(gg) @ qp.theta

    tbar, cov = viscous_prior(fe, d["s_grid"])
    eta_fe0 = fe_solve(0.0)                                # FE, no prior (overfits)
    eta_fe = fe_solve(1.0)                                 # FE + family prior
    th_b, *_ = np.linalg.lstsq(A_b, bvec, rcond=None)      # Bingham [tau_y, eta]
    eta_b = th_b[1] + th_b[0] / np.sqrt(gg ** 2 + EPS ** 2)
    eta_t = eta_app_true(gg)

    def rel(e):
        return float(np.sqrt(np.mean((e - eta_t) ** 2)) / np.sqrt(np.mean(eta_t ** 2)))
    print(f"\ndough eta_app(gd) recovery over gd=[{gd_lo:.2f},{gd_hi:.2f}] /s  (truth HB {TRUTH})")
    print(f"  Bingham fit (2-param)    relL2 = {rel(eta_b)*100:5.1f}%  (misspecified)")
    print(f"  FE, no prior (K=8)       relL2 = {rel(eta_fe0)*100:5.1f}%  (overfits narrow band)")
    print(f"  FE + family prior (K=8)  relL2 = {rel(eta_fe)*100:5.1f}%")
    _figure(gg, eta_t, eta_b, eta_fe0, eta_fe, rel(eta_b), rel(eta_fe0), rel(eta_fe))
    return {"bingham_relL2": rel(eta_b), "fe_noprior_relL2": rel(eta_fe0), "fe_relL2": rel(eta_fe)}


def _figure(gg, eta_t, eta_b, eta_fe0, eta_fe, rb, rf0, rf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.loglog(gg, eta_t, "k-", lw=2.6, label="truth (Herschel-Bulkley)")
    ax.loglog(gg, eta_b, color="#adb5bd", lw=1.8, ls="--",
              label=f"Bingham fit (relL2 {rb*100:.0f}%)")
    ax.loglog(gg, eta_fe0, color="#e8590c", lw=1.6, ls=":",
              label=f"FE, no prior (relL2 {rf0*100:.0f}%)")
    ax.loglog(gg, eta_fe, color="#1c7ed6", lw=2.4,
              label=f"FE + family prior (relL2 {rf*100:.0f}%)")
    ax.set_xlabel("shear rate  gamma_dot  (1/s)")
    ax.set_ylabel("apparent viscosity  eta_app  (Pa.s)")
    ax.set_title("3D dough eta_app(gd): FE basis vs Bingham\n"
                 "(grid-impulse force, 3 pooled speeds)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); p = OUT / "dough_fe_viscous.png"; fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="Warp MPM device, e.g. cuda:0 or cuda:1")
    args = parser.parse_args()
    run(device=args.device)
