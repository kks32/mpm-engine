"""Strong-form (pointwise constitutive) vs weak-form (wall-force power balance) recovery.

Both recover eta_app(gd) = sum_k theta_k g_k(gd), but from different data:
  weak   : the wall-force power balance INT eta_app gd^2 dV = v_wall F_x + Pg - dKE. Uses only
           the measurable boundary force (what a real sensor gives); carries the discrete
           closure factor (wall power ~1.4x the particle-gd dissipation) and is identifiability-
           limited.
  strong : the local constitutive relation at each particle, sigma_dev:D_dev = eta_app(gd)
           (gd^2 - eps^2), regressed pointwise from the dumped Cauchy stress. An oracle (needs
           the full stress field, unavailable in a real experiment) but exact for the right
           basis, with no closure factor.

The comparison separates measurement/method limits (weak, force-based) from model-class limits:
the flexible FE basis is near-exact in the strong form and rides ~1.4x high in the weak form
(the closure factor), while the misspecified Bingham fit fails in both (it cannot represent the
shear-thinning shape regardless of the data form). Run:
  python experiments/strong_vs_weak.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import shear_cell_3d as M3
import shear_cell_fe as M2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parents[1] / "out" / "strong_vs_weak"
SPEEDS = (0.006, 0.012, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
FDT = 2.0e-3


def _weak_rows(seg, fe, M):
    a, b2, bb, gd, w = M._power_rows(seg, fe)
    return a, b2, bb, gd, w


def _strong_pool(segs, gd_lo, gd_hi):
    """Pool per-particle (gd, vol, sigma_dev:D_dev) over the sweep, in the excited band."""
    gd_all, vol_all, dd_all = [], [], []
    for seg in segs:
        rows = seg["strong_rows"]
        n = len(rows)
        for i, (gd, vol, dd) in enumerate(rows):
            if not (0.2 * (n - 1) <= i <= 0.9 * (n - 1)):
                continue
            m = (gd >= gd_lo) & (gd <= gd_hi) & np.isfinite(dd)
            gd_all.append(gd[m]); vol_all.append(vol[m]); dd_all.append(dd[m])
    return (np.concatenate(gd_all), np.concatenate(vol_all), np.concatenate(dd_all))


def run(dim="3d"):
    from ident.features.function_encoder import FunctionEncoderDict
    from ident.solve.qp import constrained_solve
    OUT.mkdir(parents=True, exist_ok=True)
    M = M3 if dim == "3d" else M2
    eps = M.EPS
    d = np.load(M.FE_TABLE)
    fe = FunctionEncoderDict(d["s_grid"], d["table"])
    Lx = M.GEOM[0] if dim == "3d" else M.COL_W

    # one sweep, recording both the wall-force power balance and the per-particle stress
    segs, A_fe, A_b, bvec, p50s = [], [], [], [], []
    t0 = time.time()
    for vp in SPEEDS:
        nf = int(np.clip(round(0.4 * Lx / (vp * FDT)), 40, 160))
        seg = M.shear_segment(vp, M._truth_material(), n_frames=nf,
                              record_power=True, record_stress=True)
        segs.append(seg)
        a, b2, bb, gd, w = _weak_rows(seg, fe, M)
        A_fe.append(a); A_b.append(b2); bvec.append(bb)
        order = np.argsort(gd)
        cw = np.cumsum(w[order]) / w.sum()
        p50s.append(float(gd[order][np.searchsorted(cw, 0.5)]))
    print(f"  [{dim}] swept {len(SPEEDS)} speeds in {time.time()-t0:.0f}s")
    gd_lo, gd_hi = max(min(p50s), 0.1), max(p50s)
    gg = np.logspace(np.log10(gd_lo), np.log10(gd_hi), 80)
    eta_tr = M.eta_app_true(gg)
    G = fe.gram((10.0 ** np.linspace(-1, 2, 257), np.ones(257)))
    Icon = np.logspace(np.log10(gd_lo), np.log10(gd_hi), 40)

    # ---- weak: wall-force power balance (no prior, data only) -----------------------
    A_fe = np.vstack(A_fe); A_b = np.vstack(A_b); bvec = np.concatenate(bvec)
    sc = float(np.sqrt(np.mean(bvec ** 2))) + 1e-30
    qp = constrained_solve(A_fe / sc, bvec / sc, fe, lam=1e-3, G=G, mu_min=0.5,
                           I_constraint_grid=Icon, nonnegativity=False, monotonic=False)
    eta_fe_weak = fe.phi(gg) @ qp.theta
    thb_w, *_ = np.linalg.lstsq(A_b, bvec, rcond=None)
    eta_bg_weak = thb_w[1] + thb_w[0] / np.sqrt(gg ** 2 + eps ** 2)

    # ---- strong: local apparent viscosity eta_est = sigma_dev:D_dev/(gd^2 - eps^2),
    # regressed pointwise against gd (the constitutive curve directly; volume-weighted so it
    # is not dissipation-biased like the weak/force balance). Restrict to gd where gd^2-eps^2
    # is well above the regularization floor so eta_est is not noise-amplified. ------------
    gd_p, vol_p, dd_p = _strong_pool(segs, gd_lo, gd_hi)
    g2 = gd_p ** 2 - eps ** 2                                    # = 2 dev(D):dev(D)
    keep = g2 > (2.0 * eps) ** 2
    gd_p, vol_p, eta_est = gd_p[keep], vol_p[keep], dd_p[keep] / g2[keep]
    wts = np.sqrt(np.maximum(vol_p, 0.0))
    A_s = fe.phi(gd_p) * wts[:, None]
    b_s = eta_est * wts
    qps = constrained_solve(A_s, b_s, fe, lam=1e-3, G=G, mu_min=0.5,
                            I_constraint_grid=Icon, nonnegativity=False, monotonic=False)
    eta_fe_strong = fe.phi(gg) @ qps.theta
    g_eps = np.sqrt(gd_p ** 2 + eps ** 2)
    A_bs = np.stack([1.0 / g_eps, np.ones_like(g_eps)], 1) * wts[:, None]    # -> [tau_y, eta]
    thb_s, *_ = np.linalg.lstsq(A_bs, b_s, rcond=None)
    eta_bg_strong = thb_s[1] + thb_s[0] / np.sqrt(gg ** 2 + eps ** 2)

    def rel(e):
        return float(np.sqrt(np.mean((e - eta_tr) ** 2)) / np.sqrt(np.mean(eta_tr ** 2)))
    out = dict(gg=gg, eta_tr=eta_tr, eta_fe_weak=eta_fe_weak, eta_fe_strong=eta_fe_strong,
               eta_bg_weak=eta_bg_weak, eta_bg_strong=eta_bg_strong, band=(gd_lo, gd_hi),
               theta_fe_strong=qps.theta, n_strong=len(gd_p))
    print(f"  [{dim}] eta_app(gd) curve relL2 over gd=[{gd_lo:.2f},{gd_hi:.2f}] ({len(gd_p)} "
          f"strong-form particle rows):")
    print(f"    FE      weak {rel(eta_fe_weak)*100:5.1f}%   strong {rel(eta_fe_strong)*100:5.1f}%")
    print(f"    Bingham weak {rel(eta_bg_weak)*100:5.1f}%   strong {rel(eta_bg_strong)*100:5.1f}%")
    print(f"    closure factor (weak FE / truth, median) ~ {np.median(eta_fe_weak/eta_tr):.2f}x")
    np.savez(OUT / f"curves_{dim}.npz", **{k: v for k, v in out.items()
                                           if k not in ("band",)}, band=np.array(out["band"]))
    return out


def _figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(res), figsize=(6.6 * len(res), 4.8), squeeze=False)
    for j, (dim, r) in enumerate(res.items()):
        a = ax[0][j]
        gg = r["gg"]
        a.loglog(gg, r["eta_tr"], "k-", lw=2.8, label="truth (HB)")
        a.loglog(gg, r["eta_fe_strong"], color="#2f9e44", lw=2.4, label="FE strong (local sigma)")
        a.loglog(gg, r["eta_fe_weak"], color="#1864ab", lw=2.0, label="FE weak (wall force)")
        a.loglog(gg, r["eta_bg_strong"], color="#f08c00", lw=1.7, ls=":", label="Bingham strong")
        a.loglog(gg, r["eta_bg_weak"], color="#e8590c", lw=1.7, ls="--", label="Bingham weak")
        a.set_xlabel("shear rate  gamma_dot  (1/s)")
        a.set_ylabel("apparent viscosity  eta_app  (Pa.s)")
        a.set_title(f"({'ab'[j]}) {dim.upper()} shear cell: strong vs weak eta_app(gd)\n"
                    "strong (local stress) ~ truth; weak (wall force) under-determined")
        a.legend(fontsize=8)
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / "strong_vs_weak.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


def replot():
    """Redraw the figure from cached curves_{dim}.npz (no re-sweep)."""
    res = {}
    for dim in ("3d", "2d"):
        p = OUT / f"curves_{dim}.npz"
        if p.exists():
            d = np.load(p)
            res[dim] = {k: d[k] for k in d.files}
    _figure(res)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which == "replot":
        replot()
    else:
        res = {dim: run(dim) for dim in (["3d", "2d"] if which == "both" else [which])}
        if len(res) > 1:
            _figure(res)
