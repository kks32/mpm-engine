"""Control: is FE's win over the true-form Herschel-Bulkley fit an identifiability/
regularization effect, not a model-class effect?

The 3D shear cell gave FE 11% and Bingham 34% held-out force rollout. Fitting the TRUE
3-term form eta_app = eta + tau_y/g + pk*g^(pn-1) to the same data by plain least squares
gave a WORSE 30% rollout: the data under-determines the curve (the yield term tau_y/g and
the power-law term pk*g^(pn-1) are nearly collinear over a finite band), so the unregularized
fit collapses tau_y -> 0 and mispredicts. This adds a RIDGE-REGULARIZED HB fit toward a
generic (not truth) dough prior, the parametric analogue of FE's Gram smoothness + corpus
basis, re-simulates both HB variants on the held-out speed, and plots the eta_app curves and
the force-rollout traces for truth / FE / Bingham / HB-LSQ / HB-ridge.

Run:  PYTHONPATH=src ../.venv/bin/python examples/correct_model_check.py
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

import shear_cell_3d as S
from warpmpm import newtonian

OUT = S.OUT
V_HOLDOUT = 0.16
NF_H = 160
FDT = 2.0e-3
PN_GRID = np.linspace(0.2, 1.0, 33)


def _training_frames(speeds):
    frames = []
    for vp in speeds:
        nf = int(np.clip(round(0.4 * S.GEOM[0] / (vp * FDT)), 40, 160))
        seg = S.shear_segment(vp, S._truth_material(), n_frames=nf, record_power=True)
        rows = seg["diss_rows"]
        KE = np.array(seg["KE"])
        n = len(rows)
        for i, (_f, Pwall, Pg, gd, vol) in enumerate(rows):
            if not (0.2 * (n - 1) <= i <= 0.9 * (n - 1)):
                continue
            dKE = (KE[min(i + 1, n - 1)] - KE[max(i - 1, 0)]) / (2 * FDT)
            frames.append((gd, vol, Pwall + Pg - dKE))
        print(f"  swept v={vp:5.3f} ({nf} frames)")
    return frames


def _design(frames, pn):
    """Per-frame HB design rows: [INT gd^2 dV, INT gd^2/g dV, INT gd^2 g^(pn-1) dV]."""
    A = np.zeros((len(frames), 3))
    for j, (gd, vol, _) in enumerate(frames):
        g = np.sqrt(gd ** 2 + S.EPS ** 2)
        w = gd ** 2 * vol
        A[j] = [w.sum(), (w / g).sum(), (w * g ** (pn - 1.0)).sum()]
    return A


def fit_hb_lsq(frames):
    """Unregularized nonneg least squares over (eta, tau_y, pk), pn by 1-D search."""
    diss = np.array([d for _, _, d in frames])
    best = None
    for pn in PN_GRID:
        A = _design(frames, pn)
        sc = A.max(0) + 1e-30
        coef, _ = nnls(A / sc, diss)
        coef = coef / sc
        r = float(np.linalg.norm(A @ coef - diss) / (np.linalg.norm(diss) + 1e-30))
        if best is None or r < best[0]:
            best = (r, pn, coef)
    r, pn, c = best
    return dict(eta=float(c[0]), tau_y=float(c[1]), pk=float(c[2]), pn=float(pn), resid=r)


def fit_hb_ridge(frames, rho=1.0, pn_prior=None, pn_sig=0.25, w_pn=0.01):
    """Ridge toward a GENERIC dough prior (not the truth), in column-normalized coords so
    the prior strength is commensurate with the data. The prior pins the collinear
    (tau_y, pk) direction the data cannot resolve, exactly like FE's smoothness/corpus bias.
    With pn_prior set, the nonlinear exponent is also regularized (the residual is nearly
    flat in pn over the collinear band, so a mild prior moves it off the search boundary)."""
    diss = np.array([d for _, _, d in frames])
    m_phys = np.array([20.0, 80.0, 80.0])         # generic dough mean (truth is 10/40/60)
    best = None
    for pn in PN_GRID:
        A = _design(frames, pn)
        s = np.linalg.norm(A, axis=0) + 1e-30
        An = A / s                                 # unit-norm columns
        mn = s * m_phys                            # prior mean in normalized coords
        AtA = An.T @ An
        coef_n = np.linalg.solve(AtA + rho * np.eye(3), An.T @ diss + rho * mn)
        coef = np.clip(coef_n / s, 0.0, None)      # physical, nonneg
        r = float(np.linalg.norm(A @ coef - diss) / (np.linalg.norm(diss) + 1e-30))
        pen = 0.0 if pn_prior is None else w_pn * ((pn - pn_prior) / pn_sig) ** 2
        j = r * r + pen
        if best is None or j < best[0]:
            best = (j, pn, coef, r)
    _j, pn, c, r = best
    return dict(eta=float(c[0]), tau_y=float(c[1]), pk=float(c[2]), pn=float(pn), resid=r)


def _eta_curve(hb, gg):
    g = np.sqrt(gg ** 2 + S.EPS ** 2)
    return hb["eta"] + hb["tau_y"] / g + hb["pk"] * g ** (hb["pn"] - 1.0)


def _rollout(hb, tF, m0):
    mat = (newtonian(eta=hb["eta"], density=S.RHO, bulk_modulus=9.0e5)
           .with_yield(hb["tau_y"]).with_powerlaw(K=hb["pk"], n=hb["pn"]))
    Fx = S.shear_segment(V_HOLDOUT, mat, n_frames=NF_H)["Fx"]
    n = min(len(Fx), len(tF))
    err = float(np.linalg.norm(Fx[m0:n] - tF[m0:n]) / (np.linalg.norm(tF[m0:n]) + 1e-30))
    return Fx, err


def run(speeds=(0.006, 0.012, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)):
    print("re-running the 3D training sweep ...")
    frames = _training_frames(speeds)
    hb = {
        "HB lsq": fit_hb_lsq(frames),
        "HB ridge": fit_hb_ridge(frames, rho=1.0),
        "HB ridge+pn": fit_hb_ridge(frames, rho=1.0, pn_prior=0.5),
    }
    print("\n  recovered Herschel-Bulkley (truth eta=10, tau_y=40, pk=60, pn=0.4):")
    for name, h in hb.items():
        print(f"    {name:12s}: eta={h['eta']:5.1f} tau_y={h['tau_y']:5.1f} "
              f"pk={h['pk']:6.1f} pn={h['pn']:.2f}  (train resid {h['resid']*100:.1f}%)")

    c = np.load(OUT / "curve_3d.npz")
    d = np.load(OUT / "rollout_3d.npz")
    gg, eta_tr, eta_bg, eta_fe = c["gg"], c["eta_tr"], c["eta_bg"], c["eta_fe0"]
    tF, m0 = d["Fx_truth"], int(d["m0"])
    fr = {"FE": float(d["fr_FE"]), "Bingham": float(d["fr_Bingham"])}
    for h in hb.values():
        h["Fx"], h["err"] = _rollout(h, tF, m0)

    print(f"\n  held-out force rollout at v={V_HOLDOUT} m/s (relL2 vs truth):")
    print(f"    FE (flexible basis + reg)        = {fr['FE']*100:5.1f}%")
    print(f"    HB ridge+pn (true form, full reg)= {hb['HB ridge+pn']['err']*100:5.1f}%")
    print(f"    HB ridge (true form, lin reg)    = {hb['HB ridge']['err']*100:5.1f}%")
    print(f"    HB lsq (true form, no reg)       = {hb['HB lsq']['err']*100:5.1f}%")
    print(f"    Bingham (wrong class)            = {fr['Bingham']*100:5.1f}%")

    _figure(gg, eta_tr, eta_bg, eta_fe, hb, d, tF, m0, fr)
    return {"hb": hb, "fr": fr}


def _figure(gg, eta_tr, eta_bg, eta_fe, hb, d, tF, m0, fr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.0))
    rp, lsq = hb["HB ridge+pn"], hb["HB lsq"]

    ax[0].loglog(gg, eta_tr, "k-", lw=3.0, label="truth (HB)")
    ax[0].loglog(gg, eta_fe, color="#1864ab", lw=2.2, label="FE (flexible + reg)")
    ax[0].loglog(gg, _eta_curve(rp, gg), color="#2f9e44", lw=2.4,
                 label="HB ridge+pn (true form, full reg)")
    ax[0].loglog(gg, _eta_curve(lsq, gg), color="#f08c00", lw=1.8, ls=":",
                 label="HB lsq (true form, no reg)")
    ax[0].loglog(gg, eta_bg, color="#e8590c", lw=1.8, ls="--", label="Bingham (wrong class)")
    ax[0].set_xlabel("shear rate  gamma_dot  (1/s)")
    ax[0].set_ylabel("apparent viscosity  eta_app  (Pa.s)")
    ax[0].set_title("(a) recovered eta_app(gd)\nregularizing pn too recovers the true shape")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    th = d["t"]
    ax[1].plot(th, -tF, "k-", lw=3.0, label="truth")
    ax[1].plot(th, -d["Fx_FE"], color="#1864ab", lw=2.0, label=f"FE ({fr['FE']*100:.0f}%)")
    fr_rp, fr_l = rp["Fx"], lsq["Fx"]
    ax[1].plot(th[:len(fr_rp)], -fr_rp, color="#2f9e44", lw=2.2,
               label=f"HB ridge+pn ({rp['err']*100:.0f}%)")
    ax[1].plot(th[:len(fr_l)], -fr_l, color="#f08c00", lw=1.8, ls=":",
               label=f"HB lsq ({lsq['err']*100:.0f}%)")
    ax[1].plot(th, -d["Fx_Bingham"], color="#e8590c", lw=1.8, ls="--",
               label=f"Bingham ({fr['Bingham']*100:.0f}%)")
    ax[1].axvspan(0, th[m0], color="#f1f3f5", alpha=0.8)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("wall shear force  -F_x  (N)")
    ax[1].set_title(f"(b) held-out force rollout at v={V_HOLDOUT} m/s\n(relL2 vs truth)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "correct_model_compare.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    run()
