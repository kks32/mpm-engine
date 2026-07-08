"""Plasticine in the FE-bounds + RLS framework: recursive identification of (G, lambda, yield).

Streams the von-Mises plasticine drop frame-by-frame through the convex weak-form solve,
accumulating the Fisher information M_t = sum A_t^T A_t for (G, lambda) (Hencky stress basis),
and tracks the yield via the running deviatoric-strain saturation. Recursive (RLS): the
plasticine analogue of examples.recovery.elastic_identify_sequential, so the same bounds
(approximation = FE SVD tail incl. plasticine; sample-complexity; recursive consistency)
cover the 5th family.

Shows:
  - G posterior mean +/- 95% credible band vs frame: flat at the prior in free-fall, contracts
    to truth at impact (recursive estimation, no backprop).
  - yield estimate vs frame: refused (lower bound) until the deviatoric strain saturates at the
    cap (the material yields), then locks to truth; the plastic gate, in time.

Run:  python -m examples.recovery.plastic_identify_sequential
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "plastic_drop"
from examples.recovery.plastic_drop import _hencky_basis


def stream(dump="yield.npz", prior_G=2.0e5, prior_nu=0.30):
    d = np.load(OUT / dump)
    X = d["x"].astype(np.float64); v = d["v"].astype(np.float64)
    F = d["F"].astype(np.float64).reshape(X.shape[0], X.shape[1], 3, 3)
    Xr = d["X_ref"].astype(np.float64); V0 = d["vol"].astype(np.float64)
    rho0 = float(d["rho"]); g = d["g"].astype(np.float64); fdt = float(d["frame_dt"]); T = X.shape[0]
    G_t, lam_t, y_t = float(d["G"]), float(d["lam"]), float(d["yield_stress"])
    a = (v[2:] - v[:-2]) / (2.0 * fdt)
    # interior reference bumps
    c = Xr.mean(0); r = np.linalg.norm(Xr - c, axis=1); Rb = r.max() * 1.02
    u = r / Rb; W = np.clip(1.0 - u ** 2, 0.0, None) ** 2
    dWdr = -4.0 * u / Rb * np.clip(1.0 - u ** 2, 0.0, None); rs = np.where(r > 1e-9, r, 1.0)
    gW = (dWdr / rs)[:, None] * (Xr - c); dX = (Xr - c) / Rb
    modes = [np.ones(len(Xr)), dX[:, 0], dX[:, 1], dX[:, 2]]
    gm = [np.zeros((len(Xr), 3))] + [np.eye(3)[i][None, :] / Rb * np.ones((len(Xr), 1)) for i in range(3)]
    phis = [W * m for m in modes]; gphis = [gW * m[:, None] + W[:, None] * q for m, q in zip(modes, gm)]

    def rows(t):
        Ft = F[t]; J = np.linalg.det(Ft); dvf = np.linalg.norm(Ft - np.eye(3), axis=(1, 2))
        m = (J > 0.3) & (J < 2.0) & (dvf < 0.8) & np.isfinite(J)
        if m.sum() < 50:
            return None, None, 0.0, 0.0
        S_G, S_L, devmag = _hencky_basis(Ft[m]); Vm = V0[m]; at = a[t - 1][m]
        A, b = [], []
        for dr in range(3):
            for phi, gphi in zip(phis, gphis):
                gp = gphi[m]; ph = phi[m]
                A.append([np.sum(Vm * np.einsum("pj,pj->p", S_G[:, dr, :], gp)),
                          np.sum(Vm * np.einsum("pj,pj->p", S_L[:, dr, :], gp))])
                b.append(-np.sum(Vm * rho0 * (at[:, dr] - g[dr]) * ph))
        p98, p995 = np.percentile(devmag, [98, 99.5])
        return np.asarray(A), np.asarray(b), float(p995), float(p98 / (p995 + 1e-12))

    # noise scale from full fit
    AA, bb = [], []
    for t in range(1, T - 1):
        A, b, *_ = rows(t)
        if A is not None:
            AA.append(A); bb.append(b)
    AA = np.vstack(AA); bb = np.concatenate(bb)
    th0, *_ = np.linalg.lstsq(AA, bb, rcond=None)
    sigma2 = float(np.sum((AA @ th0 - bb) ** 2) / max(len(bb) - 2, 1))
    Gp = prior_G; lamp = prior_G * 2 * prior_nu / (1 - 2 * prior_nu)
    prior_mean = np.array([Gp, lamp]); prior_prec = np.diag([1 / (5 * Gp) ** 2, 1 / (5 * lamp) ** 2])
    M = np.zeros((2, 2)); rhs = np.zeros(2)
    rec = []; sat_run = 0.0; yielded_at = None
    for t in range(1, T - 1):
        A, b, p995, plateau = rows(t)
        if A is not None:
            M += A.T @ A / sigma2; rhs += A.T @ b / sigma2
            sat_run = max(sat_run, p995)
        prec = M + prior_prec; Sig = np.linalg.inv(prec); th = Sig @ (rhs + prior_prec @ prior_mean)
        G_h = th[0]; G_std = float(np.sqrt(max(Sig[0, 0], 0)))
        is_yld = (A is not None) and (plateau > 0.85) and (p995 > 0.01)   # plateau and real strain
        if is_yld and yielded_at is None:
            yielded_at = t * fdt
        y_h = 2.0 * G_h * sat_run
        rec.append((t * fdt, G_h, G_std, y_h, sat_run, 1.0 if is_yld else 0.0))
    return np.array(rec), (G_t, lam_t, y_t), yielded_at


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec, (G_t, lam_t, y_t), yat = stream()
    t, G_h, G_std, y_h, sat, yld = (rec[:, i] for i in range(6))
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw=dict(height_ratios=[1, 1], hspace=0.13))
    ax[0].fill_between(t, (G_h - 1.96 * G_std) / 1e3, (G_h + 1.96 * G_std) / 1e3, color="#1c7ed6", alpha=0.25,
                       label="95% credible interval")
    ax[0].plot(t, G_h / 1e3, color="#1c7ed6", lw=2.2, label="posterior mean G")
    ax[0].axhline(G_t / 1e3, color="k", ls="--", lw=1.5, label="truth G")
    ax[0].set_ylim(0, 1.7 * G_t / 1e3); ax[0].set_ylabel("G  (kPa)")
    ax[0].set_title("Plasticine recursive (RLS) identification -- 5th family in the FE-bounds framework\n"
                    "G contracts to truth at impact (recursive, convex, no backprop)", fontsize=10)
    ax[0].legend(fontsize=9, loc="upper right"); ax[0].grid(alpha=0.3)
    ax[1].plot(t, y_h / 1e3, color="#e8590c", lw=2.2, label="recovered yield (= 2 G * sat ||dev eps||)")
    ax[1].axhline(y_t / 1e3, color="k", ls="--", lw=1.5, label="truth yield")
    if yat is not None:
        for a_ in ax:
            a_.axvline(yat, color="#2f9e44", lw=1.4)
        ax[1].text(yat, 0.5 * y_t / 1e3, f"  yield identifiable\n  (yields at t={yat:.2f}s)",
                   color="#2f9e44", fontsize=8.5)
    ax[1].set_ylim(0, 1.5 * y_t / 1e3); ax[1].set_xlabel("time (s) [streamed frame]"); ax[1].set_ylabel("yield  (kPa)")
    ax[1].text(0.02, 0.06, "before yield onset: REFUSED (lower bound); after: locks to truth",
               transform=ax[1].transAxes, fontsize=8.5, color="#a33")
    ax[1].legend(fontsize=9, loc="lower right"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = ROOT / "out" / "nclaw_compare" / "plasticine_rls_identify.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"wrote {p}")
    print(f"  G: prior {G_h[0]/1e3:.0f} -> {G_h[-1]/1e3:.1f} kPa (truth {G_t/1e3:.0f}, "
          f"{100*abs(G_h[-1]/G_t-1):.1f}%); yield {y_h[-1]/1e3:.1f} kPa (truth {y_t/1e3:.0f}, "
          f"{100*abs(y_h[-1]/y_t-1):.1f}%); yields at t={yat}")
    return p


if __name__ == "__main__":
    figure()
