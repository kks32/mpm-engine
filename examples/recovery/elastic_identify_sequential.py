"""Sequential elastic identifiability: confidence vs measurements, gated by the deformation seen.

Streams the elastic gravity-drop through the convex weak-form solve frame-by-frame, accumulating
the Fisher information  M_t = sum_{tau<=t} A_tau^T A_tau  (the same A as examples.recovery.elastic_drop.recover,
columns = the moduli mu, lambda). At each frame it forms the posterior covariance
Sigma = (M_t/sigma^2 + prior_prec)^{-1} and propagates it to the quantities of interest E and nu.

The figure answers the questions directly:
  - posterior std of E and nu vs frame: flat during free-fall (no strain -> A ~ 0 -> M does not
    grow), then drops sharply at impact -> "how many measurements before confident".
  - E (shear-dominated) crosses the confidence threshold; nu (needs volumetric strain) stays
    starved because a bounce barely compresses -> "enough deformation to extract this property?"
    is per-property: shear yes, bulk no.

Run:  python -m examples.recovery.elastic_identify_sequential [sphere|box]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "elastic_drop"


def _polar_R(F):
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    bad = np.linalg.det(R) < 0
    if bad.any():
        U[bad, :, -1] *= -1.0
        R = U @ Vt
    return R


def _E_nu(mu, lam):
    nu = lam / (2.0 * (lam + mu))
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    return E, nu


def _jac(mu, lam, h=1.0):
    """Numerical Jacobian of (E, nu) w.r.t. (mu, lam)."""
    f0 = np.array(_E_nu(mu, lam))
    jm = (np.array(_E_nu(mu + h, lam)) - f0) / h
    jl = (np.array(_E_nu(mu, lam + h)) - f0) / h
    return np.stack([jm, jl], axis=1)          # (2 QoI, 2 params)


def _E_nu_to_mulam(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def stream(dump="truth.npz", prior_E=1.2e5, prior_nu=0.30):
    d = np.load(OUT / dump)
    X = d["x"].astype(np.float64)
    v = d["v"].astype(np.float64)
    F = d["F"].astype(np.float64).reshape(X.shape[0], X.shape[1], 3, 3)
    Xr = d["X_ref"].astype(np.float64)
    V0 = d["vol"].astype(np.float64)
    rho0 = float(d["rho"]); g = d["g"].astype(np.float64); fdt = float(d["frame_dt"])
    mu_t, lam_t = float(d["mu"]), float(d["lam"]); T = X.shape[0]
    a = (v[2:] - v[:-2]) / (2.0 * fdt)

    # interior reference bumps (same as recover): radial window x low-order modes
    c = Xr.mean(0); r = np.linalg.norm(Xr - c, axis=1); Rb = r.max() * 1.02
    u = r / Rb; W = np.clip(1.0 - u ** 2, 0.0, None) ** 2
    dWdr = -4.0 * u / Rb * np.clip(1.0 - u ** 2, 0.0, None)
    rsafe = np.where(r > 1e-9, r, 1.0)
    gradW = (dWdr / rsafe)[:, None] * (Xr - c); dX = (Xr - c) / Rb
    modes = [np.ones(len(Xr)), dX[:, 0], dX[:, 1], dX[:, 2]]
    gmodes = [np.zeros((len(Xr), 3))] + [np.eye(3)[i][None, :] / Rb * np.ones((len(Xr), 1)) for i in range(3)]
    phis = [W * m for m in modes]
    gphis = [gradW * m[:, None] + W[:, None] * gm for m, gm in zip(modes, gmodes)]

    def frame_rows(t):
        Ft = F[t]; J = np.linalg.det(Ft); devF = np.linalg.norm(Ft - np.eye(3), axis=(1, 2))
        m = (J > 0.3) & (J < 2.0) & (devF < 0.8) & np.isfinite(J)
        if m.sum() < 50:
            return None, None, 0.0, 0.0
        Fm = Ft[m]; Jm = J[m]; R = _polar_R(Fm)
        Pmu = 2.0 * (Fm - R)
        Finvt = np.transpose(np.linalg.inv(Fm), (0, 2, 1))
        Plam = (Jm * (Jm - 1.0))[:, None, None] * Finvt
        Vm = V0[m]; at = a[t - 1][m]
        A, b = [], []
        for dirn in range(3):
            for phi, gphi in zip(phis, gphis):
                gp = gphi[m]; ph = phi[m]
                A.append([np.sum(Vm * np.einsum("pj,pj->p", Pmu[:, dirn, :], gp)),
                          np.sum(Vm * np.einsum("pj,pj->p", Plam[:, dirn, :], gp))])
                b.append(-np.sum(Vm * rho0 * (at[:, dirn] - g[dirn]) * ph))
        # deviatoric vs volumetric strain coverage this frame
        meaneps = np.trace(Fm - np.eye(3), axis1=1, axis2=2) / 3.0
        dev = float(np.mean(np.linalg.norm((Fm - np.eye(3)) - meaneps[:, None, None] * np.eye(3), axis=(1, 2))))
        vol = float(np.mean(np.abs(Jm - 1.0)))
        return np.asarray(A), np.asarray(b), dev, vol

    # estimate noise scale from the full-data fit
    allA, allb = [], []
    for t in range(1, T - 1):
        A, b, *_ = frame_rows(t)
        if A is not None:
            allA.append(A); allb.append(b)
    AA = np.vstack(allA); bb = np.concatenate(allb)
    th_full, *_ = np.linalg.lstsq(AA, bb, rcond=None)
    sigma2 = float(np.sum((AA @ th_full - bb) ** 2) / max(len(bb) - 2, 1))

    # Bayesian / RLS init from a nominal prior (in practice the FE corpus prior theta_mean/cov);
    # before any informative frame the estimate is the prior, then each measurement updates it.
    mu_p, lam_p = _E_nu_to_mulam(prior_E, prior_nu)
    prior_mean = np.array([mu_p, lam_p])
    prior_prec = np.diag([1.0 / (5 * mu_p) ** 2, 1.0 / (5 * lam_p) ** 2])   # weak prior (~500% std)
    M = np.zeros((2, 2)); rhs = np.zeros(2)
    rec = []
    for t in range(1, T - 1):
        A, b, dev, vol = frame_rows(t)
        if A is not None:
            M += A.T @ A / sigma2; rhs += A.T @ b / sigma2     # recursive information update
        prec = M + prior_prec
        Sig = np.linalg.inv(prec)
        th = Sig @ (rhs + prior_prec @ prior_mean)             # posterior mean (RLS w/ prior)
        mu_h, lam_h = th[0], th[1]
        if mu_h <= 0 or (lam_h + mu_h) <= 0:
            E_std = nu_std = np.nan; E_h = nu_h = np.nan
        else:
            E_h, nu_h = _E_nu(mu_h, lam_h)
            Jc = _jac(mu_h, lam_h)
            cov_qoi = Jc @ Sig @ Jc.T
            E_std = float(np.sqrt(max(cov_qoi[0, 0], 0))); nu_std = float(np.sqrt(max(cov_qoi[1, 1], 0)))
        ev = np.linalg.eigvalsh(M)            # data Fisher information eigenvalues (no prior)
        rec.append((t, t * fdt, E_h, E_std, nu_h, nu_std, dev, vol, ev.min(), ev.max()))
    return np.array(rec), (mu_t, lam_t)


def figure(dump="truth.npz"):
    """Bayesian sequential identification: posterior mean +/- 95% credible interval of E and nu
    vs frame. Starts at the (wide) FE prior, contracts to truth as informative frames arrive.
    The contraction is sharp for E (shear-driven) and weaker/biased for nu (bulk under-excited)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec, (mu_t, lam_t) = stream(dump)
    E_t, nu_t = _E_nu(mu_t, lam_t)
    t = rec[:, 1]; E_h, E_std, nu_h, nu_std, dev, vol = (rec[:, i] for i in (2, 3, 4, 5, 6, 7))
    below = np.where(100 * E_std / E_t < 5.0)[0]
    imp_t = float(rec[below[0], 1]) if len(below) else t[-1]

    fig, ax = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 2, 1.1], hspace=0.13))
    # E with 95% credible band (kPa)
    ax[0].fill_between(t, (E_h - 1.96 * E_std) / 1e3, (E_h + 1.96 * E_std) / 1e3,
                       color="#1c7ed6", alpha=0.25, label="95% credible interval")
    ax[0].plot(t, E_h / 1e3, color="#1c7ed6", lw=2.2, label="posterior mean E")
    ax[0].axhline(E_t / 1e3, color="k", ls="--", lw=1.5, label="truth E")
    ax[0].axvline(imp_t, color="0.7", lw=1.0)
    ax[0].set_ylim(0, 1.6 * E_t / 1e3); ax[0].set_ylabel("E  (kPa)")
    ax[0].set_title("Bayesian sequential identification (FE prior -> conjugate posterior, recursive)\n"
                    "credible band starts at the prior, contracts to truth once impact is observed",
                    fontsize=11)
    ax[0].legend(fontsize=9, loc="upper right"); ax[0].grid(alpha=0.3)
    # nu with band
    ax[1].fill_between(t, nu_h - 1.96 * nu_std, nu_h + 1.96 * nu_std, color="#e8590c", alpha=0.25,
                       label="95% credible interval")
    ax[1].plot(t, nu_h, color="#e8590c", lw=2.2, label="posterior mean nu")
    ax[1].axhline(nu_t, color="k", ls="--", lw=1.5, label="truth nu")
    ax[1].axvline(imp_t, color="0.7", lw=1.0)
    ax[1].set_ylim(0, 0.5); ax[1].set_ylabel("nu  (Poisson)")
    ax[1].legend(fontsize=9, loc="upper right"); ax[1].grid(alpha=0.3)
    ax[1].text(0.02, 0.06, "bulk under-excited -> nu biased low despite a TIGHT band "
               "(posterior variance misses the coverage bias)",
               transform=ax[1].transAxes, fontsize=8, color="#a33")
    # strain coverage (the cause)
    ax[2].plot(t, dev, color="#1c7ed6", lw=2.0, label="deviatoric strain (informs E)")
    ax[2].plot(t, vol, color="#e8590c", lw=2.0, label="volumetric strain |J-1| (informs nu)")
    ax[2].axvline(imp_t, color="0.7", lw=1.0)
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("strain / frame")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    p = ROOT / "out" / "nclaw_compare" / "elastic_bayesian_identify.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"wrote {p}")
    print(f"  E: prior {E_h[0]/1e3:.0f} kPa (+/-{1.96*E_std[0]/1e3:.0f}) -> posterior {E_h[-1]/1e3:.1f} kPa "
          f"(+/-{1.96*E_std[-1]/1e3:.2f}), truth {E_t/1e3:.0f}; confident at t={imp_t:.2f}s")
    print(f"  nu: posterior {nu_h[-1]:.3f} +/-{1.96*nu_std[-1]:.3f} (truth {nu_t:.2f}) -- wider/biased (bulk starved)")
    return p


def rollout_vs_frames(dump="truth.npz", checks=(60, 95, 105, 112, 122, 140, 200, 320, 450)):
    """Measure full-trajectory rollout error after recovery from the first N frames.

    The plot overlays the posterior standard deviation of E to test whether it can serve
    as an online proxy for rollout error.
    """
    from examples.recovery.elastic_drop import run_drop, _pos_err, TRUTH
    d = np.load(OUT / dump)
    shape = str(d["shape"]) if "shape" in d.files else "sphere"
    size = float(d["size"]) if "size" in d.files else 0.14
    rho = float(d["rho"])
    rec, (mu_t, lam_t) = stream(dump)
    E_t, _ = _E_nu(mu_t, lam_t)
    out = []
    for N in checks:
        row = rec[rec[:, 0] <= N][-1]
        E_N, E_std_N, nu_N = float(row[2]), float(row[3]), float(row[4])
        if not (E_N > 1e3 and 0.0 < nu_N < 0.49):
            out.append((N, np.nan, 100 * E_std_N / E_t)); continue
        rp = run_drop(E_N, nu_N, rho, OUT / f"_roll_{N}.npz", shape=shape, size=size, log=lambda *a: None)
        rmse, _, _ = _pos_err(OUT / dump, rp)
        out.append((N, rmse, 100 * E_std_N / E_t))
        (OUT / f"_roll_{N}.npz").unlink(missing_ok=True)
    out = np.array(out)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(out[:, 0], out[:, 1], "o-", color="#1c7ed6", lw=2.2, label="rollout prediction RMSE (mm)  [re-sim, expensive]")
    ax.set_xlabel("number of frames observed for identification (N)")
    ax.set_ylabel("rollout RMSE vs truth (mm)", color="#1c7ed6")
    ax.tick_params(axis="y", labelcolor="#1c7ed6"); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(out[:, 0], out[:, 2], "s--", color="#e8590c", lw=2.0, label="posterior std of E (%)  [online Fisher surrogate, cheap]")
    ax2.set_ylabel("posterior std of E (%)", color="#e8590c"); ax2.tick_params(axis="y", labelcolor="#e8590c")
    ax.set_title("Sequential elastic ID: rollout prediction error vs frames observed\n"
                 "free-fall frames carry no info; once impact/deformation is seen, both the "
                 "rollout error and\nthe (cheap) Fisher confidence collapse together -> the "
                 "surrogate is the stopping rule", fontsize=10)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=9, loc="upper right")
    fig.tight_layout()
    p = ROOT / "out" / "nclaw_compare" / "elastic_rollout_vs_frames.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)
    for N, rmse, estd in out:
        print(f"  N={int(N):3d} frames: rollout RMSE {rmse:6.2f} mm | online E-std {estd:6.2f}%")
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "rollout"
    if mode == "rollout":
        rollout_vs_frames()
    else:
        figure(mode + ".npz")
