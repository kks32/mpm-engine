"""Sample-complexity experiment: recovery error vs number of weak-form samples, against the
Gauss-Markov bound, on the elastic drop (shear mode excited, bulk mode starved).

The least-squares estimator of A theta = b has covariance sigma^2 (A^T A)^{-1} (Gauss-Markov). In the
eigenbasis of A^T A the per-mode standard deviation is sigma / sqrt(lambda_k), and lambda_k grows
linearly with the number of rows N, so the per-mode error falls as 1/sqrt(N) with a rate set by the
per-sample Fisher information of that mode. A mode the motion does not excite has lambda_k ~ 0 and does
not improve with N: that is the refusal. We verify this directly: assemble the full FCR weak form
(columns P_mu, P_lambda), draw N-row subsamples, recover (mu, lambda), and compare the empirical
spread to the Gauss-Markov prediction. The shear mode (large Fisher eigenvalue) tracks 1/sqrt(N) to
its bias floor; the bulk mode (tiny eigenvalue) stays high regardless of N.

Run:  python -m examples.recovery.sample_complexity
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.recovery.elastic_drop import _polar_R

OUT = ROOT / "out" / "elastic_drop"
RNG = np.random.default_rng(0)


def _assemble_fcr(dump_path, n_modes=4):
    """Full FCR weak-form rows: A (Nrows,2) columns (P_mu, P_lambda), load b, truth (mu,lambda)."""
    d = np.load(dump_path)
    X = d["x"].astype(np.float64); v = d["v"].astype(np.float64)
    F = d["F"].astype(np.float64).reshape(X.shape[0], X.shape[1], 3, 3)
    Xr = d["X_ref"].astype(np.float64); V0 = d["vol"].astype(np.float64)
    rho0 = float(d["rho"]); g = d["g"].astype(np.float64); fdt = float(d["frame_dt"]); T = X.shape[0]
    mu_t, lam_t = float(d["mu"]), float(d["lam"])
    a = (v[2:] - v[:-2]) / (2.0 * fdt); Frng = np.arange(1, T - 1)
    c = Xr.mean(0); r = np.linalg.norm(Xr - c, axis=1); Rb = r.max() * 1.02
    u = r / Rb; W = np.clip(1.0 - u ** 2, 0.0, None) ** 2
    dWdr = -4.0 * u / Rb * np.clip(1.0 - u ** 2, 0.0, None); rsafe = np.where(r > 1e-9, r, 1.0)
    gradW = (dWdr / rsafe)[:, None] * (Xr - c); dX = (Xr - c) / Rb
    modes = [np.ones(len(Xr))] + [dX[:, i] for i in range(3)][: max(0, n_modes - 1)]
    gmodes = [np.zeros((len(Xr), 3))] + [np.eye(3)[i][None, :] / Rb * np.ones((len(Xr), 1))
                                         for i in range(3)][: max(0, n_modes - 1)]
    phis = [W * m for m in modes]
    gphis = [gradW * m[:, None] + W[:, None] * gm for m, gm in zip(modes, gmodes)]
    rows_A, rows_b = [], []
    for ti, t in enumerate(Frng):
        Ft = F[t]
        sig = np.clip(np.linalg.svd(Ft, compute_uv=False), 1e-9, None)
        J = sig.prod(-1); hk = np.linalg.norm(np.log(sig), axis=1)
        m = (J > 0.3) & (J < 2.0) & (hk < 0.8) & np.isfinite(J)
        if m.sum() < 50:
            continue
        Fm = Ft[m]; Jm = J[m]; R = _polar_R(Fm)
        Pmu = 2.0 * (Fm - R)
        Finvt = np.transpose(np.linalg.inv(Fm), (0, 2, 1))
        Plam = (Jm * (Jm - 1.0))[:, None, None] * Finvt
        Vm = V0[m]; at = a[ti][m]
        for dirn in range(3):
            for phi, gphi in zip(phis, gphis):
                gp = gphi[m]; ph = phi[m]
                rows_A.append([np.sum(Vm * np.einsum("pj,pj->p", Pmu[:, dirn, :], gp)),
                               np.sum(Vm * np.einsum("pj,pj->p", Plam[:, dirn, :], gp))])
                rows_b.append(-np.sum(Vm * rho0 * (at[:, dirn] - g[dirn]) * ph))
    return np.asarray(rows_A), np.asarray(rows_b), np.array([mu_t, lam_t])


def main(dump_path=str(OUT / "box_truth.npz")):
    A, b, theta_t = _assemble_fcr(dump_path)
    Nfull = A.shape[0]
    th_full, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ th_full - b
    sigma = float(np.sqrt(resid @ resid / (Nfull - 2)))   # residual noise scale
    # CLEAN Gauss-Markov validation by NOISE INJECTION: take the first N rows, build a clean signal
    # b0 = A_N th_full, add iid N(0,sigma^2) noise, recover, repeat. The empirical covariance must
    # equal sigma^2 (A_N^T A_N)^{-1} EXACTLY (Gauss-Markov is exact for the linear-Gaussian model),
    # so per-mode std = sigma / sqrt(lambda_k(N)) and falls as 1/sqrt(N). No finite-population issue.
    Ns = np.unique(np.round(np.logspace(np.log10(60), np.log10(Nfull), 12)).astype(int))
    B = 400
    out = {"N": [], "mu_err": [], "lam_err": [], "mu_std": [], "lam_std": [],
           "mu_gm_std": [], "lam_gm_std": []}
    R = 24                                                # random representative subsets per N
    for N in Ns:
        emp_list, gm_list, err_list = [], [], []
        for _ in range(R):
            idx = RNG.choice(Nfull, size=N, replace=False)
            AN = A[idx]; bN = b[idx]
            AtA = AN.T @ AN
            if np.linalg.cond(AtA) > 1e14:
                continue
            gm = np.sqrt(np.diag(sigma ** 2 * np.linalg.inv(AtA)))
            b0 = AN @ th_full
            ths = np.array([np.linalg.lstsq(AN, b0 + RNG.normal(0, sigma, N), rcond=None)[0]
                            for _ in range(B // R + 1)])
            emp_list.append(ths.std(0)); gm_list.append(gm)
            tN, *_ = np.linalg.lstsq(AN, bN, rcond=None)
            err_list.append([abs(tN[0] / theta_t[0] - 1), abs(tN[1] / theta_t[1] - 1)])
        emp = np.mean(emp_list, 0); gm = np.mean(gm_list, 0); err = np.mean(err_list, 0)
        out["N"].append(int(N))
        out["mu_err"].append(float(err[0])); out["lam_err"].append(float(err[1]))
        out["mu_std"].append(float(emp[0])); out["lam_std"].append(float(emp[1]))
        out["mu_gm_std"].append(float(gm[0])); out["lam_gm_std"].append(float(gm[1]))
    evals = np.linalg.eigvalsh(A.T @ A)
    out["fisher_eigs"] = sorted(evals.tolist())
    out["fisher_ratio"] = float(evals.max() / max(evals.min(), 1e-30))
    out["sigma"] = sigma; out["Nfull"] = Nfull
    out["mu_truth"], out["lam_truth"] = float(theta_t[0]), float(theta_t[1])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample_complexity.json").write_text(json.dumps(out, indent=2))
    print(f"Nfull={Nfull} sigma={out['sigma']:.3e} Fisher eig ratio (shear/bulk)={out['fisher_ratio']:.1e}")
    print(f"{'N':>6} {'mu_err%':>8} {'mu_std':>10} {'mu_GMstd':>10} | {'lam_err%':>8} {'lam_std':>10} {'lam_GMstd':>10}")
    for i, N in enumerate(out["N"]):
        print(f"{N:6d} {100*out['mu_err'][i]:8.1f} {out['mu_std'][i]:10.2e} {out['mu_gm_std'][i]:10.2e} | "
              f"{100*out['lam_err'][i]:8.1f} {out['lam_std'][i]:10.2e} {out['lam_gm_std'][i]:10.2e}")
    _figure(out)
    return out


def _figure(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = np.array(out["N"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # Panel A: empirical estimator spread vs N, with Gauss-Markov 1/sqrt(N) prediction
    ax[0].loglog(N, out["mu_std"], "o", color="#2b8cbe", label=r"shear $\mu$ (empirical std)")
    ax[0].loglog(N, out["mu_gm_std"], "-", color="#2b8cbe", label=r"Gauss-Markov $\sigma/\sqrt{\lambda_\mu}$")
    ax[0].loglog(N, out["lam_std"], "s", color="#e34a33", label=r"bulk $\lambda$ (empirical std)")
    ax[0].loglog(N, out["lam_gm_std"], "-", color="#e34a33", label=r"Gauss-Markov $\sigma/\sqrt{\lambda_\lambda}$")
    ax[0].set_xlabel("number of weak-form samples $N$"); ax[0].set_ylabel("estimator standard deviation")
    ax[0].legend(fontsize=8); ax[0].set_title("Estimator spread follows the Gauss-Markov bound\n(empirical std vs prediction)", fontsize=10)
    # Panel B: error-to-truth vs N (decays to the bias floor; bulk mode starved)
    ax[1].loglog(N, 100 * np.array(out["mu_err"]), "o-", color="#2b8cbe", label=r"shear $\mu$")
    ax[1].loglog(N, 100 * np.array(out["lam_err"]), "s-", color="#e34a33", label=r"bulk $\lambda$")
    ax[1].set_xlabel("number of weak-form samples $N$"); ax[1].set_ylabel("error to truth (%)")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which="both")
    ax[1].set_title(f"Excited mode improves with $N$; starved mode does not\n"
                    f"Fisher eigenvalue ratio $\\lambda_\\mu/\\lambda_\\lambda={out['fisher_ratio']:.0e}$", fontsize=10)
    fig.tight_layout()
    p = ROOT / "docs/writeup/figs/sample_complexity.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
