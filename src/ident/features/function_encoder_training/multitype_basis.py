"""Multi-type function-encoder basis: can ONE learned basis span many material families?

We represent every material -- elastic, plastic, viscous, granular -- by its STRESS RESPONSE
over a canonical PROBE BATTERY (four blocks), giving a universal fixed-length fingerprint:

  vol(J)        volumetric stress vs J          (bulk / compressibility)
  shear(gamma)  dev stress vs simple-shear      (elastic shear modulus; plastic yield plateau)
  rate(gdot)    dev stress vs shear RATE        (viscosity / rate dependence)
  fric(I)       mu(I) vs inertial number        (granular friction)

Each family lights up a different subset of blocks with a different shape, so a material is a
point on a low-dimensional constitutive manifold. We sample ~16 types x parameter ranges, build
the fingerprint corpus, and learn a single basis (weighted SVD). We report (a) how many basis
dimensions span all types to a tolerance, (b) held-out reconstruction error, (c) whether the
types separate in coefficient space (nearest-centroid classification).

This is the offline "space of possible solutions" over MANY material types. At recovery time the
weak-form solve constrains only the blocks the loading excited (coverage); the rest are filled by
the basis prior or refused.

Pure numpy. Run:  .venv/bin/python -m ident.features.function_encoder_training.multitype_basis
"""
from __future__ import annotations

import numpy as np

# ---- probe battery (fixed grids) ----
NJ, NG, NR, NF = 10, 10, 10, 10
J_GRID = np.linspace(0.88, 1.12, NJ)
G_GRID = np.linspace(0.0, 0.30, NG)               # simple-shear strain
R_GRID = np.logspace(-2, 2, NR)                   # shear rate
I_GRID = np.logspace(-3.5, -0.3, NF)              # inertial number
EPS = 1e-3
P_REF = 1.0e3                                     # reference pressure for friction block


def _vol(K):                                      # volumetric stress vs J
    return K * (1.0 - J_GRID)


def _shear_elastic(G, tau_y=np.inf):              # dev stress vs shear strain (cap at yield)
    return np.minimum(G * G_GRID, tau_y)


def _rate(eta, tau_y, K, n):                      # Herschel-Bulkley-family dev stress vs rate
    g = np.sqrt(R_GRID ** 2 + EPS ** 2)
    return tau_y + eta * R_GRID + K * g ** n


def _fric(mu_s, dmu, I0):                          # mu(I) * P_REF
    return (mu_s + dmu * I_GRID / (I_GRID + I0)) * P_REF


Z = np.zeros


# ---- ~16 material types: each returns (vol, shear, rate, fric) blocks ----
def material(kind, rng):
    K = 10.0 ** rng.uniform(4.5, 6.0)             # bulk modulus (shared scale)
    if kind in ("neohookean", "fcr", "stvk", "mooney", "hookean"):
        G = 10.0 ** rng.uniform(4.0, 5.5)
        bend = {"neohookean": 1.0, "fcr": 1.05, "stvk": 1.15, "mooney": 0.9, "hookean": 1.0}[kind]
        return _vol(K), _shear_elastic(G) * bend, Z(NR), Z(NF)
    if kind in ("vonmises", "druckerprager", "camclay"):
        G = 10.0 ** rng.uniform(4.0, 5.5); tau_y = 10.0 ** rng.uniform(3.0, 4.3)
        vol = _vol(K) * (1.3 if kind == "camclay" else 1.0)
        return vol, _shear_elastic(G, tau_y), Z(NR), Z(NF)
    if kind == "newtonian":
        return _vol(K), Z(NG), _rate(10.0 ** rng.uniform(0, 2), 0, 0, 1), Z(NF)
    if kind in ("powerlaw_thin", "powerlaw_thick"):
        n = rng.uniform(0.3, 0.8) if kind == "powerlaw_thin" else rng.uniform(1.2, 1.8)
        return _vol(K), Z(NG), _rate(0, 0, 10.0 ** rng.uniform(1, 2.5), n), Z(NF)
    if kind == "bingham":
        return _vol(K), Z(NG), _rate(10.0 ** rng.uniform(0, 1.5), 10.0 ** rng.uniform(1, 2.5), 0, 1), Z(NF)
    if kind == "herschel":
        return _vol(K), Z(NG), _rate(0, 10.0 ** rng.uniform(1, 2.5), 10.0 ** rng.uniform(1, 2),
                                     rng.uniform(0.3, 0.8)), Z(NF)
    if kind == "carreau":
        n = rng.uniform(0.3, 0.7); return _vol(K), Z(NG), _rate(0, 0, 10.0 ** rng.uniform(1, 2.5), n), Z(NF)
    if kind == "mu_i":
        return _vol(K) * 0.3, Z(NG), Z(NR), _fric(rng.uniform(0.2, 0.45),
                                                  rng.uniform(0.1, 0.5), 10.0 ** rng.uniform(-2, 0))
    if kind == "mu_i_phi":
        return _vol(K) * 0.5, Z(NG), Z(NR), _fric(rng.uniform(0.2, 0.4),
                                                  rng.uniform(0.15, 0.5), 10.0 ** rng.uniform(-2, -0.5))
    if kind == "coulomb":
        return _vol(K) * 0.3, Z(NG), Z(NR), _fric(rng.uniform(0.25, 0.55), 0.0, 1.0)
    raise ValueError(kind)


TYPES = ["neohookean", "fcr", "stvk", "mooney", "hookean",        # elastic (5)
         "vonmises", "druckerprager", "camclay",                  # plastic (3)
         "newtonian", "powerlaw_thin", "powerlaw_thick", "bingham", "herschel", "carreau",  # viscous (6)
         "mu_i", "mu_i_phi", "coulomb"]                           # granular (3)  -> 17 types
FAMILY = ({**{k: 0 for k in TYPES[:5]}, **{k: 1 for k in TYPES[5:8]},
           **{k: 2 for k in TYPES[8:14]}, **{k: 3 for k in TYPES[14:]}})
FAM_NAMES = ["elastic", "plastic", "viscous", "granular"]


def fingerprint(kind, rng):
    return np.concatenate(material(kind, rng))


def build_corpus(n_per_type=150, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ti, k in enumerate(TYPES):
        for _ in range(n_per_type):
            X.append(fingerprint(k, rng)); y.append(ti)
    return np.asarray(X), np.asarray(y)


def main():
    X, y = build_corpus()
    N, D = X.shape
    # per-block RMS normalization so blocks are comparable
    scale = np.sqrt((X ** 2).mean(0)) + 1e-12
    Xn = X / scale
    rng = np.random.default_rng(1)
    perm = rng.permutation(N); ntr = int(0.8 * N)
    tr, te = perm[:ntr], perm[ntr:]
    mean = Xn[tr].mean(0)
    U, S, Vt = np.linalg.svd(Xn[tr] - mean, full_matrices=False)
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)

    print(f"corpus: {N} materials, {len(TYPES)} types, fingerprint dim {D}  (blocks vol/shear/rate/fric = {NJ}/{NG}/{NR}/{NF})")
    for thr in (0.99, 0.999, 0.9999):
        k = int(np.searchsorted(energy, thr) + 1)
        print(f"  basis dims for {thr*100:.2f}% energy: K={k}")
    for K in (8, 16, 24, 32):
        B = Vt[:K]
        coeff_te = (Xn[te] - mean) @ B.T
        recon = coeff_te @ B + mean
        rel = np.linalg.norm(Xn[te] - recon, axis=1) / (np.linalg.norm(Xn[te] - mean, axis=1) + 1e-12)
        # type separation: nearest train-centroid in coeff space
        coeff_tr = (Xn[tr] - mean) @ B.T
        cents = np.stack([coeff_tr[y[tr] == t].mean(0) for t in range(len(TYPES))])
        pred = np.argmin(((coeff_te[:, None, :] - cents[None]) ** 2).sum(-1), axis=1)
        acc = (pred == y[te]).mean()
        fam = np.array([FAMILY[TYPES[t]] for t in range(len(TYPES))])
        fcent = np.stack([coeff_tr[fam[y[tr]] == f].mean(0) for f in range(4)])
        fpred = np.argmin(((coeff_te[:, None, :] - fcent[None]) ** 2).sum(-1), axis=1)
        facc = (fpred == fam[y[te]]).mean()
        print(f"  K={K:2d}: recon relL2 mean {rel.mean():.2e} worst {rel.max():.2e} | "
              f"family acc {facc*100:.1f}%  type acc {acc*100:.1f}%")
    return Xn, mean, U, S, Vt, te, mean


def figure():
    """Approximation-bound figure: the FE basis truncation error over 17 material types is the
    SVD tail epsilon_approx(K)^2 = sum_{k>K} s_k^2 -- a hard, computable bound. Plot the singular
    spectrum and the held-out reconstruction error vs K."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, y = build_corpus()
    scale = np.sqrt((X ** 2).mean(0)) + 1e-12
    Xn = X / scale
    rng = np.random.default_rng(1); perm = rng.permutation(len(Xn)); ntr = int(0.8 * len(Xn))
    tr, te = perm[:ntr], perm[ntr:]
    mean = Xn[tr].mean(0)
    U, S, Vt = np.linalg.svd(Xn[tr] - mean, full_matrices=False)
    tail = np.sqrt(np.cumsum((S ** 2)[::-1])[::-1] / (S ** 2).sum())   # sqrt SVD tail vs K (bound)
    Ks = np.arange(1, min(33, len(S)) + 1)
    recon = []
    for K in Ks:
        B = Vt[:K]; r = (Xn[te] - mean) @ B.T @ B + mean
        recon.append(np.median(np.linalg.norm(Xn[te] - r, axis=1) / (np.linalg.norm(Xn[te] - mean, axis=1) + 1e-12)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].semilogy(np.arange(1, len(S) + 1), S / S[0], "o-", color="#1c7ed6", ms=3)
    ax[0].set_xlabel("singular index k"); ax[0].set_ylabel("s_k / s_1")
    ax[0].set_title("Constitutive-manifold spectrum (17 material types)\nintrinsic dim ~ 12-24")
    ax[0].grid(alpha=0.3, which="both")
    ax[1].semilogy(Ks, [tail[k - 1] for k in Ks], "s--", color="#888", label="bound: sqrt SVD tail")
    ax[1].semilogy(Ks, recon, "o-", color="#e8590c", label="held-out recon (median relL2)")
    ax[1].axhline(1e-2, color="0.7", ls=":"); ax[1].set_ylim(1e-6, 1)
    ax[1].set_xlabel("basis dimension K"); ax[1].set_ylabel("reconstruction error")
    ax[1].set_title("Approximation bound: truncation error vs K\n(elastic/plastic/viscous/granular)")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = "out/nclaw_compare/fe_multitype_bounds.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)
    return p


def _oof_fingerprints(n=60, seed=7):
    """Laws that look admissible but are OUTSIDE the corpus family (the refusal targets):
    velocity-WEAKENING friction (mu decreasing in I), non-monotone (bumped) viscosity, an
    oscillatory friction law, and a rate+friction HYBRID no single family has."""
    rng = np.random.default_rng(seed)
    X = []
    for _ in range(n):
        kind = rng.integers(4)
        K = 10.0 ** rng.uniform(4.5, 6.0)
        if kind == 0:                                   # velocity-weakening mu(I) (decreasing)
            mu_s = rng.uniform(0.35, 0.6); dmu = rng.uniform(0.1, 0.4); I0 = 10.0 ** rng.uniform(-2, 0)
            fric = (mu_s - dmu * I_GRID / (I_GRID + I0)) * P_REF
            X.append(np.concatenate([_vol(K), Z(NG), Z(NR), fric]))
        elif kind == 1:                                 # non-monotone (thin-then-thick) viscosity
            r = _rate(0, 0, 10.0 ** rng.uniform(1, 2), rng.uniform(0.3, 0.6)) * (1 + 0.8 * np.sin(3 * np.log(R_GRID)))
            X.append(np.concatenate([_vol(K), Z(NG), r, Z(NF)]))
        elif kind == 2:                                 # oscillatory friction
            fric = (rng.uniform(0.3, 0.5) + 0.15 * np.sin(4 * np.log10(I_GRID))) * P_REF
            X.append(np.concatenate([_vol(K) * 0.3, Z(NG), Z(NR), fric]))
        else:                                           # hybrid: strong rate AND friction blocks
            X.append(np.concatenate([_vol(K), Z(NG),
                                     _rate(10.0 ** rng.uniform(0, 2), 0, 0, 1),
                                     _fric(rng.uniform(0.2, 0.45), rng.uniform(0.1, 0.5), 10.0 ** rng.uniform(-2, 0))]))
    return np.asarray(X)


def out_of_family():
    """Refusal demo: the FE-basis projection residual (distance to subspace, Bound 1) separates
    IN-family held-out laws from OUT-of-family laws -> a refusal/abstention gate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, y = build_corpus()
    scale = np.sqrt((X ** 2).mean(0)) + 1e-12; Xn = X / scale
    rng = np.random.default_rng(1); perm = rng.permutation(len(Xn)); ntr = int(0.8 * len(Xn))
    tr, te = perm[:ntr], perm[ntr:]
    mean = Xn[tr].mean(0)
    U, S, Vt = np.linalg.svd(Xn[tr] - mean, full_matrices=False)
    K = 24; B = Vt[:K]
    def relproj(M):
        c = (M - mean) @ B.T; rec = c @ B + mean
        return np.linalg.norm(M - rec, axis=1) / (np.linalg.norm(M - mean, axis=1) + 1e-12)
    in_err = relproj(Xn[te])
    oof_raw = _oof_fingerprints(); oof = oof_raw / scale
    oof_err = relproj(oof)
    thr = float(np.percentile(in_err, 99) * 3)          # refusal threshold: 3x the in-family p99

    # second refusal layer: a physically admissible granular/viscous law is monotone non-decreasing
    # in its friction (I) and rate blocks; velocity-weakening/oscillatory/non-monotone violate it.
    def nonmono(M):                                     # M unscaled fingerprints (rows)
        rate = M[:, NJ + NG:NJ + NG + NR]; fric = M[:, NJ + NG + NR:]
        bad = np.zeros(len(M), bool)
        for blk in (rate, fric):
            active = np.abs(blk).max(1) > 1e-6 * np.abs(M).max()
            viol = (np.diff(blk, axis=1) < -1e-3 * (np.abs(blk).max(1, keepdims=True) + 1e-9)).any(1)
            bad |= active & viol
        return bad
    X_te_raw = (Xn[te] * scale)
    in_refuse = (in_err > thr) | nonmono(X_te_raw)
    oof_refuse = (oof_err > thr) | nonmono(oof_raw)
    tpr = float(oof_refuse.mean()); fpr = float(in_refuse.mean())
    tpr_basis = float((oof_err > thr).mean())
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.hist(np.log10(in_err + 1e-12), bins=40, color="#2f9e44", alpha=0.7, label=f"in-family held-out (accept, median {np.median(in_err):.1e})")
    ax.hist(np.log10(oof_err + 1e-12), bins=40, color="#e8590c", alpha=0.7, label=f"out-of-family (refuse, median {np.median(oof_err):.1e})")
    ax.axvline(np.log10(thr), color="k", ls="--", lw=1.6, label="basis-residual threshold")
    ax.set_xlabel("log10  FE-basis projection residual  (distance to subspace)")
    ax.set_ylabel("count")
    ax.set_title("Layered refusal / out-of-family detection\n"
                 f"basis residual catches {tpr_basis*100:.0f}%; + monotonicity layer -> {tpr*100:.0f}% caught, "
                 f"{fpr*100:.0f}% false-refuse", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3); fig.tight_layout()
    p = "out/nclaw_compare/fe_refusal_oof.png"; fig.savefig(p, dpi=140); plt.close(fig)
    print(f"wrote {p}")
    print(f"  in-family residual: median {np.median(in_err):.2e}, p99 {np.percentile(in_err,99):.2e}")
    print(f"  out-of-family residual: median {np.median(oof_err):.2e}")
    print(f"  basis-residual alone: TPR {tpr_basis*100:.0f}%")
    print(f"  basis + monotonicity layers: TPR {tpr*100:.0f}%, FPR {fpr*100:.0f}%")
    return p


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "main"
    if arg == "figure":
        figure()
    elif arg == "oof":
        out_of_family()
    else:
        main()
