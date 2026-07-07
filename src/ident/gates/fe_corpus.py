"""Function-encoder corpus identification and span-coverage report.

The 'simulate varied properties, then learn with the function encoder' program:
for a corpus of collapses with DIFFERENT mu(I) laws, identify each material's
constitutive curve from its collapse through the weak form using the LEARNED
function-encoder basis (Mode F), staying linear in theta. Reports the
per-material span-coverage table (true law, recovered relative L2 over the
observed band) and an overlay figure. Uses the grid-consistent engine with the
true MPM pressure (the clean oracle engine; the 'isolate perception' choice).

The learned basis is frozen (mpm_engine/fe-weights/granular_mu_i.npz); this is pure
identification of each material through that basis. Pure ident.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ident.features.function_encoder import FunctionEncoderDict
from ident.io.schema import load_dump
from ident.solve.qp import constrained_solve
from ident.weakform.from_dump import build_inplane_frames
from ident.weakform.grid_assembly import assemble_grid_consistent


def _load_fe(path="mpm_engine/fe-weights/granular_mu_i.npz") -> FunctionEncoderDict:
    d = np.load(path)
    return FunctionEncoderDict(d["s_grid"], d["table"])


def identify_one(dump_path, fe: FunctionEncoderDict, lam=1e-6, refuse_thresh=0.5):
    dump = load_dump(dump_path)
    meta = dump.meta
    bundle = build_inplane_frames(dump)
    gs = assemble_grid_consistent(dump, fe, flow_frac_min=0.90)
    if gs.n_rows == 0:
        return None
    Ilo = max(bundle.I_observed[0], 1e-3)
    Ihi = max(min(bundle.I_observed[1], 1.0), Ilo * 10)
    Icon = np.logspace(np.log10(Ilo), np.log10(Ihi), 40)
    G = fe.gram((10.0 ** np.linspace(-4, 0, 257), np.ones(257)))
    qp = constrained_solve(gs.A, gs.b, fe, lam=lam, G=G, mu_min=0.05,
                           I_constraint_grid=Icon, nonnegativity=False, monotonic=True)
    Igrid = np.logspace(np.log10(Ilo), np.log10(Ihi), 80)
    mu_hat = fe.phi(Igrid) @ qp.theta

    rec = {"dump": str(dump_path), "law": meta.law, "I_band": [Ilo, Ihi],
           "n_rows": gs.n_rows, "qp_status": qp.status,
           "residual_rel": qp.residual_rel, "Igrid": Igrid, "mu_hat": mu_hat}
    if meta.law == "pouliquen":
        lp = meta.law_params
        mu_true = lp["mu_s"] + lp["delta_mu"] * Igrid / (Igrid + lp["I0"])
        rec["true_params"] = {"mu_s": lp["mu_s"], "delta_mu": lp["delta_mu"], "I0": lp["I0"]}
        rec["mu_true"] = mu_true
        rec["curve_relL2"] = float(np.sqrt(np.mean((mu_hat - mu_true) ** 2))
                                   / np.sqrt(np.mean(mu_true ** 2)))
    elif meta.law == "constant":
        mu_s = meta.law_params["mu_s"]
        rec["true_params"] = {"mu_s": mu_s}
        rec["mu_true"] = np.full_like(Igrid, mu_s)
        rec["curve_relL2"] = float(np.sqrt(np.mean((mu_hat - mu_s) ** 2)) / mu_s)
    # refusal: weak-form projection residual flags model/observability failure
    rec["refused"] = bool(qp.residual_rel > refuse_thresh)
    return rec


def run_corpus(dumps, out_dir="out/corpus", fe_path="mpm_engine/fe-weights/granular_mu_i.npz"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fe = _load_fe(fe_path)
    recs = [r for r in (identify_one(p, fe) for p in dumps) if r is not None]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(recs)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), dpi=130,
                             squeeze=False)
    table = []
    for i, rec in enumerate(recs):
        ax = axes[i // ncol][i % ncol]
        ax.semilogx(rec["Igrid"], rec["mu_true"], "k-", lw=2, label="truth")
        ax.semilogx(rec["Igrid"], rec["mu_hat"], "r--", lw=1.8, label="Mode F")
        tp = rec["true_params"]
        ttl = ", ".join(f"{k}={v:.2f}" for k, v in tp.items())
        ax.set_title(f"{ttl}\nrelL2={rec.get('curve_relL2', float('nan')):.3f}",
                     fontsize=8)
        ax.grid(alpha=0.3, which="both"); ax.set_xlabel("I", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
        table.append({"dump": Path(rec["dump"]).stem, "law": rec["law"],
                      "true_params": tp, "I_band": rec["I_band"],
                      "curve_relL2": rec.get("curve_relL2"),
                      "residual_rel": rec["residual_rel"], "refused": rec["refused"]})
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Function-encoder span-coverage: learn each material's mu(I) from its collapse")
    fig.tight_layout()
    figp = out_dir / "span_coverage.png"
    fig.savefig(figp); plt.close(fig)

    worst = max((t["curve_relL2"] for t in table if t["curve_relL2"] is not None), default=None)
    report = {"n_materials": len(table), "worst_curve_relL2": worst,
              "table": table, "figure": str(figp)}
    with open(out_dir / "span_coverage.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dumps = sys.argv[1:]
    else:
        dumps = ["out/dumps/column_constant_a2.npz",
                 "out/dumps/column_pouliquen_a2.npz",
                 "out/dumps/corpus_matA.npz", "out/dumps/corpus_matB.npz",
                 "out/dumps/corpus_matC.npz", "out/dumps/corpus_matD.npz"]
    dumps = [d for d in dumps if Path(d).exists()]
    rep = run_corpus(dumps)
    print(json.dumps({k: v for k, v in rep.items() if k != "table"}, indent=2, default=float))
    for t in rep["table"]:
        print(f"  {t['dump']:24s} {t['law']:9s} relL2={t['curve_relL2']:.3f} "
              f"resid={t['residual_rel']:.2f} refused={t['refused']}")
