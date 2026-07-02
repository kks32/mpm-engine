"""#75 -- force-based identification of von-Mises (G, yield) from ONE robot squeeze probe.

The observable is the PLATE REACTION FORCE (grid-impulse, the wrist-F/T analog) plus kinematics;
no constitutive parameter is read from the dumped stress except a small volumetric (pressure)
correction, exactly as the viscoplastic squeeze recovery does (squeeze_plate_franka.py).

Mechanical power balance over the squeeze (P does no NET unknown work; the volumetric part is a
known elastic correction):
    P_internal(t) = P_plate + P_gravity - dKE/dt           (measured, force + kinematics)
    P_internal - P_vol = G * c1(t)  +  yield * c2(t)
        c1 = INT 2 dev(eps):dev(D) dV0   (elastic deviatoric power coefficient, linear in G)
        c2 = INT ||dev D||_F dV0         (plastic dissipation coefficient: ||dev tau||=yield at flow)
Two windows make it a pair of convex 1-parameter regressions:
    elastic window (early, pre-yield):  c2 ~ 0  -> G   = sum(P_internal-P_vol) / sum(c1)
    plastic window (late, steady flow):           yield = sum(P_internal-P_vol-G c1) / sum(c2)

The yield recovery rides on the fork's exact yield convention (||dev Kirchhoff||>yield_stress), so
it is robust to the StVK elastic-predictor stress factor (which only biases the elastic G slightly).

Run:  ../.venv/bin/python examples/vonmises_identify.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver
from warpmpm.materials import vonmises
from warpmpm.scenes import block

OUT = Path(__file__).resolve().parents[1] / "out" / "vonmises_identify"


def _hencky_dev(F):
    """Deviatoric Hencky strain tensors eps_dev (N,3,3) from F (N,3,3) via SVD."""
    U, s, Vt = np.linalg.svd(F)
    eps_i = np.log(np.clip(np.abs(s), 1e-6, None))                 # principal Hencky (N,3)
    eps = np.einsum("nik,nk,njk->nij", U, eps_i, U)                # U diag(log sig) U^T
    tr = eps_i.sum(1)
    return eps - (tr / 3.0)[:, None, None] * np.eye(3)[None]


def _dev(T):
    tr = np.trace(T, axis1=1, axis2=2)
    return T - (tr / 3.0)[:, None, None] * np.eye(3)[None]


def probe(G_true_E=5e5, nu=0.30, yield_true=3000.0, density=1000.0, bulk=9e5,
          n_grid=32, ppc=2, v_plate=0.08, n_frames=220, sub=4, compress=0.014,
          size=(0.12, 0.08, 0.06), device="auto"):
    """Run one squeeze probe; return per-frame power-balance quantities. `size` sets the block
    (object instance) -- the identified (G, yield) is a MATERIAL property and must not depend on it."""
    g = GridConfig(n_grid=n_grid, grid_lim=0.30)
    pos, vol0, floor = block(g, size=size, center=(0.15, 0.15, 0.05), ppc=ppc)
    N = len(pos); m = density * vol0
    dx = g.dx; dt = 2e-4; dt_ctrl = dt * sub
    s = Solver(g, device=device).load_particles(pos, vol0).set_material(
        vonmises(E=G_true_E, nu=nu, yield_stress=yield_true, density=density))
    s.add_plane(point=(0, 0, floor), normal=(0, 0, 1), surface="sticky")
    half = (size[0] / 2 + 0.01, size[1] / 2 + 0.01, 2 * dx); ztop = floor + size[2]; zc = ztop + half[2]
    h = s.add_box(center=(0.15, 0.15, zc), half_size=half, velocity=(0, 0, -v_plate))
    npress = int(compress / v_plate / dt_ctrl)
    npress = min(npress, n_frames)
    g_acc = np.array([0.0, 0.0, -9.81])
    rec = {k: [] for k in ("P_plate", "P_grav", "dKE", "P_vol", "c1", "c2", "disp", "Fz")}
    KE_prev = 0.5 * float((m[:, None] * s.v() ** 2).sum())
    z0 = float(pos[:, 2].max()); disp = 0.0
    for f in range(npress):
        zc -= v_plate * dt_ctrl; disp += v_plate * dt_ctrl
        s.reset_tool_force(h)
        s.set_box(h, center=(0.15, 0.15, zc + v_plate * dt_ctrl), velocity=(0, 0, -v_plate))
        s.step(dt, substeps=sub)
        Fz = float(s.tool_force(h, dt_ctrl)[2])                     # +z reaction (compression)
        v = s.v(); F = s.F(); L = s.L(); cauchy = s.cauchy(); volc = s.vol()
        KE = 0.5 * float((m[:, None] * v ** 2).sum())
        D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
        devD = _dev(D); deveps = _hencky_dev(F)
        p_or = -np.trace(cauchy, axis1=1, axis2=2) / 3.0           # oracle Cauchy pressure
        trD = np.trace(D, axis1=1, axis2=2)
        rec["P_plate"].append(Fz * v_plate)
        rec["P_grav"].append(float((m * (v @ g_acc)).sum()))
        rec["dKE"].append((KE - KE_prev) / dt_ctrl); KE_prev = KE
        rec["P_vol"].append(float((-p_or * trD * volc).sum()))     # current-config volumetric power
        rec["c1"].append(float((2.0 * np.einsum("nij,nij->n", deveps, devD) * vol0).sum()))
        rec["c2"].append(float((np.linalg.norm(devD, axis=(1, 2)) * vol0).sum()))
        rec["disp"].append(disp); rec["Fz"].append(Fz)
    for k in rec:
        rec[k] = np.array(rec[k])
    rec["_meta"] = dict(N=N, z0=z0, G_true=G_true_E / (2 * (1 + nu)), yield_true=yield_true,
                        E_true=G_true_E, nu=nu, v_plate=v_plate, npress=npress,
                        dt_ctrl=dt_ctrl, device=device)
    return rec


def identify(rec, elastic_frac=0.06, plastic_frac=0.40):
    """Two-window convex recovery of (G, yield) from the power-balance record.

    The deviatoric power rhs = P_internal - P_vol decomposes as
        rhs = (elastic stored-energy rate)  +  (plastic dissipation)
    ELASTIC window (first few frames, dev strain < yield/2G): purely elastic, rhs = G c1.
    PLASTIC window (steady flow): elastic deviatoric strain is SATURATED so its rate ~ 0, hence
        rhs ~ plastic dissipation = yield * INT||dev D|| dV = yield c2.
    No G c1 is subtracted in the plastic window: there the total Hencky dev strain in c1 is mostly
    PLASTIC (dissipated, not stored), so G c1 would be a spurious over-subtraction."""
    n = len(rec["Fz"])
    P_int = rec["P_plate"] + rec["P_grav"] - rec["dKE"]
    rhs = P_int - rec["P_vol"]
    ne = max(3, int(elastic_frac * n)); npl0 = int((1 - plastic_frac) * n)
    el = slice(0, ne); pl = slice(npl0, n)
    G_hat = float(rhs[el].sum() / max(rec["c1"][el].sum(), 1e-30))   # elastic: rhs = G c1
    yld_hat = float(rhs[pl].sum() / max(rec["c2"][pl].sum(), 1e-30)) # plastic dissipation = yield c2
    return dict(G_hat=G_hat, yield_hat=yld_hat,
                G_true=rec["_meta"]["G_true"], yield_true=rec["_meta"]["yield_true"],
                G_err=abs(G_hat / rec["_meta"]["G_true"] - 1),
                yield_err=abs(yld_hat / rec["_meta"]["yield_true"] - 1),
                elastic_frames=[0, ne], plastic_frames=[npl0, n])


def run(device="auto"):
    print("=== #75 force-based von-Mises (G, yield) identification from one squeeze ===", flush=True)
    rec = probe(device=device)
    meta = rec["_meta"]
    comp = 100 * (meta["z0"] - (meta["z0"] - rec["disp"][-1])) / meta["z0"]
    print(f"probe: N={meta['N']}, {meta['npress']} frames, v_plate={meta['v_plate']} m/s, "
          f"peak Fz={rec['Fz'].max():.2f} N", flush=True)
    res = identify(rec)
    print(f"\n  G:     hat={res['G_hat']:.3e}  true={res['G_true']:.3e}  ({100*res['G_err']:.1f}% err)", flush=True)
    print(f"  yield: hat={res['yield_hat']:.1f}  true={res['yield_true']:.1f}  ({100*res['yield_err']:.1f}% err)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in res.items()},
              open(OUT / "results.json", "w"), indent=2)
    # force-displacement curve + windows
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(rec["disp"] * 1000, rec["Fz"], "-", color="C0")
        e0, e1 = res["elastic_frames"]; p0, p1 = res["plastic_frames"]
        ax.axvspan(rec["disp"][e0] * 1000, rec["disp"][e1 - 1] * 1000, color="C2", alpha=0.2, label="elastic window (G)")
        ax.axvspan(rec["disp"][p0] * 1000, rec["disp"][p1 - 1] * 1000, color="C3", alpha=0.2, label="plastic window (yield)")
        ax.set_xlabel("plate displacement (mm)"); ax.set_ylabel("reaction force Fz (N)")
        ax.set_title(f"Force-based ID: G {100*res['G_err']:.0f}% err, yield {100*res['yield_err']:.0f}% err")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "force_displacement.png", dpi=130)
        print(f"figure -> {OUT/'force_displacement.png'}", flush=True)
    except Exception as e:
        print("plot skipped:", e, flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    args = parser.parse_args()
    run(device=args.device)
