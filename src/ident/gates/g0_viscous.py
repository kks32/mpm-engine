"""G0-style manufactured proof of generality: viscous and viscoplastic laws.

Demonstrates that the SAME linear-in-theta weak form identifies a Newtonian
viscosity and a Bingham (yield-stress + viscosity) rheology, exactly as it
identifies the granular mu(I). The constitutive stress is written as a sum of
known tensor bases linear in the unknowns:

  Newtonian:  tau = eta * (2D)                          -> theta = (eta,)
  Bingham:    tau = tau_y * (2D/|gd|_eps) + eta * (2D)  -> theta = (tau_y, eta)

KEY REUSE (no new quadrature code): the G0-verified assemble_system computes
A[j] = INT vol * p * (2D/|gd|_eps):D[w] * phi(I). With ConstantDict (phi=1):
  - p = 1        gives the YIELD basis   INT (2D/|gd|):D[w]   (coeff tau_y)
  - p = |gd|_eps gives the VISCOUS basis INT (2D):D[w]        (coeff eta)
so the two basis columns are reweightings of the same verified integrand. The
load b is the balance-completing manufactured force (independent of the basis),
identical to G0. Divergence-free bump test functions kill the pressure, so the
deviatoric identification is pressure-free.

Protocol per law: recovery convergence under quadrature refinement (the
assembly-correctness gate), and a sign-flip negative control that must fail by
order one. For Bingham we also report the 2x2 conditioning, which is finite only
because the manufactured flow spans a range of shear rates (the same excitation
requirement that separates tau_y from eta).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from common.conventions import EPS_GAMMA_DEFAULT, SEED_DEFAULT, equivalent_shear_rate
from ident.features.constant import ConstantDict
from ident.gates.g0_manufactured import (
    ManufacturedConfig,
    ManufacturedFields,
    build_frames,
    default_rows,
)
from ident.solve.ridge import ridge_solve
from ident.weakform.assembly import assemble_system


class ViscousFields(ManufacturedFields):
    """Manufactured fields with a Newtonian or Bingham deviatoric stress.

    tau = tau_y * (2D/|gd|_eps) + eta * (2D) = 2 (eta + tau_y/|gd|_eps) D.
    Reuses the base velocity / acceleration / pressure and the FD div-sigma
    body force; only the stress is overridden.
    """

    def __init__(self, cfg: ManufacturedConfig, tau_y: float, eta: float):
        super().__init__(cfg, ConstantDict(), np.array([0.0]))
        self.tau_y = float(tau_y)
        self.eta = float(eta)

    def tau(self, x, z, t):
        D = self.D(x, z, t)
        gd = equivalent_shear_rate(D, self.cfg.eps_gamma)
        eta_app = self.eta + self.tau_y / gd
        return (2.0 * eta_app)[..., None, None] * D


def _basis_frames(frames, kind):
    """Return frames with p set to select a basis column under assemble_system.

    kind='yield' -> p=1 -> column INT (2D/|gd|):D[w]   (coefficient tau_y)
    kind='visc'  -> p=|gd|_eps -> column INT (2D):D[w]  (coefficient eta)
    """
    out = []
    for f in frames:
        if kind == "yield":
            p = np.ones_like(f.p)
        elif kind == "visc":
            p = equivalent_shear_rate(f.D, EPS_GAMMA_DEFAULT)
        else:
            raise ValueError(kind)
        out.append(dataclasses.replace(f, p=p))
    return out


def _design(frames, rows):
    """A = [yield_col, visc_col], b = balance load (shared)."""
    dic = ConstantDict()
    sy = assemble_system(_basis_frames(frames, "yield"), rows, dic, EPS_GAMMA_DEFAULT)
    sv = assemble_system(_basis_frames(frames, "visc"), rows, dic, EPS_GAMMA_DEFAULT)
    A = np.column_stack([sy.A[:, 0], sv.A[:, 0]])
    return A, sy.b_acc


def _recover(A, b, columns):
    res = ridge_solve(A[:, columns], b, lam=1.0e-12)
    return res


def run_law(tau_y, eta, columns, label, refinements, t_end=0.4):
    """Convergence of (subset of) theta=(tau_y,eta) recovery under refinement."""
    cfg = ManufacturedConfig(unsteady=True)
    fields = ViscousFields(cfg, tau_y, eta)
    rows = default_rows(t_end)
    theta_true_full = np.array([tau_y, eta])
    theta_true = theta_true_full[columns]

    levels = []
    last = None
    for h_q, dt_q in refinements:
        frames = build_frames(fields, h_q, dt_q, t_end)
        A, b = _design(frames, rows)
        res = _recover(A, b, columns)
        err = float(np.linalg.norm(res.theta - theta_true) /
                    max(np.linalg.norm(theta_true), 1e-30))
        levels.append({"h_q": h_q, "dt_q": dt_q, "theta_hat": res.theta.tolist(),
                       "theta_err_rel": err, "cond_AtA": res.cond_AtA})
        last = (A, b)
    hs = np.array([lv["h_q"] for lv in levels])
    errs = np.array([lv["theta_err_rel"] for lv in levels])
    rate = float(np.polyfit(np.log(hs), np.log(np.maximum(errs, 1e-300)), 1)[0])

    # sign-flip negative control: flipped A must wreck recovery
    A, b = last
    res_flip = ridge_solve(-A[:, columns], b, lam=1.0e-12)
    flip_err = float(np.linalg.norm(res_flip.theta - theta_true) /
                     max(np.linalg.norm(theta_true), 1e-30))

    # Two acceptable regimes: (a) ALREADY EXACT -- a law that is linear and
    # smooth in the kinematics (e.g. Newtonian eta*2D) is recovered to the FD
    # div-sigma floor at every resolution, so the error is flat at ~1e-4 with no
    # quadrature slope; (b) QUADRATURE-LIMITED -- a law with the near-singular
    # 2D/|gd| yield term (Bingham) has a genuine convergence rate. Require
    # accuracy in both, and a real rate only when not already exact.
    exact = errs[-1] < 1.0e-3
    converging = bool(np.all(np.diff(errs) < 0.0) and rate > 1.2)
    return {
        "label": label, "theta_true": theta_true.tolist(), "columns": columns,
        "levels": levels, "observed_rate": rate,
        "final_theta_err_rel": float(errs[-1]),
        "final_cond_AtA": levels[-1]["cond_AtA"],
        "negative_control_flip_err": flip_err,
        "passed": bool(errs[-1] < 2.0e-2 and (exact or converging) and flip_err > 0.5),
    }


def run_gate(out_dir="out/g0_viscous", quick=False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    refinements = ([(1 / 16, 1 / 32), (1 / 32, 1 / 64)] if quick else
                   [(1 / 24, 1 / 48), (1 / 48, 1 / 96), (1 / 96, 1 / 192)])

    # shear-rate range the manufactured flow excites (separability of tau_y, eta)
    probe = ViscousFields(ManufacturedConfig(unsteady=True), 0.0, 1.0)
    gd_lo, gd_hi = probe.gamma_dot_bounds()

    newtonian = run_law(0.0, 60.0, columns=[1], label="Newtonian eta=60",
                        refinements=refinements)
    # also fit the Newtonian data with BOTH columns: tau_y must come out ~0
    cfg = ManufacturedConfig(unsteady=True); fields = ViscousFields(cfg, 0.0, 60.0)
    rows = default_rows(0.4)
    A, b = _design(build_frames(fields, refinements[-1][0], refinements[-1][1], 0.4), rows)
    res_both = ridge_solve(A, b, lam=1.0e-12)
    newtonian["fit_both_columns_theta"] = res_both.theta.tolist()  # expect ~[0, 60]

    bingham = run_law(120.0, 60.0, columns=[0, 1], label="Bingham tau_y=120 eta=60",
                      refinements=refinements)

    result = {
        "gamma_dot_range": [gd_lo, gd_hi],
        "newtonian": newtonian,
        "bingham": bingham,
        "passed": bool(newtonian["passed"] and bingham["passed"]),
    }

    # figure: recovered vs truth shear-stress response tau(gd) = tau_y + eta*gd
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    gd = np.linspace(max(gd_lo, 1e-3), gd_hi, 100)
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=140)
    ax.plot(gd, 120.0 + 60.0 * gd, "k-", lw=2.5, label="Bingham truth tau=120+60*gd")
    th = np.array(bingham["levels"][-1]["theta_hat"])
    ax.plot(gd, th[0] + th[1] * gd, "r--", lw=2,
            label="recovered tau_y=%.1f eta=%.1f" % (th[0], th[1]))
    ax.plot(gd, 60.0 * gd, "k:", lw=1.8, label="Newtonian truth tau=60*gd")
    thn = newtonian["levels"][-1]["theta_hat"][0]
    ax.plot(gd, thn * gd, "b-.", lw=1.6, label="recovered eta=%.1f" % thn)
    ax.set_xlabel("shear rate |gamma_dot| (1/s)"); ax.set_ylabel("shear stress tau (Pa)")
    ax.set_title("Manufactured G0: viscosity + yield stress recovered by the linear weak form")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "g0_viscous.png")
    result["figure"] = str(out_dir / "g0_viscous.png")

    json.dump(result, open(out_dir / "results.json", "w"), indent=2, default=float)
    return result


if __name__ == "__main__":
    import sys
    r = run_gate(quick="--quick" in sys.argv)
    print(json.dumps({k: v for k, v in r.items() if k != "figure"}, indent=2, default=float))
    print("G0-VISCOUS", "PASSED" if r["passed"] else "FAILED")
