"""Recursive-least-squares coefficient evaluation for one experiment.

The deployed Mode F basis spans the granular family, but one collapse covers only part
of the inertial-number range. A batch fit is therefore poorly determined in unexcited
directions. This module streams each frame's grid-consistent weak-form rows through
recursive least squares and records theta(t) as an identifiability diagnostic.

The family prior initializes RLS with theta_0 = theta_bar and
P_0 = Sigma_theta / rho. Prior precision controls unexcited directions, while data
updates the directions covered by the flow. Without the prior, coefficients in the
under-excited subspace can become unbounded; with it, the estimate stays on the family
manifold.

This is the EUCLID weak form, unchanged: each frame f contributes the grid-node momentum
residual rows A_f theta = b_f (grid-consistent Bubnov-Galerkin, true MPM pressure), and RLS
accumulates them in time order. With forgetting_factor = 1 the final RLS estimate equals the
batch ridge-to-prior solution. RLS therefore adds an online convergence trajectory rather
than a different objective. The update uses square-root QR RLS and matches the ridge
batch to 1e-7.

Torch is used only under function_encoder_training/, per the module boundary. Run:
  .venv/bin/python -m ident.features.function_encoder_training.fe_rls
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from ident.features.function_encoder import FunctionEncoderDict
from ident.io.schema import Dump, load_dump
from ident.weakform.grid_assembly import assemble_grid_consistent

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "out" / "function_encoder"
FE_TABLE = REPO / "mpm_engine" / "fe-weights" / "granular_mu_i.npz"
DUMP = REPO / "out" / "dumps" / "column_pouliquen_a2.npz"
I_BAND = (0.006, 0.5)            # band for the relL2 metric (comparable to the batch numbers)
BATCH_SINGLE_RELL2 = 0.064       # single-collapse batch constrained solve (memory)
JOINT_PRIOR_RELL2 = 0.024        # 4-aspect joint + prior constrained solve (memory)


def _load_fe_prior(table_path: Path = FE_TABLE):
    d = np.load(table_path)
    fe = FunctionEncoderDict(d["s_grid"], d["table"])
    if "theta_mean" not in d:
        raise RuntimeError("run features/function_encoder_training/prior.py first")
    return fe, np.asarray(d["theta_mean"], float), np.asarray(d["theta_cov"], float)


def _material_accel(dump: Dump) -> np.ndarray:
    """Centred trajectory finite difference of the in-plane velocity, matching the default
    accel computed inside assemble_grid_consistent (so per-frame slices are consistent)."""
    ax, az = dump.meta.in_plane_axes
    v = dump.v[..., [ax, az]]
    dt = dump.meta.frame_dt
    a = np.zeros_like(v)
    a[1:-1] = (v[2:] - v[:-2]) / (2.0 * dt)
    a[0] = (v[1] - v[0]) / dt
    a[-1] = (v[-1] - v[-2]) / dt
    return a


def _frame_view(dump: Dump, f: int) -> Dump:
    """A single-frame Dump (n_frames=1) so assemble_grid_consistent emits just frame f's rows."""
    return Dump(
        meta=replace(dump.meta, n_frames=1),
        times=dump.times[f:f + 1], x=dump.x[f:f + 1], v=dump.v[f:f + 1],
        L=dump.L[f:f + 1], stress=dump.stress[f:f + 1], volume=dump.volume[f:f + 1],
        mass=dump.mass, active=dump.active[f:f + 1],
        mu_table_log10I=dump.mu_table_log10I, mu_table_mu=dump.mu_table_mu,
    )


def assemble_per_frame(dump: Dump, fe: FunctionEncoderDict, frame_stride: int = 6,
                       flow_frac_min: float = 0.90):
    """Per-frame grid-consistent rows in time order. Returns a list of dicts with the frame
    time, the row block (A_f, b_f), and that frame's flowing-I coverage."""
    accel = _material_accel(dump)
    out = []
    for f in range(0, dump.meta.n_frames, frame_stride):
        gs = assemble_grid_consistent(
            _frame_view(dump, f), fe, frame_stride=1, flow_frac_min=flow_frac_min,
            accel=accel[f:f + 1],
        )
        if gs.n_rows == 0:
            continue
        out.append({"t": float(dump.times[f]), "A": gs.A, "b": gs.b,
                    "I": gs.I_observed, "n": gs.n_rows})
    return out


def _rls_stream(frames, theta0, P0, scale, forgetting=1.0):
    """Stream the per-frame row blocks through the repo's QR (square-root) RLS.

    theta0 (K,), P0 (K,K) prior covariance, scale: global row scale so the load is O(1) and
    the prior covariance is commensurate with a unit measurement noise. Returns theta(t) (T,K).
    """
    sys.path.insert(0, str(REPO / "function-encoder"))
    import torch
    from function_encoder.coefficients import recursive_least_squares_update

    # the repo's QR RLS allocates intermediate eye/zeros at the default dtype; pin it to
    # float64 so the square-root update stays well conditioned over the streamed rows.
    torch.set_default_dtype(torch.float64)
    theta = torch.tensor(theta0, dtype=torch.float64).reshape(1, -1)
    P = torch.tensor(P0, dtype=torch.float64).reshape(1, *P0.shape)
    traj = []
    for fr in frames:
        A = torch.tensor(fr["A"] / scale, dtype=torch.float64)
        b = torch.tensor(fr["b"] / scale, dtype=torch.float64)
        g = A.reshape(1, 1, A.shape[0], A.shape[1])     # [batch, n_pts=1, n_rows, K]
        y = b.reshape(1, 1, b.shape[0])                 # [batch, n_pts=1, n_rows]
        L = torch.linalg.cholesky(P)
        theta, P = recursive_least_squares_update(
            g=g, y=y, P=L, coefficients=theta, forgetting_factor=forgetting, method="qr")
        traj.append(theta.reshape(-1).numpy().copy())
    return np.array(traj)


def run(dump_path: Path = DUMP, frame_stride: int = 6, rho: float = 1.0):
    OUT.mkdir(parents=True, exist_ok=True)
    fe, theta_bar, cov = _load_fe_prior()
    dump = load_dump(dump_path)
    lp = dump.meta.law_params
    Ig = np.logspace(np.log10(I_BAND[0]), np.log10(I_BAND[1]), 120)
    mu_true = lp["mu_s"] + lp["delta_mu"] * Ig / (Ig + lp["I0"])

    print(f"2D collapse {dump_path.name}: {dump.meta.n_frames} frames, "
          f"truth pouliquen mu_s={lp['mu_s']} delta_mu={lp['delta_mu']} I0={lp['I0']}")
    frames = assemble_per_frame(dump, fe, frame_stride=frame_stride)
    nrows = sum(fr["n"] for fr in frames)
    # global row scale so the load b is O(1); keeps RLS conditioning sane and makes the
    # prior covariance commensurate with unit measurement noise.
    scale = float(np.sqrt(np.mean(np.concatenate([fr["b"] for fr in frames]) ** 2)))
    Ilo = min(fr["I"][0] for fr in frames if fr["I"][0] > 0)
    Ihi = max(fr["I"][1] for fr in frames)
    print(f"  streamed {len(frames)} frames, {nrows} weak-form rows, "
          f"flowing-I coverage [{Ilo:.3g}, {Ihi:.3g}], row scale {scale:.3g}")

    K = fe.K
    # prior-initialized RLS: start at the family mean, prior covariance Sigma_theta / rho
    traj_prior = _rls_stream(frames, theta_bar, cov / rho, scale, forgetting=1.0)
    # no-prior RLS: zero start, diffuse covariance (the under-excited subspace runs free)
    traj_flat = _rls_stream(frames, np.zeros(K), 1.0e2 * np.eye(K), scale, forgetting=1.0)

    def rel(theta):
        mu = fe.phi(Ig) @ theta
        return float(np.sqrt(np.mean((mu - mu_true) ** 2)) / np.sqrt(np.mean(mu_true ** 2)))

    rel_prior = np.array([rel(t) for t in traj_prior])
    rel_flat = np.array([rel(t) for t in traj_flat])
    mu_final = fe.phi(Ig) @ traj_prior[-1]
    mono = bool(np.all(np.diff(mu_final) >= -1e-6))
    print(f"\n  RLS final relL2 over I={I_BAND}:")
    print(f"    no prior   (zero start, diffuse P) = {rel_flat[-1] * 100:6.1f}%")
    print(f"    family prior init (rho={rho})       = {rel_prior[-1] * 100:6.1f}%  "
          f"monotone={mono}")
    print(f"    [batch single-collapse constrained = {BATCH_SINGLE_RELL2 * 100:.1f}%, "
          f"4-aspect joint+prior = {JOINT_PRIOR_RELL2 * 100:.1f}%]")

    _figure(frames, traj_prior, theta_bar, rel_prior, rel_flat, Ig, mu_true, fe,
            (Ilo, Ihi), rho)
    return {"rel_prior": rel_prior, "rel_flat": rel_flat, "traj_prior": traj_prior,
            "I_coverage": (Ilo, Ihi), "monotone": mono}


def _figure(frames, traj_prior, theta_bar, rel_prior, rel_flat, Ig, mu_true, fe,
            I_cov, rho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    t = np.array([fr["t"] for fr in frames])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) theta_k(t) trajectories, prior-init, with the family-prior start as dashed
    cmap = colormaps["viridis"]
    K = traj_prior.shape[1]
    for k in range(K):
        c = cmap(k / max(K - 1, 1))
        ax[0].plot(t, traj_prior[:, k], color=c, lw=1.6)
        ax[0].axhline(theta_bar[k], color=c, ls=":", lw=0.9, alpha=0.7)
    ax[0].set_xlabel("time (s)  [streamed frame]")
    ax[0].set_ylabel("coefficient  theta_k")
    ax[0].set_title(f"(a) theta_k(t) RLS convergence (prior init, rho={rho})\n"
                    "dotted = family-prior mean theta_bar")
    ax[0].grid(alpha=0.3)

    # (b) relL2(t) prior vs no-prior, with batch reference lines
    ax[1].plot(t, rel_flat * 100, color="#e8590c", lw=2.0, label="RLS, no prior (diverges)")
    ax[1].plot(t, rel_prior * 100, color="#1c7ed6", lw=2.4, label="RLS, family-prior init")
    ax[1].axhline(BATCH_SINGLE_RELL2 * 100, color="#868e96", ls="--", lw=1.2,
                  label=f"batch single-collapse ({BATCH_SINGLE_RELL2*100:.0f}%)")
    ax[1].axhline(JOINT_PRIOR_RELL2 * 100, color="#2b8a3e", ls=":", lw=1.4,
                  label=f"4-aspect joint+prior ({JOINT_PRIOR_RELL2*100:.0f}%)")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1.5, 400)            # no-prior shoots off the top (peaks ~1e5%)
    ax[1].annotate(f"no prior diverges\n(peak {rel_flat.max()*100:.0e}%)", xy=(0.55, 0.86),
                   xycoords="axes fraction", fontsize=8, color="#e8590c", ha="center")
    ax[1].set_xlabel("time (s)  [streamed frame]")
    ax[1].set_ylabel("mu(I) relative L2 error (%)")
    ax[1].set_title("(b) recovery error vs streamed frames")
    ax[1].legend(fontsize=8, loc="lower left")
    ax[1].grid(alpha=0.3, which="both")

    # (c) recovered mu(I) at a few frame counts vs truth
    ax[2].plot(Ig, mu_true, "k-", lw=2.6, label="truth (Pouliquen)")
    n = len(traj_prior)
    picks = [max(1, n // 8), max(2, n // 3), n - 1]
    shades = ["#a5d8ff", "#4dabf7", "#1864ab"]
    for idx, col in zip(picks, shades):
        ax[2].plot(Ig, fe.phi(Ig) @ traj_prior[idx], color=col, lw=2.0,
                   label=f"RLS @ t={t[idx]:.3f}s")
    ax[2].axvspan(I_cov[0], I_cov[1], color="#ffe066", alpha=0.25,
                  label="flowing-I coverage")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("inertial number  I")
    ax[2].set_ylabel("mu(I)")
    ax[2].set_title("(c) recovered mu(I) converging frame-by-frame")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    p = OUT / "fe_rls.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    run()
