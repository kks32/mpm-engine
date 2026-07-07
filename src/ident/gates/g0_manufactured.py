"""G0 manufactured-solution gate (MATH_REFERENCE.md Section 9).

Protocol implemented here:
  1. Trivial case: constant p, constant mu, simple analytic divergence-free
     v, one bump test function, both sides by dense quadrature.
  2. Sign-flip negative controls: flipping the sign of b or of D[w] must
     fail by order one.
  3. Full manufactured solution: analytic v(x, z, t), p(x, z, t), mu(I) in
     the Mode P dictionary span; theta recovery converges under quadrature
     refinement and the observed rate sets tolerances.
  4. Derivative cross-checks: dictionary phi derivatives, the pressure
     sensitivity operator, and test-function gradients against finite
     differences.
  5. The acceleration-identity probe on oracle data lives with the
     simulator tasks, not here.

Manufactured force: the fields do not satisfy momentum balance by
themselves, so the balance-completing body force

  f_man = rho a - rho g - div sigma,   sigma = -p Id + tau

is added to the load. div sigma is evaluated by 4th order central finite
differences of the analytic stress field, exact far below quadrature
tolerances. The assembly formulas are used verbatim from
ident/weakform/assembly.py.

Field design constraint: the regularized direction tensor 2 D / |gamma_dot|_eps
is nearly discontinuous wherever |gamma_dot| -> 0 (the same reason the real
pipeline masks non-flowing material). The manufactured velocity therefore
superposes a strictly positive shear base flow

  v_x += S0 z + 0.5 S1 z^2

with perturbations small enough that D_xz never crosses zero, keeping the
integrand smooth inside every patch and the realized I band about two
decades wide.

Mismatch metrics are signal relative: |LHS - RHS| / max(|LHS|, |RHS|).
Row scaling by inertia-gravity magnitude is a solve-time choice and is not
used as a match denominator here, because gravity integrates to zero against
compact divergence-free w and would understate errors by orders of magnitude.

All hand-coded analytic derivatives (grad v, acceleration) are self-checked
against finite differences before any case runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.conventions import (
    EPS_GAMMA_DEFAULT,
    SEED_DEFAULT,
    equivalent_shear_rate,
    git_rev,
    gravity_vector_inplane,
    inertial_number,
)
from ident.features.base import Dictionary
from ident.features.constant import ConstantDict
from ident.features.pouliquen_grid import PouliquenGridDict
from ident.gates.plotting import plot_convergence, plot_mu_curves
from ident.pressure.sensitivity import assemble_dA, theta_sensitivity
from ident.solve.ridge import ridge_solve
from ident.weakform.assembly import FrameData, assemble_system
from ident.weakform.closure import closure_diagnostic
from ident.weakform.test_functions import BumpTestFunction, patch_rows

RESULTS_SCHEMA_VERSION = "g0-1.1"

# the recovery dictionary: I0 atoms restricted to the band the manufactured
# fields actually excite (about I in [0.1, 0.8]). An atom whose I0 sits far
# below the band has f_g nearly constant there, collinear with the constant
# column; that conditioning story belongs to G1 stratification, not to the
# assembly-correctness story of G0.
RECOVERY_I0_GRID = np.logspace(-1.0, 0.0, 3)
THETA_TRUE_MU_S = 0.38
THETA_TRUE_RISE = 0.26
THETA_TRUE_ATOM = 1  # index into RECOVERY_I0_GRID: I0 = 10^-0.5


def recovery_dictionary() -> PouliquenGridDict:
    return PouliquenGridDict(I0_grid=RECOVERY_I0_GRID)


def theta_true_vector(dic: PouliquenGridDict) -> np.ndarray:
    theta = np.zeros(dic.K)
    theta[0] = THETA_TRUE_MU_S
    theta[1 + THETA_TRUE_ATOM] = THETA_TRUE_RISE
    return theta


# ---------------- manufactured fields ----------------

@dataclass
class ManufacturedConfig:
    """Dynamically consistent manufactured scales.

    The time-weak load integrates rho (v (x) v) : grad w, so its quadrature
    noise scales with rho v^2 while the stress signal scales with mu p. The
    fields keep rho v^2 / p of order one, as in a physical collapse; the I
    band is widened by the time envelope on the base shear and by the
    synthetic grain size, not by large velocities.
    """

    S0: float = 1.0            # base shear rate at z = 0, 1/s
    S1: float = 1.2            # base shear-rate gradient, 1/(s m)
    env_sh_amp: float = 0.75   # time envelope amplitude on the base shear
    omega_S: float = 2.0       # base-shear envelope frequency, 1/s
    A_tg: float = 0.25         # Taylor-Green amplitude, m/s
    k: float = 2.0 * np.pi     # TG wavenumber, 1/m
    V_sh: float = 0.02         # sinusoidal shear amplitude, m/s
    m_sh: float = 1.5 * np.pi
    phase_sh: float = 0.6
    U_osc: float = 0.3         # uniform oscillation, m/s
    omega_tg: float = 3.0      # 1/s
    omega_sh: float = 2.2
    omega_u: float = 4.0
    p0: float = 2000.0         # Pa
    p_amp: float = 0.45
    kp: float = 1.3 * np.pi
    mp: float = 1.7 * np.pi
    omega_p: float = 2.6
    rho: float = 1500.0        # bulk density kg/m^3
    rho_s: float = 2500.0      # grain density kg/m^3
    d: float = 0.12            # synthetic grain diameter m, sets the I scale
    eps_gamma: float = EPS_GAMMA_DEFAULT
    unsteady: bool = True      # trivial case sets False
    constant_p: bool = False


class ManufacturedFields:
    """Analytic v, grad v, a, p plus stress and balance force for theta_true."""

    def __init__(self, cfg: ManufacturedConfig, dictionary: Dictionary, theta_true: np.ndarray):
        self.cfg = cfg
        self.dictionary = dictionary
        self.theta_true = np.asarray(theta_true, dtype=float)

    # time envelopes
    def _f(self, t):
        return 1.0 + (0.5 * np.sin(self.cfg.omega_tg * t) if self.cfg.unsteady else 0.0)

    def _fdot(self, t):
        return 0.5 * self.cfg.omega_tg * np.cos(self.cfg.omega_tg * t) if self.cfg.unsteady else 0.0

    def _h(self, t):
        return 1.0 + (0.3 * np.cos(self.cfg.omega_sh * t) if self.cfg.unsteady else 0.0)

    def _hdot(self, t):
        return -0.3 * self.cfg.omega_sh * np.sin(self.cfg.omega_sh * t) if self.cfg.unsteady else 0.0

    def _e(self, t):
        """Base-shear envelope; widens the realized I band across time windows."""
        c = self.cfg
        return 1.0 + (c.env_sh_amp * np.sin(c.omega_S * t) if c.unsteady else 0.0)

    def _edot(self, t):
        c = self.cfg
        return c.env_sh_amp * c.omega_S * np.cos(c.omega_S * t) if c.unsteady else 0.0

    def _base_profile(self, z):
        c = self.cfg
        return c.S0 * z + 0.5 * c.S1 * z * z

    def v(self, x, z, t):
        c = self.cfg
        f, h, e = self._f(t), self._h(t), self._e(t)
        vx = (
            self._base_profile(z) * e
            - c.A_tg * np.sin(c.k * x) * np.cos(c.k * z) * f
            + c.V_sh * np.sin(c.m_sh * z + c.phase_sh) * h
        )
        vz = c.A_tg * np.cos(c.k * x) * np.sin(c.k * z) * f
        if c.unsteady:
            vx = vx + c.U_osc * np.cos(c.omega_u * t)
            vz = vz + 0.5 * c.U_osc * np.sin(c.omega_u * t)
        return np.stack([vx, vz], axis=-1)

    def grad_v(self, x, z, t):
        """L with L_ij = d v_i / d x_j, shape (..., 2, 2)."""
        c = self.cfg
        f, h, e = self._f(t), self._h(t), self._e(t)
        sx, cx = np.sin(c.k * x), np.cos(c.k * x)
        sz, cz = np.sin(c.k * z), np.cos(c.k * z)
        ones = np.ones(np.broadcast_shapes(np.shape(x), np.shape(z)))
        dvx_dx = -c.A_tg * c.k * cx * cz * f * ones
        dvx_dz = (
            (c.S0 + c.S1 * z) * e * ones
            + c.A_tg * c.k * sx * sz * f
            + c.V_sh * c.m_sh * np.cos(c.m_sh * z + c.phase_sh) * h
        )
        dvz_dx = -c.A_tg * c.k * sx * sz * f * ones
        dvz_dz = c.A_tg * c.k * cx * cz * f * ones
        return np.stack(
            [
                np.stack([dvx_dx, dvx_dz], axis=-1),
                np.stack([dvz_dx, dvz_dz], axis=-1),
            ],
            axis=-2,
        )

    def a(self, x, z, t):
        """Material acceleration a = dv/dt + L v (L contracts as L @ v)."""
        c = self.cfg
        fdot, hdot, edot = self._fdot(t), self._hdot(t), self._edot(t)
        shape = np.broadcast_shapes(np.shape(x), np.shape(z))
        at_x = (
            self._base_profile(z) * edot
            - c.A_tg * np.sin(c.k * x) * np.cos(c.k * z) * fdot
            + c.V_sh * np.sin(c.m_sh * z + c.phase_sh) * hdot
        ) * np.ones(shape)
        at_z = c.A_tg * np.cos(c.k * x) * np.sin(c.k * z) * fdot * np.ones(shape)
        if c.unsteady:
            at_x = at_x - c.U_osc * c.omega_u * np.sin(c.omega_u * t)
            at_z = at_z + 0.5 * c.U_osc * c.omega_u * np.cos(c.omega_u * t)
        at = np.stack([at_x, at_z], axis=-1)
        L = self.grad_v(x, z, t)
        vv = self.v(x, z, t)
        adv = np.einsum("...ij,...j->...i", L, vv)
        return at + adv

    def p(self, x, z, t):
        c = self.cfg
        shape = np.broadcast_shapes(np.shape(x), np.shape(z))
        if c.constant_p:
            return np.full(shape, c.p0)
        envelope = 0.6 + (0.4 * np.cos(c.omega_p * t) if c.unsteady else 0.4)
        return c.p0 * (
            1.0 + c.p_amp * np.sin(c.kp * x + 0.3) * np.cos(c.mp * z - 0.4) * envelope
        ) * np.ones(shape)

    def D(self, x, z, t):
        L = self.grad_v(x, z, t)
        return 0.5 * (L + np.swapaxes(L, -1, -2))

    def I_field(self, x, z, t):
        gd = equivalent_shear_rate(self.D(x, z, t), self.cfg.eps_gamma)
        return inertial_number(gd, self.p(x, z, t), self.cfg.d, self.cfg.rho_s)

    def mu_field(self, x, z, t):
        I = self.I_field(x, z, t)
        Phi = self.dictionary.phi(np.ravel(I))
        return (Phi @ self.theta_true).reshape(np.shape(I))

    def tau(self, x, z, t):
        D = self.D(x, z, t)
        gd = equivalent_shear_rate(D, self.cfg.eps_gamma)
        mu = self.mu_field(x, z, t)
        p = self.p(x, z, t)
        return (mu * p / gd)[..., None, None] * (2.0 * D)

    def sigma(self, x, z, t):
        tau = self.tau(x, z, t)
        p = self.p(x, z, t)
        return tau - p[..., None, None] * np.eye(2)

    def div_sigma(self, x, z, t, h_fd: float = 1.0e-4):
        """(div sigma)_i = sum_j d_j sigma_ij by 4th order central differences."""
        def s(xx, zz):
            return self.sigma(xx, zz, t)

        c1, c2 = 8.0, 1.0
        denom = 12.0 * h_fd
        dsig_dx = (
            -c2 * s(x + 2 * h_fd, z) + c1 * s(x + h_fd, z)
            - c1 * s(x - h_fd, z) + c2 * s(x - 2 * h_fd, z)
        ) / denom
        dsig_dz = (
            -c2 * s(x, z + 2 * h_fd) + c1 * s(x, z + h_fd)
            - c1 * s(x, z - h_fd) + c2 * s(x, z - 2 * h_fd)
        ) / denom
        return dsig_dx[..., :, 0] + dsig_dz[..., :, 1]

    def f_man(self, x, z, t):
        """Balance-completing body force f = rho a - rho g - div sigma."""
        g = gravity_vector_inplane()
        return (
            self.cfg.rho * (self.a(x, z, t) - g[None, :])
            - self.div_sigma(x, z, t)
        )

    def gamma_dot_bounds(self, n: int = 40000, rng: np.random.Generator | None = None):
        """Empirical |gamma_dot| range over the space-time box, for the report."""
        rng = rng or np.random.default_rng(SEED_DEFAULT)
        x = rng.uniform(0.0, 1.0, n)
        z = rng.uniform(0.0, 1.0, n)
        t = rng.uniform(0.0, 0.5, n)
        gd = equivalent_shear_rate(self.D(x, z, t), self.cfg.eps_gamma)
        return float(gd.min()), float(gd.max())

    def self_check(self, rng: np.random.Generator, n: int = 64) -> dict[str, float]:
        """FD verification of the hand-coded grad_v and acceleration."""
        x = rng.uniform(0.2, 0.8, n)
        z = rng.uniform(0.2, 0.8, n)
        t = rng.uniform(0.05, 0.45, n) if self.cfg.unsteady else np.zeros(n)
        h = 1.0e-6

        L = self.grad_v(x, z, t)
        L_fd = np.empty_like(L)
        L_fd[..., :, 0] = (self.v(x + h, z, t) - self.v(x - h, z, t)) / (2 * h)
        L_fd[..., :, 1] = (self.v(x, z + h, t) - self.v(x, z - h, t)) / (2 * h)
        err_L = float(np.max(np.abs(L - L_fd)) / np.max(np.abs(L)))

        dv_dt_fd = (self.v(x, z, t + h) - self.v(x, z, t - h)) / (2 * h)
        adv = np.einsum("...ij,...j->...i", L, self.v(x, z, t))
        a_fd = dv_dt_fd + adv
        a_an = self.a(x, z, t)
        err_a = float(np.max(np.abs(a_an - a_fd)) / max(np.max(np.abs(a_an)), 1e-12))

        div_v = L[..., 0, 0] + L[..., 1, 1]
        err_div = float(np.max(np.abs(div_v)) / np.max(np.abs(L)))
        return {"grad_v_fd": err_L, "accel_fd": err_a, "div_v": err_div}


# ---------------- quadrature data construction ----------------

def build_frames(
    fields: ManufacturedFields,
    h_q: float,
    dt_q: float,
    t_end: float,
    domain: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    with_force: bool = True,
) -> list[FrameData]:
    x0, x1, z0, z1 = domain
    xs = np.arange(x0 + 0.5 * h_q, x1, h_q)
    zs = np.arange(z0 + 0.5 * h_q, z1, h_q)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    xq, zq = X.ravel(), Z.ravel()
    vol = np.full(xq.shape, h_q * h_q)  # per unit out-of-plane width
    rho = np.full(xq.shape, fields.cfg.rho)

    frames = []
    for t in np.arange(0.0, t_end + 0.5 * dt_q, dt_q):
        D = fields.D(xq, zq, t)
        gd = equivalent_shear_rate(D, fields.cfg.eps_gamma)
        p = fields.p(xq, zq, t)
        I = inertial_number(gd, p, fields.cfg.d, fields.cfg.rho_s)
        frames.append(
            FrameData(
                t=float(t),
                x=np.stack([xq, zq], axis=-1),
                v=fields.v(xq, zq, t),
                D=D,
                a=fields.a(xq, zq, t),
                p=p,
                I=I,
                vol=vol,
                rho=rho,
                extra_force=fields.f_man(xq, zq, t) if with_force else None,
            )
        )
    return frames


def default_rows(t_end: float) -> list[BumpTestFunction]:
    rows: list[BumpTestFunction] = []
    for xc in (0.3, 0.5, 0.7):
        for zc in (0.3, 0.5, 0.7):
            for tc in (t_end / 3.0, 2.0 * t_end / 3.0):
                rows.extend(
                    patch_rows(xc, zc, tc, r_x=0.22, r_z=0.22, r_t=t_end / 4.0)
                )
    return rows


def _signal_relative_mismatch(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)


def _tw_gross_scale(frames: list[FrameData], tf: BumpTestFunction) -> float:
    """Gross magnitude of the time-weak load integrand for one row.

    The M2 identity replaces INT rho a . w by minus INT rho [v . dw/dt +
    (v (x) v) : grad w]. With a manufactured (non-equilibrium) field the net
    signal is much smaller than this gross term, so quadrature error of the
    M2 side must be judged against the gross scale, not the net signal. In
    oracle data the load itself is balance-scale and this distinction
    disappears.
    """
    times = np.array([f.t for f in frames])
    sel = np.where((times > tf.t_window[0]) & (times < tf.t_window[1]))[0]
    if sel.size < 2:
        return 0.0
    lo, hi = max(sel[0] - 1, 0), min(sel[-1] + 1, len(frames) - 1)
    idx = np.arange(lo, hi + 1)
    tw = np.zeros(len(idx))
    dtt = np.diff(times[idx])
    tw[:-1] += 0.5 * dtt
    tw[1:] += 0.5 * dtt
    gross = 0.0
    for w_t, fi in zip(tw, idx):
        fr = frames[fi]
        xs, zs = fr.x[:, 0], fr.x[:, 1]
        inside = tf.in_support(xs, zs, fr.t)
        if not np.any(inside):
            continue
        x_, z_ = xs[inside], zs[inside]
        gw = tf.grad_w(x_, z_, fr.t)
        dwdt = tf.dw_dt(x_, z_, fr.t)
        v_ = fr.v[inside]
        vol_, rho_ = fr.vol[inside], fr.rho[inside]
        vmag = np.linalg.norm(v_, axis=1)
        gross += w_t * float(
            np.sum(
                vol_ * rho_ * (
                    vmag * np.linalg.norm(dwdt, axis=1)
                    + vmag**2 * np.linalg.norm(gw.reshape(len(x_), 4), axis=1)
                )
            )
        )
    return gross


# ---------------- gate cases ----------------

def trivial_case(h_q: float = 1.0 / 32, dt_q: float = 1.0 / 64, levels: int = 3) -> dict:
    """Constant p, constant mu, steady divergence-free v, one bump.

    Both sides by dense quadrature at successive joint (h, dt) halvings.
    Signal-relative mismatch must shrink at the quadrature rate; the observed
    rate sets the tolerance (fit across levels, slope about 2).
    """
    dic = ConstantDict()
    theta = np.array([0.42])
    cfg = ManufacturedConfig(unsteady=False, constant_p=True)
    fields = ManufacturedFields(cfg, dic, theta)
    t_end = 0.4
    row = [BumpTestFunction(0.5, 0.5, 0.2, 0.28, 0.28, 0.16)]

    out: dict = {"levels": []}
    for lev in range(levels):
        h, dt = h_q / 2**lev, dt_q / 2**lev
        frames = build_frames(fields, h, dt, t_end)
        sys = assemble_system(frames, row, dic, cfg.eps_gamma)
        lhs = float((sys.A @ theta)[0])
        rhs = float(sys.b_acc[0])
        rhs_tw = float(sys.b_tw[0])
        gross_tw = _tw_gross_scale(frames, row[0])
        out["levels"].append(
            {
                "h_q": h,
                "dt_q": dt,
                "lhs": lhs,
                "rhs_acc": rhs,
                "rhs_timeweak": rhs_tw,
                "rel_mismatch_acc": _signal_relative_mismatch(lhs, rhs),
                "rel_mismatch_timeweak_gross": abs(lhs - rhs_tw) / max(gross_tw, 1e-300),
            }
        )
    hs = np.array([lv["h_q"] for lv in out["levels"]])
    e_acc = np.array([lv["rel_mismatch_acc"] for lv in out["levels"]])
    e_tw = np.array([lv["rel_mismatch_timeweak_gross"] for lv in out["levels"]])
    # least squares slope of log error vs log h
    rate_acc = float(np.polyfit(np.log(hs), np.log(np.maximum(e_acc, 1e-300)), 1)[0])
    rate_tw = float(np.polyfit(np.log(hs), np.log(np.maximum(e_tw, 1e-300)), 1)[0])
    out["observed_rate_acc"] = rate_acc
    out["observed_rate_timeweak"] = rate_tw
    out["passed"] = bool(
        e_acc[-1] < 2.0e-3
        and e_tw[-1] < 5.0e-3
        and rate_acc > 1.5
        and rate_tw > 1.5
    )
    return out


def negative_controls(h_q: float = 1.0 / 64, dt_q: float = 1.0 / 128) -> dict:
    """Sign flips must produce order-one mismatch, not pass within tolerance."""
    dic = ConstantDict()
    theta = np.array([0.42])
    cfg = ManufacturedConfig(unsteady=False, constant_p=True)
    fields = ManufacturedFields(cfg, dic, theta)
    t_end = 0.4
    row = [BumpTestFunction(0.5, 0.5, 0.2, 0.28, 0.28, 0.16)]
    frames = build_frames(fields, h_q, dt_q, t_end)
    sys = assemble_system(frames, row, dic, cfg.eps_gamma)
    lhs = float((sys.A @ theta)[0])
    rhs = float(sys.b_acc[0])

    baseline = _signal_relative_mismatch(lhs, rhs)
    flip_Dw = _signal_relative_mismatch(-lhs, rhs)  # D[w] sign flip negates A
    flip_b = _signal_relative_mismatch(lhs, -rhs)
    return {
        "baseline_rel_mismatch": baseline,
        "flip_Dw_rel_mismatch": flip_Dw,
        "flip_b_rel_mismatch": flip_b,
        # order-one failure, far separated from the quadrature baseline
        "passed": bool(
            flip_Dw > 0.5 and flip_b > 0.5 and flip_Dw > 25.0 * baseline
        ),
    }


SPAN_COLUMNS = [0, 1 + THETA_TRUE_ATOM]  # constant column plus the true atom


def full_case(refinements: list[tuple[float, float]] | None = None) -> dict:
    """Time-dependent manufactured solution, recovery, convergence.

    Two solves per refinement from ONE assembly:
      span (K = 2): columns [constant, f at the true I0]. The truth is in
        this span and the system is well conditioned, so theta must converge
        to theta_true at the quadrature rate. This is the assembly gate.
      mode_p (K = 4): the in-band Mode P grid. Sigmoid atoms are mutually
        near-collinear by construction, so this solve is reported with its
        condition number as the conditioning baseline that G1 stratification
        and constraints address; it is not an assembly assertion.
    """
    dic = recovery_dictionary()
    theta_true = theta_true_vector(dic)
    theta_true_span = theta_true[SPAN_COLUMNS]
    cfg = ManufacturedConfig()
    fields = ManufacturedFields(cfg, dic, theta_true)
    t_end = 0.5
    rows = default_rows(t_end)

    if refinements is None:
        refinements = [
            (1.0 / 24, 1.0 / 48),
            (1.0 / 48, 1.0 / 96),
            (1.0 / 96, 1.0 / 192),
            (1.0 / 192, 1.0 / 384),
        ]

    results = []
    last = None
    for h_q, dt_q in refinements:
        frames = build_frames(fields, h_q, dt_q, t_end)
        sys = assemble_system(frames, rows, dic, cfg.eps_gamma).scaled()
        A_span = sys.A[:, SPAN_COLUMNS]
        res_span = ridge_solve(A_span, sys.b_acc, lam=1.0e-12)
        res_span_tw = ridge_solve(A_span, sys.b_tw, lam=1.0e-12)
        res_p = ridge_solve(sys.A, sys.b_acc, lam=1.0e-10)
        I_all = np.concatenate([f.I for f in frames])
        identity_res = float(
            np.linalg.norm(sys.A @ theta_true - sys.b_acc) / np.linalg.norm(sys.b_acc)
        )
        entry = {
            "h_q": h_q,
            "dt_q": dt_q,
            "theta_hat_span": res_span.theta.tolist(),
            "theta_err_rel": float(
                np.linalg.norm(res_span.theta - theta_true_span)
                / np.linalg.norm(theta_true_span)
            ),
            "theta_err_rel_timeweak": float(
                np.linalg.norm(res_span_tw.theta - theta_true_span)
                / np.linalg.norm(theta_true_span)
            ),
            "cond_AtA_span": res_span.cond_AtA,
            "theta_hat_mode_p": res_p.theta.tolist(),
            "cond_AtA_mode_p": res_p.cond_AtA,
            "effective_rank_mode_p": res_p.effective_rank,
            "identity_residual_rel": identity_res,
            "I_min": float(I_all.min()),
            "I_max": float(I_all.max()),
            "dual_form_b_rel_diff": float(
                np.linalg.norm(sys.b_acc - sys.b_tw) / np.linalg.norm(sys.b_acc)
            ),
        }
        results.append(entry)
        last = (sys, res_span, frames)

    errs = np.array([r["theta_err_rel"] for r in results])
    id_res = np.array([r["identity_residual_rel"] for r in results])
    hs = np.array([r["h_q"] for r in results])
    rates = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])
    # least-squares rate over all levels; consecutive-pair rates wobble when
    # spatial, temporal, and rough-integrand error components mix
    rate_ls = float(np.polyfit(np.log(hs), np.log(np.maximum(errs, 1e-300)), 1)[0])

    # negative control on the full solve: flipped A must wreck recovery
    sys_last, res_span_last, frames_last = last
    res_flip = ridge_solve(-sys_last.A[:, SPAN_COLUMNS], sys_last.b_acc, lam=1.0e-12)
    flip_err = float(
        np.linalg.norm(res_flip.theta - theta_true_span)
        / np.linalg.norm(theta_true_span)
    )

    errs_tw = np.array([r["theta_err_rel_timeweak"] for r in results])
    return {
        "theta_true": theta_true.tolist(),
        "theta_true_span": theta_true_span.tolist(),
        "I0_grid": RECOVERY_I0_GRID.tolist(),
        "span_columns": SPAN_COLUMNS,
        "refinements": results,
        "observed_rates": rates.tolist(),
        "observed_rate_ls": rate_ls,
        "negative_control_flip_A_theta_err": flip_err,
        # the time-weak solve carries the rho v^2 gross-term quadrature noise
        # of a manufactured non-equilibrium field, so its theta tolerance is
        # looser; on oracle data the load is balance-scale and the two forms
        # are compared as a refusal component instead.
        # convergence under refinement is the gate: strictly decreasing theta
        # and identity errors, a least-squares rate consistent with the
        # quadrature order, a finest error consistent with that rate, and an
        # order-one failure under the sign flip
        "passed": bool(
            errs[-1] < 2.0e-2
            and np.all(np.diff(errs) < 0.0)
            and np.all(np.diff(id_res) < 0.0)
            and rate_ls > 1.2
            and flip_err > 0.5
            and errs_tw[-1] < 0.2
        ),
        "_last": last,
    }


def fd_checks(rng: np.random.Generator) -> dict:
    """Dictionary, test-function, and sensitivity-operator FD cross-checks."""
    out: dict = {}

    I = 10.0 ** rng.uniform(-3.5, -0.1, 200)
    h = 1.0e-7
    for dic, name in [
        (ConstantDict(), "constant"),
        (PouliquenGridDict(), "pouliquen_grid"),
    ]:
        fd = (dic.phi(I * (1 + h)) - dic.phi(I * (1 - h))) / (2 * h * I[:, None])
        an = dic.dphi_dI(I)
        denom = max(float(np.max(np.abs(an))), 1.0)
        out[f"dphi_dI_{name}"] = float(np.max(np.abs(an - fd)) / denom)
        chain = dic.dphi_dlogI(I) - dic.dphi_dI(I) * I[:, None] * np.log(10.0)
        out[f"dphi_dlogI_chain_{name}"] = float(np.max(np.abs(chain)))

    tf = BumpTestFunction(0.4, 0.55, 0.2, 0.3, 0.25, 0.15)
    x = rng.uniform(0.15, 0.65, 300)
    z = rng.uniform(0.35, 0.75, 300)
    t = rng.uniform(0.08, 0.32, 300)
    hh = 1.0e-6
    g_an = tf.grad_w(x, z, t)
    g_fd = np.empty_like(g_an)
    g_fd[..., :, 0] = (tf.w(x + hh, z, t) - tf.w(x - hh, z, t)) / (2 * hh)
    g_fd[..., :, 1] = (tf.w(x, z + hh, t) - tf.w(x, z - hh, t)) / (2 * hh)
    out["test_function_grad_fd"] = float(
        np.max(np.abs(g_an - g_fd)) / np.max(np.abs(g_an))
    )
    dwdt_fd = (tf.w(x, z, t + hh) - tf.w(x, z, t - hh)) / (2 * hh)
    dwdt_an = tf.dw_dt(x, z, t)
    out["test_function_dwdt_fd"] = float(
        np.max(np.abs(dwdt_an - dwdt_fd)) / np.max(np.abs(dwdt_an))
    )
    out["div_w_max"] = float(np.max(np.abs(tf.div_w(x, z, t))))

    # sensitivity operator against finite differences, relative pressure bias.
    # Checked on the well-conditioned span columns; conditioning of the wide
    # Mode P grid would otherwise amplify the FD curvature error and test the
    # solver instead of the operator.
    dic = recovery_dictionary()
    theta_true = theta_true_vector(dic)
    cfg = ManufacturedConfig()
    fields = ManufacturedFields(cfg, dic, theta_true)
    t_end = 0.3
    rows = default_rows(t_end)[:18]
    h_q, dt_q = 1.0 / 32, 1.0 / 64
    frames = build_frames(fields, h_q, dt_q, t_end)
    sys = assemble_system(frames, rows, dic, cfg.eps_gamma)
    lam = 1.0e-8
    A_span = sys.A[:, SPAN_COLUMNS]
    res = ridge_solve(A_span, sys.b_acc, lam=lam)
    dp_frames = [f.p.copy() for f in frames]  # delta_p = p, relative bias
    dA = assemble_dA(frames, rows, dic, dp_frames, cfg.eps_gamma)[:, SPAN_COLUMNS]
    dtheta = theta_sensitivity(A_span, sys.b_acc, res.theta, dA, lam=lam)

    eps = 1.0e-5

    def theta_at(scale: float) -> np.ndarray:
        fr2 = []
        for f in frames:
            p2 = f.p * scale
            gd = equivalent_shear_rate(f.D, cfg.eps_gamma)
            I2 = inertial_number(gd, p2, cfg.d, cfg.rho_s)
            fr2.append(
                FrameData(
                    t=f.t, x=f.x, v=f.v, D=f.D, a=f.a, p=p2, I=I2,
                    vol=f.vol, rho=f.rho, extra_force=f.extra_force,
                )
            )
        s2 = assemble_system(fr2, rows, dic, cfg.eps_gamma)
        return ridge_solve(s2.A[:, SPAN_COLUMNS], sys.b_acc, lam=lam).theta

    fd_dtheta = (theta_at(1.0 + eps) - theta_at(1.0 - eps)) / (2 * eps)
    out["sensitivity_fd"] = float(
        np.linalg.norm(dtheta - fd_dtheta) / max(np.linalg.norm(fd_dtheta), 1e-14)
    )
    out["passed"] = bool(
        out["test_function_grad_fd"] < 1e-7
        and out["test_function_dwdt_fd"] < 1e-7
        and out["div_w_max"] < 1e-12
        and out["sensitivity_fd"] < 1e-4
        and all(out[k] < 1e-5 for k in out if k.startswith("dphi_dI_"))
    )
    return out


# ---------------- gate driver ----------------

def run_gate(out_dir: str | Path = "out/g0", quick: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED_DEFAULT)

    cfg = ManufacturedConfig()
    dic_probe = recovery_dictionary()
    theta_probe = theta_true_vector(dic_probe)
    probe_fields = ManufacturedFields(cfg, dic_probe, theta_probe)
    self_chk = probe_fields.self_check(rng)
    gd_lo, gd_hi = probe_fields.gamma_dot_bounds()

    trivial = trivial_case() if not quick else trivial_case(1.0 / 24, 1.0 / 48)
    negative = negative_controls()
    refinements = (
        [(1.0 / 16, 1.0 / 32), (1.0 / 32, 1.0 / 64)]
        if quick
        else None
    )
    full = full_case(refinements)
    sys_last, res_last, frames_last = full.pop("_last")
    checks = fd_checks(rng)

    # closure diagnostic on the manufactured data, full sigma available
    sigma_frames = [
        probe_fields.sigma(f.x[:, 0], f.x[:, 1], f.t) for f in frames_last
    ]
    patches = [
        (0.4, 0.45, 0.25, 0.2, 0.2, 0.12),
        (0.6, 0.55, 0.25, 0.2, 0.2, 0.12),
    ]
    closure = closure_diagnostic(frames_last, sigma_frames, patches)

    # figures
    I_grid = np.logspace(-2.5, 0.0, 200)
    mu_true = dic_probe.phi(I_grid) @ np.asarray(full["theta_true"])
    mu_hat = dic_probe.phi(I_grid)[:, SPAN_COLUMNS] @ res_last.theta
    I_lo = min(r["I_min"] for r in full["refinements"])
    I_hi = max(r["I_max"] for r in full["refinements"])
    fig1 = plot_mu_curves(
        I_grid,
        {"truth": mu_true, "true_p": mu_hat},
        out_dir / "g0_mu_recovery.png",
        observed_I=(max(I_lo, I_grid[0]), min(I_hi, I_grid[-1])),
        title="G0 manufactured solution, Mode P recovery",
    )
    hs = np.array([r["h_q"] for r in full["refinements"]])
    fig2 = plot_convergence(
        hs,
        {
            "theta error": np.array([r["theta_err_rel"] for r in full["refinements"]]),
            "identity residual": np.array(
                [r["identity_residual_rel"] for r in full["refinements"]]
            ),
        },
        out_dir / "g0_convergence.png",
        title="G0 quadrature refinement",
    )

    config = {
        "manufactured": {k: getattr(cfg, k) for k in vars(cfg)},
        "quick": quick,
    }
    last_ref = full["refinements"][-1]
    results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "git_rev": git_rev(),
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=float).encode()
        ).hexdigest()[:16],
        "random_seed": SEED_DEFAULT,
        "mode": "G0",
        "pressure_source": "manufactured_true",
        "dictionary_mode": "P",
        "regularization_lambda": 1.0e-10,
        "row_count_before_gating": int(sys_last.A.shape[0]),
        "row_count_after_gating": int(sys_last.A.shape[0]),
        "condition_number": last_ref["cond_AtA_mode_p"],
        "effective_rank": last_ref["effective_rank_mode_p"],
        "observed_I_min": last_ref["I_min"],
        "observed_I_max": last_ref["I_max"],
        "theta_hat": last_ref["theta_hat_span"],
        "theta_hat_mode_p": last_ref["theta_hat_mode_p"],
        "posterior_summary": {
            "sigma2": res_last.sigma2,
            "theta_std": np.sqrt(np.diag(res_last.Sigma_theta)).tolist(),
        },
        "closure_error_summary": {
            "worst_full": closure.worst_full,
            "worst_admissible": closure.worst_admissible,
        },
        "paths_to_figures": [str(fig1), str(fig2)],
        "field_self_check": self_chk,
        "gamma_dot_bounds": [gd_lo, gd_hi],
        "trivial_case": trivial,
        "negative_controls": negative,
        "full_case": full,
        "fd_checks": checks,
        "passed": bool(
            trivial["passed"] and negative["passed"] and full["passed"] and checks["passed"]
        ),
    }
    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    return results


if __name__ == "__main__":
    import sys as _sys

    quick = "--quick" in _sys.argv
    res = run_gate(quick=quick)
    print(json.dumps(
        {k: res[k] for k in ("trivial_case", "negative_controls", "fd_checks")},
        indent=2,
    ))
    print("full_case rates:", res["full_case"]["observed_rates"])
    print("full_case errors:", [r["theta_err_rel"] for r in res["full_case"]["refinements"]])
    print("G0", "PASSED" if res["passed"] else "FAILED")
