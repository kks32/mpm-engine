"""Quasi-static implicit MPM: equilibrium solves for press/squeeze scenes.

Formulation per docs/implicit_plan.md (displacement-increment quasi-statics after
GeoWarp, MIT, reimplemented; see AUTHORS.md): the unknown is the grid displacement
increment u for one load step, the trial stress is Kirchhoff from Hencky strain, and
the residual per node is internal minus external force. Newton-Krylov, matrix-free:
the Jacobian action is a finite-difference directional derivative, so the solver
needs residual evaluations only and adds no autodiff requirement to the kernels.
Transfers are the engine's quadratic B-splines, in float64 (the FD directional
derivative is noise-limited in float32).

The warp kernel carries the O(27 n_p) residual scatter; activation masks, Dirichlet
conditions, and the GMRES loop stay in numpy/scipy. Gate history: the equilibrium
column (experiments/qs_prototype.py) passed against the analytic profile before this
port, and the port is tested against the same gate.
"""
from __future__ import annotations

import numpy as np
import warp as wp
from scipy.sparse.linalg import LinearOperator, gmres

from warpmpm.core.solver import GridConfig, _ensure_warp

f64 = wp.float64


@wp.kernel
def _qs_residual(
    x_p: wp.array(dtype=wp.vec3d),
    b_e: wp.array(dtype=wp.mat33d),       # elastic left Cauchy-Green per particle
    vol0: wp.array(dtype=f64),
    mass: wp.array(dtype=f64),
    u: wp.array(dtype=wp.vec3d),          # grid displacement increment, flat nodes
    gravity: wp.vec3d,
    lam: f64,
    mu: f64,
    sigma_y: f64,                          # von Mises yield; 0 disables plasticity
    inv_dx: f64,
    ngy: wp.int32,
    ngz: wp.int32,
    R: wp.array(dtype=wp.vec3d),          # out: residual per node (atomic)
):
    p = wp.tid()
    one = f64(1.0)
    half = f64(0.5)

    base_x = wp.int(x_p[p][0] * inv_dx - half)
    base_y = wp.int(x_p[p][1] * inv_dx - half)
    base_z = wp.int(x_p[p][2] * inv_dx - half)
    fx = x_p[p] * inv_dx - wp.vec3d(f64(base_x), f64(base_y), f64(base_z))

    # quadratic B-spline weights and gradients per axis (offset 0, 1, 2)
    wa = wp.vec3d(half * (f64(1.5) - fx[0]) * (f64(1.5) - fx[0]),
                  half * (f64(1.5) - fx[1]) * (f64(1.5) - fx[1]),
                  half * (f64(1.5) - fx[2]) * (f64(1.5) - fx[2]))
    wb = wp.vec3d(f64(0.75) - (fx[0] - one) * (fx[0] - one),
                  f64(0.75) - (fx[1] - one) * (fx[1] - one),
                  f64(0.75) - (fx[2] - one) * (fx[2] - one))
    wc = wp.vec3d(half * (fx[0] - half) * (fx[0] - half),
                  half * (fx[1] - half) * (fx[1] - half),
                  half * (fx[2] - half) * (fx[2] - half))
    da = wp.vec3d((fx[0] - f64(1.5)) * inv_dx, (fx[1] - f64(1.5)) * inv_dx,
                  (fx[2] - f64(1.5)) * inv_dx)
    db = wp.vec3d(f64(-2.0) * (fx[0] - one) * inv_dx, f64(-2.0) * (fx[1] - one) * inv_dx,
                  f64(-2.0) * (fx[2] - one) * inv_dx)
    dc = wp.vec3d((fx[0] - half) * inv_dx, (fx[1] - half) * inv_dx,
                  (fx[2] - half) * inv_dx)
    w = wp.matrix_from_rows(wa, wb, wc)      # w[offset][axis]
    dw = wp.matrix_from_rows(da, db, dc)

    # trial kinematics: grad(du) = sum_i u_i outer grad(w_i)
    du_grad = wp.mat33d(f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0),
                        f64(0.0), f64(0.0), f64(0.0))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                gw = wp.vec3d(dw[i, 0] * w[j, 1] * w[k, 2],
                              w[i, 0] * dw[j, 1] * w[k, 2],
                              w[i, 0] * w[j, 1] * dw[k, 2])
                idx = ((base_x + i) * ngy + (base_y + j)) * ngz + (base_z + k)
                du_grad += wp.outer(u[idx], gw)

    G = wp.identity(n=3, dtype=f64) + du_grad
    b = G * b_e[p] * wp.transpose(G)      # trial elastic left Cauchy-Green

    # Kirchhoff stress from Hencky strain, principal space; radial return for
    # von Mises (dev tau = 2 mu dev e, so the map scales the deviator in place).
    # The commit kernel repeats this map to store the returned elastic state.
    Q = wp.mat33d(f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0),
                  f64(0.0), f64(0.0), f64(0.0))
    lam2 = wp.vec3d(f64(0.0), f64(0.0), f64(0.0))
    wp.eig3(b, Q, lam2)
    e0 = half * wp.log(wp.max(lam2[0], f64(1e-12)))
    e1 = half * wp.log(wp.max(lam2[1], f64(1e-12)))
    e2 = half * wp.log(wp.max(lam2[2], f64(1e-12)))
    tr = e0 + e1 + e2
    t0 = lam * tr + f64(2.0) * mu * e0
    t1 = lam * tr + f64(2.0) * mu * e1
    t2 = lam * tr + f64(2.0) * mu * e2
    if sigma_y > f64(0.0):
        mean = (t0 + t1 + t2) / f64(3.0)
        s0 = t0 - mean
        s1 = t1 - mean
        s2 = t2 - mean
        vm = wp.sqrt(f64(1.5) * (s0 * s0 + s1 * s1 + s2 * s2))
        if vm > sigma_y:
            sc = sigma_y / vm
            t0 = mean + sc * s0
            t1 = mean + sc * s1
            t2 = mean + sc * s2
    tau_diag = wp.mat33d(t0, f64(0.0), f64(0.0),
                         f64(0.0), t1, f64(0.0),
                         f64(0.0), f64(0.0), t2)
    tau = Q * tau_diag * wp.transpose(Q)

    # scatter internal minus external force
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[i, 0] * w[j, 1] * w[k, 2]
                gw = wp.vec3d(dw[i, 0] * w[j, 1] * w[k, 2],
                              w[i, 0] * dw[j, 1] * w[k, 2],
                              w[i, 0] * w[j, 1] * dw[k, 2])
                idx = ((base_x + i) * ngy + (base_y + j)) * ngz + (base_z + k)
                f_node = vol0[p] * (tau * gw) - mass[p] * weight * gravity
                wp.atomic_add(R, idx, f_node)


@wp.kernel
def _qs_commit(
    x_p: wp.array(dtype=wp.vec3d),
    b_e: wp.array(dtype=wp.mat33d),
    u: wp.array(dtype=wp.vec3d),
    lam: f64,
    mu: f64,
    sigma_y: f64,
    inv_dx: f64,
    ngy: wp.int32,
    ngz: wp.int32,
):
    """Converged increment to particles: x += sum w u, and the elastic state advances
    through the same trial + return map as the residual (the plastic part of the
    step stays out of b_e, which is what makes the flow permanent)."""
    p = wp.tid()
    one = f64(1.0)
    half = f64(0.5)
    base_x = wp.int(x_p[p][0] * inv_dx - half)
    base_y = wp.int(x_p[p][1] * inv_dx - half)
    base_z = wp.int(x_p[p][2] * inv_dx - half)
    fx = x_p[p] * inv_dx - wp.vec3d(f64(base_x), f64(base_y), f64(base_z))
    wa = wp.vec3d(half * (f64(1.5) - fx[0]) * (f64(1.5) - fx[0]),
                  half * (f64(1.5) - fx[1]) * (f64(1.5) - fx[1]),
                  half * (f64(1.5) - fx[2]) * (f64(1.5) - fx[2]))
    wb = wp.vec3d(f64(0.75) - (fx[0] - one) * (fx[0] - one),
                  f64(0.75) - (fx[1] - one) * (fx[1] - one),
                  f64(0.75) - (fx[2] - one) * (fx[2] - one))
    wc = wp.vec3d(half * (fx[0] - half) * (fx[0] - half),
                  half * (fx[1] - half) * (fx[1] - half),
                  half * (fx[2] - half) * (fx[2] - half))
    da = wp.vec3d((fx[0] - f64(1.5)) * inv_dx, (fx[1] - f64(1.5)) * inv_dx,
                  (fx[2] - f64(1.5)) * inv_dx)
    db = wp.vec3d(f64(-2.0) * (fx[0] - one) * inv_dx, f64(-2.0) * (fx[1] - one) * inv_dx,
                  f64(-2.0) * (fx[2] - one) * inv_dx)
    dc = wp.vec3d((fx[0] - half) * inv_dx, (fx[1] - half) * inv_dx,
                  (fx[2] - half) * inv_dx)
    w = wp.matrix_from_rows(wa, wb, wc)
    dw = wp.matrix_from_rows(da, db, dc)

    du = wp.vec3d(f64(0.0), f64(0.0), f64(0.0))
    du_grad = wp.mat33d(f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0),
                        f64(0.0), f64(0.0), f64(0.0))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[i, 0] * w[j, 1] * w[k, 2]
                gw = wp.vec3d(dw[i, 0] * w[j, 1] * w[k, 2],
                              w[i, 0] * dw[j, 1] * w[k, 2],
                              w[i, 0] * w[j, 1] * dw[k, 2])
                idx = ((base_x + i) * ngy + (base_y + j)) * ngz + (base_z + k)
                du += weight * u[idx]
                du_grad += wp.outer(u[idx], gw)
    G = wp.identity(n=3, dtype=f64) + du_grad
    b = G * b_e[p] * wp.transpose(G)
    if sigma_y > f64(0.0):
        # twin of the residual's return map: scale the deviatoric Hencky strain so
        # the stored elastic state sits on the yield surface
        Q = wp.mat33d(f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0), f64(0.0),
                      f64(0.0), f64(0.0), f64(0.0))
        lam2 = wp.vec3d(f64(0.0), f64(0.0), f64(0.0))
        wp.eig3(b, Q, lam2)
        e0 = half * wp.log(wp.max(lam2[0], f64(1e-12)))
        e1 = half * wp.log(wp.max(lam2[1], f64(1e-12)))
        e2 = half * wp.log(wp.max(lam2[2], f64(1e-12)))
        tr = e0 + e1 + e2
        d0 = e0 - tr / f64(3.0)
        d1 = e1 - tr / f64(3.0)
        d2 = e2 - tr / f64(3.0)
        vm = f64(2.0) * mu * wp.sqrt(f64(1.5) * (d0 * d0 + d1 * d1 + d2 * d2))
        if vm > sigma_y:
            sc = sigma_y / vm
            n0 = tr / f64(3.0) + sc * d0
            n1 = tr / f64(3.0) + sc * d1
            n2 = tr / f64(3.0) + sc * d2
            b_diag = wp.mat33d(wp.exp(f64(2.0) * n0), f64(0.0), f64(0.0),
                               f64(0.0), wp.exp(f64(2.0) * n1), f64(0.0),
                               f64(0.0), f64(0.0), wp.exp(f64(2.0) * n2))
            b = Q * b_diag * wp.transpose(Q)
    b_e[p] = b
    x_p[p] = x_p[p] + du


class QuasiStaticSolver:
    """Load-stepped equilibrium solves on a fixed background grid.

    Positions and volumes come in like Solver.load_particles; the constitutive law is
    Hencky elasticity (lam, mu from E, nu). Dirichlet conditions are per-node,
    per-component masks (fix_floor for the common case) and displacement-controlled
    tools (prescribe_nodes). yield_stress > 0 turns on rate-independent von Mises
    plasticity via a radial return in principal Hencky space; the particle state is
    then the elastic left Cauchy-Green, and plastic flow is what the return map
    leaves out of it. solve_gravity ramps gravity over load steps; each step is one
    Newton solve."""

    def __init__(self, pos: np.ndarray, vol: np.ndarray, rho: float, E: float,
                 nu: float, grid: GridConfig, yield_stress: float = 0.0,
                 device: str = "cpu"):
        _ensure_warp()
        self.grid = grid
        self.device = device
        self.n_p = len(pos)
        self.ng = np.array([grid.n_grid] * 3)
        self.n_nodes = int(np.prod(self.ng))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))
        self.rho = rho
        self.sigma_y = float(yield_stress)
        self._vol = vol.astype(np.float64)
        self._mass = rho * self._vol

        self.x_p = wp.from_numpy(pos.astype(np.float64), dtype=wp.vec3d, device=device)
        b0 = np.tile(np.eye(3), (self.n_p, 1, 1))
        self.b_e = wp.from_numpy(b0, dtype=wp.mat33d, device=device)
        self.vol0 = wp.from_numpy(self._vol, dtype=f64, device=device)
        self.mass = wp.from_numpy(self._mass, dtype=f64, device=device)
        self.R_wp = wp.zeros(self.n_nodes, dtype=wp.vec3d, device=device)
        self.u_wp = wp.zeros(self.n_nodes, dtype=wp.vec3d, device=device)

        # free-DOF mask: 1 on active, unconstrained (node, component) pairs
        self._mask = np.zeros((self.n_nodes, 3))
        self._refresh_active()

    # ---- masks ------------------------------------------------------------------
    def _refresh_active(self) -> None:
        x = self.x_p.numpy()
        base = np.floor(x * (1.0 / self.grid.dx) - 0.5).astype(int)
        off = np.stack(np.meshgrid([0, 1, 2], [0, 1, 2], [0, 1, 2],
                                   indexing="ij"), -1).reshape(-1, 3)
        ids = ((base[:, None, 0] + off[None, :, 0]) * self.ng[1]
               + (base[:, None, 1] + off[None, :, 1])) * self.ng[2] \
            + (base[:, None, 2] + off[None, :, 2])
        active = np.zeros(self.n_nodes, dtype=bool)
        active[ids.ravel()] = True
        self._active = active
        self._mask[:] = 0.0
        self._mask[active] = 1.0
        self._u_pre = np.zeros((self.n_nodes, 3))
        self._pre_dofs = np.zeros((self.n_nodes, 3), dtype=bool)

    def fix_nodes(self, node_mask: np.ndarray, components: str = "xyz") -> None:
        """Zero the given components of u on the masked nodes."""
        for c, ax in (("x", 0), ("y", 1), ("z", 2)):
            if c in components:
                self._mask[node_mask, ax] = 0.0

    def fix_floor(self, k_plane: int, components: str = "z") -> None:
        """Dirichlet on the node plane k <= k_plane (the floor)."""
        kz = np.arange(self.n_nodes) % self.ng[2]
        self.fix_nodes(kz <= k_plane, components)

    def prescribe_nodes(self, node_mask: np.ndarray, displacement,
                        components: str = "xyz") -> None:
        """Inhomogeneous Dirichlet: a rigid tool in displacement control. The masked
        nodes move by `displacement` (length 3, metres) on EVERY load step of the
        next solve; the affected components leave the free system, and the reaction
        wrench the material exerts on them comes back in solve info as tool_force.
        The node set is fixed until re-prescribed, which is the right granularity for
        small increments; a moving tool re-prescribes between solves."""
        disp = np.asarray(displacement, dtype=float)
        for c, ax in (("x", 0), ("y", 1), ("z", 2)):
            if c in components:
                self._mask[node_mask, ax] = 0.0
                self._u_pre[node_mask, ax] = disp[ax]
                self._pre_dofs[node_mask, ax] = True

    # ---- residual and solve -----------------------------------------------------
    def _residual(self, u_flat: np.ndarray, g_now: float, masked: bool = True
                  ) -> np.ndarray:
        u = (u_flat.reshape(self.n_nodes, 3) * self._mask) + self._u_pre
        wp.copy(self.u_wp, wp.from_numpy(u, dtype=wp.vec3d, device=self.device))
        self.R_wp.zero_()
        wp.launch(_qs_residual, dim=self.n_p,
                  inputs=[self.x_p, self.b_e, self.vol0, self.mass, self.u_wp,
                          wp.vec3d(0.0, 0.0, -g_now), f64(self.lam), f64(self.mu),
                          f64(self.sigma_y),
                          f64(1.0 / self.grid.dx), int(self.ng[1]), int(self.ng[2]),
                          self.R_wp],
                  device=self.device)
        R = self.R_wp.numpy()
        return (R * self._mask).ravel() if masked else R.ravel()

    def solve_gravity(self, g: float = 9.81, n_steps: int = 5, newton_max: int = 30,
                      tol_per_particle: float = 1e-8, verbose: bool = False) -> dict:
        """Ramp gravity over n_steps load steps; one Newton solve per step, committed
        to particle positions and deformation gradients. Returns convergence info."""
        n_dof = self.n_nodes * 3
        info = {"newton_iters": [], "residual_norms": []}
        for step in range(n_steps):
            g_now = g * (step + 1) / n_steps
            u = np.zeros(n_dof)
            rn = np.inf
            for it in range(newton_max):  # noqa: B007  (recorded in info after loop)
                R = self._residual(u, g_now)
                rn = float(np.linalg.norm(R))
                if rn < tol_per_particle * self.n_p:
                    break
                eps = 1e-6 * max(1.0, float(np.linalg.norm(u)))

                def jvp(p, u=u, R0=R, eps=eps, g_now=g_now):
                    return (self._residual(u + eps * p, g_now) - R0) / eps

                J = LinearOperator((n_dof, n_dof), matvec=jvp)
                p, _ = gmres(J, -R, rtol=1e-6, maxiter=400, restart=80)
                u = u + p
            info["newton_iters"].append(it)
            info["residual_norms"].append(rn)
            if self._pre_dofs.any():
                # equilibrium is not enforced on prescribed DOFs; the leftover
                # residual there is the constraint force the tool applies to the
                # material, so the material's reaction on the tool is its negative
                R_raw = self._residual(u, g_now, masked=False).reshape(-1, 3)
                info.setdefault("tool_force", []).append(
                    -np.array([R_raw[self._pre_dofs[:, ax], ax].sum()
                               for ax in range(3)]))
            if verbose:
                print(f"load step {step}: g={g_now:.2f} iters={it} |R|={rn:.3e}")
            # commit: particle positions and the elastic state advance; masks refresh
            # for the moved particles (Dirichlet planes re-applied by the caller)
            u_m = (u.reshape(self.n_nodes, 3) * self._mask) + self._u_pre
            wp.copy(self.u_wp, wp.from_numpy(u_m, dtype=wp.vec3d, device=self.device))
            wp.launch(_qs_commit, dim=self.n_p,
                      inputs=[self.x_p, self.b_e, self.u_wp, f64(self.lam),
                              f64(self.mu), f64(self.sigma_y),
                              f64(1.0 / self.grid.dx),
                              int(self.ng[1]), int(self.ng[2])],
                      device=self.device)
        return info

    # ---- readback -----------------------------------------------------------------
    def x(self) -> np.ndarray:
        return self.x_p.numpy().copy()

    def cauchy_stress(self) -> np.ndarray:
        """Per-particle Cauchy stress from the committed elastic state (numpy, small
        n). b_e is post-return-map, so the stress is already on the yield surface."""
        b = self.b_e.numpy()
        evals, evecs = np.linalg.eigh(b)
        log_e = 0.5 * np.log(np.clip(evals, 1e-12, None))
        tr = log_e.sum(1)
        tau_p = self.lam * tr[:, None] + 2.0 * self.mu * log_e
        tau = np.einsum("nij,nj,nkj->nik", evecs, tau_p, evecs)
        J_e = np.sqrt(np.linalg.det(b))
        return tau / J_e[:, None, None]
