"""Quasi-static implicit MPM prototype (numpy): elastic column settling under gravity.

Formulation (GeoWarp quasi_static_solver, MIT, reimplemented; no autodiff):
  unknown   u        grid displacement increment this load step (active DOFs)
  trial     F_new_p  = (I + grad(du)_p) F_old_p,  grad(du)_p = sum_i u_i outer grad(w_ip)
  stress    Kirchhoff tau from Hencky strain: tau = lambda tr(eps) I + 2 mu eps,
            eps = 0.5 log(b),  b = F F^T   (principal-space log via eigh)
  residual  R_i = sum_p V0_p tau_p grad(w_ip) - sum_p m_p g w_ip     (per component)
  solve     Newton; J p ~ (R(u + eps p) - R(u)) / eps, GMRES (matrix-free, FD JVP)
  BCs       Dirichlet u_z = 0 on the floor node plane (slip), R zeroed there
  transfers quadratic B-splines (our engine's), NOT GeoWarp's GIMP

Gate: any column in equilibrium with traction-free sides has sigma_zz = -rho g (h - z),
independent of the constitutive law. Compare the laterally averaged sigma_zz profile.
"""
import itertools

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

rng = np.random.default_rng(0)

# ---- scene: a 0.12 x 0.12 x 0.36 m column on a floor ------------------------------
dx = 0.03
inv_dx = 1.0 / dx
ng = np.array([12, 12, 20])          # grid nodes per axis (small dense grid)
floor_k = 3                          # floor at z = 3 dx (node plane)
col = np.array([0.12, 0.12, 0.36])
org = np.array([4, 4, floor_k]) * dx
ppd = 2
hp = dx / ppd
ax = [np.arange(hp / 2, c, hp) for c in col]
X0 = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3) + org
n_p = len(X0)
V0 = np.full(n_p, hp ** 3)
rho, g = 1000.0, 9.81
m_p = rho * V0
E, nu = 1.0e5, 0.3
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
n_nodes = int(np.prod(ng))
print(f"{n_p} particles, grid {ng}, {n_nodes} nodes")

F_old = np.tile(np.eye(3), (n_p, 1, 1))
x_p = X0.copy()


def weights(xp):
    """Quadratic B-spline weights/gradients (matches the engine's kernels)."""
    base = np.floor(xp * inv_dx - 0.5).astype(int)          # (n, 3)
    fx = xp * inv_dx - base                                  # (n, 3)
    w = np.stack([0.5 * (1.5 - fx) ** 2,
                  0.75 - (fx - 1.0) ** 2,
                  0.5 * (fx - 0.5) ** 2], axis=1)            # (n, 3, 3) [offset, axis]
    dw = np.stack([(fx - 1.5), -2.0 * (fx - 1.0), (fx - 0.5)], axis=1) * inv_dx
    return base, w, dw


def scatter_ids(base):
    """(n, 27) flat node ids and the (27, 3) offset table."""
    off = np.stack(np.meshgrid([0, 1, 2], [0, 1, 2], [0, 1, 2],
                               indexing="ij"), -1).reshape(-1, 3)
    ids = ((base[:, None, 0] + off[None, :, 0]) * ng[1]
           + (base[:, None, 1] + off[None, :, 1])) * ng[2] \
        + (base[:, None, 2] + off[None, :, 2])
    return ids, off


def hencky_tau(F):
    """Kirchhoff stress from Hencky strain, batched."""
    b = F @ np.transpose(F, (0, 2, 1))
    evals, evecs = np.linalg.eigh(b)
    log_e = 0.5 * np.log(np.clip(evals, 1e-12, None))        # principal Hencky strains
    tr = log_e.sum(1)
    tau_p = lam * tr[:, None] + 2.0 * mu * log_e             # principal Kirchhoff
    return np.einsum("nij,nj,nkj->nik", evecs, tau_p, evecs)


base, w3, dw3 = weights(x_p)
ids, off = scatter_ids(base)
W = (w3[:, off[:, 0], 0] * w3[:, off[:, 1], 1] * w3[:, off[:, 2], 2])      # (n, 27)
Gw = np.stack([dw3[:, off[:, 0], 0] * w3[:, off[:, 1], 1] * w3[:, off[:, 2], 2],
               w3[:, off[:, 0], 0] * dw3[:, off[:, 1], 1] * w3[:, off[:, 2], 2],
               w3[:, off[:, 0], 0] * w3[:, off[:, 1], 1] * dw3[:, off[:, 2], 2]],
              axis=-1)                                                      # (n, 27, 3)

# active DOFs: nodes with support; Dirichlet: floor plane u_z = 0
node_mass = np.zeros(n_nodes)
np.add.at(node_mass, ids.ravel(), (m_p[:, None] * W).ravel())
active = node_mass > 1e-12
kz = np.arange(n_nodes) % ng[2]
dirichlet_z = active & (kz <= floor_k)                      # floor plane and below
n_act = int(active.sum())
print(f"{n_act} active nodes, {int(dirichlet_z.sum())} floor-constrained")


def residual(u_flat, g_now):
    """R(u): internal - external force on active nodes; u is (n_nodes, 3) flattened."""
    u = u_flat.reshape(n_nodes, 3)
    u = np.where(active[:, None], u, 0.0)
    u[dirichlet_z, 2] = 0.0
    du_grad = np.einsum("pki,pkj->pij", u[ids], Gw)          # (n, 3, 3) sum u_i grad w
    F_new = (np.eye(3)[None] + du_grad) @ F_old
    tau = hencky_tau(F_new)
    fint = np.einsum("p,pij,pkj->pki", V0, tau, Gw)          # (n, 27, 3)
    fext = (m_p[:, None] * W)[:, :, None] * np.array([0.0, 0.0, -g_now])[None, None]
    R = np.zeros((n_nodes, 3))
    np.add.at(R, ids.reshape(-1), (fint - fext).reshape(-1, 3))
    R[~active] = 0.0
    R[dirichlet_z, 2] = 0.0
    return R.ravel()


u = np.zeros(n_nodes * 3)
n_steps = 5
for step in range(n_steps):
    g_now = g * (step + 1) / n_steps
    for it in range(30):  # noqa: B007  (reported after the loop)
        R = residual(u, g_now)
        rn = np.linalg.norm(R)
        if rn < 1e-8 * n_p:
            break
        eps = 1e-6 * max(1.0, np.linalg.norm(u))

        def jvp(p, u=u, R0=R, eps=eps, g_now=g_now):
            return (residual(u + eps * p, g_now) - R0) / eps

        J = LinearOperator((n_nodes * 3, n_nodes * 3), matvec=jvp)
        p, info = gmres(J, -R, rtol=1e-6, maxiter=400, restart=80)
        u = u + p
    print(f"step {step}: g={g_now:.2f} newton_iters={it} |R|={rn:.3e}")

# commit the converged increment and check the stress profile
uu = u.reshape(n_nodes, 3)
du_grad = np.einsum("pki,pkj->pij", uu[ids], Gw)
F_fin = (np.eye(3)[None] + du_grad) @ F_old
tau = hencky_tau(F_fin)
J_det = np.linalg.det(F_fin)
sig_zz = tau[:, 2, 2] / J_det                               # Cauchy

z_rel = x_p[:, 2] - org[2]
h = col[2]
bins = np.linspace(0, h, 13)
mid = 0.5 * (bins[:-1] + bins[1:])
prof = np.array([sig_zz[(z_rel >= a) & (z_rel < b)].mean()
                 for a, b in itertools.pairwise(bins)])
exact = -rho * g * (h - mid)
err = np.abs(prof - exact) / (rho * g * h)
print("\n z/h    sigma_zz     exact     rel.err")
for zm, pv, ev, er in zip(mid / h, prof, exact, err, strict=True):
    print(f"{zm:5.2f} {pv:10.1f} {ev:10.1f} {er:9.4f}")
print(f"\nmax relative error vs rho g h: {err.max():.4f}")
print(f"top-of-column settlement: {uu[active, 2].min()*1000:.2f} mm "
      f"(free-sided analytic rho g h^2 / (2 E) ~ {rho*g*h*h/2/E*1000:.2f} mm)")
