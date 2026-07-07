# Simulator notes (warp-mpm fork at wp-mpm/warp-mpm)

Read-before-changing survey of the fork, plus the conventions the dump
writer and the identification pipeline rely on. Probe results are recorded
at the bottom and mirrored in the dump schema metadata.

## Versions and environment

- warp-lang is PINNED to 1.10.1 (root pyproject). The P2G and G2P kernels
  build their quadratic B-spline weight matrices with wp.mat33(vec, vec, vec),
  whose column-vector semantics are deprecated and CHANGE in newer warp
  releases; upgrading warp without rewriting those constructors corrupts the
  interpolation silently. The fork's own uv.lock also resolves to 1.10.1.
- On macOS warp runs CPU only and single threaded. About 25 to 45 ms per
  p2g2p step at 13k to 20k particles on a 96^3 to 128^3 grid (M3 Max).
- Bug fixed in the fork: load_initial_data_from_torch passed device
  positionally into import_particle_x_from_torch, which binds it to clone
  and silently keeps device="cuda:0". Fatal on CPU, invisible on the CUDA
  box. Now passed as a keyword.

## Architecture (one p2g2p step, mpm_solver_warp.MPM_Simulator_WARP)

1. zero_grid
2. pre-p2g particle operations (impulses, velocity modifiers)
3. compute_stress_from_F_trial: return mapping per material, then stress
   from the mapped elastic F. THE STRESS STORED IN particle_stress IS
   KIRCHHOFF STRESS tau = J sigma, symmetrized.
4. p2g_apic_with_stress: APIC scatter; force term -V_p^0 * tau * grad(w).
   particle_vol is the INITIAL volume V_p^0 and is never updated; current
   volume is J * V_p^0.
5. grid_normalization_and_gravity, then optional damping, then grid BC
   colliders (sticky / slip / friction half-space projections).
6. g2p: PIC velocity pull (no FLIP blending), APIC C update, position
   update, and F_trial = (I + dt * L) F with L = sum_i grid_v_i (x) grad(w_i).
   The row index of L is the velocity component, the column the spatial
   derivative: L_ij = d v_i / d x_j, matching MATH_REFERENCE.md. g2p now
   also stores L into particle_L for the dump (added for TrackEUCLID).

Materials: 0 jelly (FCR), 1 metal (StVK + von Mises), 2 sand
(Drucker-Prager, Klar et al. style), 3 foam (viscoplastic), 4 snow, 5
plasticine, 6 weakly compressible fluid, 7 stationary, 8 rigid coupling,
9 mu_i_sand (TrackEUCLID local mu(I), added here).

Grid is forced cubic (n_grid^3); domain [0, grid_lim]^3, dx = grid_lim /
n_grid. Colliders apply over the whole half space behind the plane.

## The mu(I) constitutive update (material 9, added for TrackEUCLID)

Dunatunga and Kamrin style elastic predictor with a scalar mu(I) plastic
return, chosen over a direct regularized viscoplastic stress evaluation
because the direct form has an effective viscosity mu(I) p / eps_gamma in
quasi-static zones, which collapses the explicit stable time step; the
predictor-return form is stable at the elastic CFL.

Per particle, with Hencky elasticity in the principal frame of F_trial:

1. F_tr = U diag(sig) V^T, eps = log(sig), Kirchhoff trial
   tau_K = 2 G eps + lam tr(eps) Id (correct Hencky Kirchhoff form,
   assembled as U diag(.) U^T, the same form the existing Drucker-Prager
   sand uses via its 1/sig trick).
2. Volumetric expansion tr(eps) > 0: free separation. Stress zero,
   F reset to U V^T (cohesionless, no tension memory).
3. Else Cauchy pressure p_C = -tr(tau_K) / (3 J) and equivalent shear
   tau_bar_C = |dev tau_K| / (sqrt(2) J). The constitutive family
   tau = mu(I) p (2 D / |gamma_dot|) implies |dev sigma| = sqrt(2) mu p,
   so the yield surface is tau_bar_C = mu(I) p_C.
4. If tau_bar_C <= mu_s p_C: elastic step, accept trial.
5. Else solve the scalar return g(gdp) = tau_bar_K_tr - G dt gdp
   - mu(I(gdp, p_C)) p_K = 0 for the plastic shear rate gdp, where
   I = gdp * d * sqrt(rho_s / p_C) and mu(I) = mu_s + delta_mu I/(I+I0).
   g is strictly decreasing with a sign change on [0, tau_bar_K_tr/(G dt)],
   solved by 48 bisection iterations (handles constant mu, delta_mu = 0,
   and the Pouliquen form uniformly).
6. Deviatoric, non-dilatant update: eps_dev scaled by the returned
   tau_bar ratio, volumetric part unchanged, so J and p are preserved by
   the return. F = U diag(exp(eps_new)) V^T; stress = Kirchhoff of eps_new.

Which pressure does the update consume: the elastic-predictor Cauchy mean
stress p_C = -tr(tau_K)/(3J). Because the return is trace preserving, the
dumped Cauchy stress satisfies -tr(sigma)/3 = p_C exactly (float roundoff
aside). Solver pressure and stress-trace pressure coincide by construction;
the dump therefore carries the stress tensor only and the identifier always
recomputes p = -(sigma_xx + sigma_yy + sigma_zz)/3 through
conventions.pressure_from_cauchy_3d_trace. No second pressure dataset is
needed and no flag is raised.

### Constitutive verification (sim/verify_mu_i.py, PASSED)

An independent numpy reference of the identical return-map math cross-checks
the warp kernel and the physical conditions the identifier assumes. Results
on 400 random elastic predictors (G=4e5, lam=6e5, mu_s=0.38, delta_mu=0.26,
I0=0.3, d=1e-3, rho_s=2650, dt=5e-5):

- warp vs numpy reference: median 8e-7, p99 7e-6 (float32 round-off; one
  near-boundary sample at 7e-4 where f32/f64 split the elastic/plastic
  branch, harmless because stress is continuous there).
- plastic states land on the yield surface tau_bar_C = mu(I) p_C to 2e-7,
  with tau_bar_C = |dev sigma|/sqrt(2) and I = |gamma_dot_p| d sqrt(rho_s/p_C).
  CRITICAL: this uses |gamma_dot| = sqrt(2 D:D), the SAME convention as
  common/conventions.py, so the plastic multiplier gdp returned by the
  scalar solve equals |gamma_dot_p| exactly (derivation in the script
  header). Sim truth and identifier are on one convention; the canonical
  factor-of-2 bug is excluded by test.
- pressure preservation: |p_C(out) - p_C(predictor)|/p_C worst 2.5e-7, so
  the dumped 3D stress trace equals the pressure the update consumed.
- cohesionless volumetric expansion returns zero stress.

Parameters (per-type arrays, set through set_parameters_dict /
set_parameters_for_particles): mu_s, delta_mu, I0, grain_diameter,
grain_density, plus E, nu for the elastic predictor. Constant-mu oracle
runs use delta_mu = 0.

## Time-step control (scene driver)

dt = CFL * dx / c with c the largest of:
- elastic p-wave speed sqrt((lam + 2 G) / rho)
- max particle speed (advective limit)
- frictional speed sqrt(mu_2 p_max / rho) (always dominated by the elastic
  speed here; logged anyway as the spec requires)
The scene logs the governing constraint per logging interval and asserts
v_max dt < dx each dump frame. dt is rounded down to an integer divisor of
the 1 ms dump interval so frames land exactly on steps.

## Probe results (sim/probe_l_convention.py)

- L convention: CONFIRMED L_ij = dv_i/dx_j, so the material acceleration is
  a = dv/dt + L @ v (contraction L @ v, not L^T @ v). Decided by a local
  least-squares fit of dv_i/dx_j over each particle's neighbours and
  comparing to the dumped L and its transpose: median relative error 0.023
  against L vs 0.89 against L^T. The dump records this string in
  L_convention metadata; the probe is the regression guard.
- div(v) from L: in-plane median about 0.09 / s against |L| of a few per
  second. The mu(I) elastic predictor is weakly compressible (not enforced
  divergence free), which is physical for this elastoplastic model and
  harmless because the identifier treats pressure as data.
- acceleration: trajectory finite differences are the identification default
  (particles are material points, so the per-track velocity difference IS
  Dv/Dt). Grid forces are not exposed by this fork; the dv/dt + L @ v
  reconstruction is available as the cross-check and the L convention above
  is what makes it well defined.
- static column pressure vs P0: during collapse the interior is NOT
  hydrostatic (true pressure is dumped for exactly this reason), and the
  column starts stress-free at t = 0 so an elastic transient precedes
  lithostatic loading. The gate-transient rule (MATH_REFERENCE Section 4)
  drops the early frames. The P0 convention itself is unit-tested in
  tests/test_conventions.py; a dedicated settled-column pressure check is a
  later diagnostic, not a Phase 1 blocker.
