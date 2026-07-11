# Quasi-static implicit MPM (Phase 4b)

Press, squeeze, and shaping scenes move tools at cm/s while the explicit integrator
spends its substeps resolving sound waves; dt is chained to the bulk modulus through
the acoustic CFL. A quasi-static implicit solver removes the timestep restriction for
those scenes by solving equilibrium directly. This note fixes the formulation, what is
borrowed, and the validation gates. The explicit engine stays the reference oracle
throughout.

## Formulation

Reference: GeoWarp's quasi-static solvers (Yidong Zhao, MIT license; reimplemented,
not copied; cite the repository). Displacement-increment form, one nonlinear solve per
load step:

- Unknown: grid displacement increment u on active nodes (nodes with particle
  support). Dirichlet DOFs come from colliders (SDF/plane contact as constrained
  nodes) and are removed from the system.
- Trial kinematics per particle: grad(du)_p = sum_i u_i outer grad(w_ip), then
  F_trial = (I + grad(du)) F_old.
- Stress: Kirchhoff tau from Hencky strain (principal-space log of b = F F^T), which
  keeps large rotations exact and reduces to linear elasticity for small strain.
  Return mapping for plastic materials happens inside the same trial-stress call, so
  the viscoplastic materials port unchanged.
- Residual per active node: R_i = sum_p V0_p tau_p grad(w_ip) - f_ext,i (gravity,
  tractions, tool loads), with load stepping (gravity or tool displacement ramped
  across steps).
- Nonlinear solve: Newton-Krylov, matrix-free. The Jacobian action is a
  finite-difference directional derivative J p ~ (R(u + eps p) - R(u)) / eps and the
  linear solve is GMRES. GeoWarp assembles the sparse Jacobian with warp's autodiff
  tape instead; we deliberately do not, because the repository invariant forbids
  adding autodiff requirements to the simulator, and matrix-free FD-JVP needs only
  residual evaluations (two per Krylov iteration). If Krylov iteration counts become
  the bottleneck, the assembled-Jacobian route is the known fallback and the
  invariant question goes to the user first.
- Transfers: our quadratic B-splines, not GeoWarp's GIMP. Same weights as the
  explicit kernels, so dumps and identification tooling read both engines
  identically.
- After convergence: x_p += sum_i w_ip u_i, F_old = F_trial; the reaction wrench on a
  constrained tool is the residual summed over its Dirichlet nodes (the same
  grid-impulse idea the explicit wrench readout uses, in static form).

## Validation gates (in order)

1. Equilibrium column (analytic): an elastic column with traction-free sides must
   give sigma_zz = -rho g (h - z) regardless of the constitutive law. Numpy
   prototype gate; also checks Newton convergence per load step.
2. Elastic settling and plate press vs slow explicit runs: displacement fields and
   plate wrench agree within tolerance as the explicit run's loading rate goes to
   zero.
3. Identification equivalence: a dough squeeze whose implicit dump yields the same
   identified (tau_y, eta) as the explicit dump. This is the gate that matters for
   the science.

## Status

- Gate 1 PASSED (experiments/qs_prototype.py, numpy): elastic column, 1536 particles,
  5 gravity load steps. Newton converges in 2 iterations per step to |R| ~ 1e-6
  (quadratic, as the small-strain regime predicts). The sigma_zz profile matches
  -rho g (h - z) to 0.1 to 0.5 percent in the interior; the two bins within ~2 cells
  of the floor deviate (4.5 and 19 percent), which is the usual MPM basal boundary
  layer, the same artifact the explicit engine shows near Dirichlet planes. Top
  settlement 6.70 mm vs the free-sided analytic rho g h^2 / (2E) = 6.36 mm (5
  percent).
- Warp port landed: src/warpmpm/implicit/quasistatic.py (float64 kernels for the
  residual scatter and the commit; wp.eig3 for the principal-space log; masks and
  GMRES in numpy/scipy). Passes the same gate 1 as the prototype
  (tests/test_implicit_quasistatic.py).
- Next: contact-as-Dirichlet from the SDF colliders and the tool wrench from the
  residual on constrained nodes, gate 2 (implicit vs slow-explicit plate press with
  matching wrench), then gate 3 (identification equivalence).
