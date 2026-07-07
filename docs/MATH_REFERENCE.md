# TrackEUCLID Mathematical Reference (docs/MATH_REFERENCE.md)

Math authority for all phases. When code and this document disagree, the document wins. G0 is the executable arbiter of every convention. Function-encoder training mathematics lives in docs/FUNCTION_ENCODER.md (Phase 3); only its interface contract appears here.

## 0. Coordinate convention (read first)

Config A (quasi-2D, plane strain): x horizontal, z vertical UPWARD, y out-of-plane. In-plane fields depend on (x, z, t) only; v_y = 0 and w_y = 0. The in-plane gravity vector is g = (0, -g_mag) with g_mag = 9.81. The out-of-plane normal stress is sigma_yy; it exists in plane strain and never enters in-plane rows. Scalar-gravity formulas (with explicit signs) appear ONLY inside common/conventions.py helpers and in Section 5 below; all assembly formulas use the gravity vector.

## 1. Continuum setting and constitutive family

  div v = 0
  rho Dv/Dt = div sigma + rho g
  sigma = -p I + tau

  D = sym(grad v),  (grad v)_{ij} = d v_i / d x_j
  |gamma_dot|_eps = sqrt(2 D : D + eps_gamma^2),  eps_gamma default 0.02 / s
  I = |gamma_dot|_eps d / sqrt(p / rho_s)
  tau = mu(I) p (2 D / |gamma_dot|_eps)

Pressure from a dumped 3D Cauchy stress, ALWAYS the 3D trace, never the 2D trace:

  p = -( sigma_xx + sigma_yy + sigma_zz ) / 3

implemented once as conventions.pressure_from_cauchy_3d_trace(...) with a unit test in which sigma_yy is nonzero.

Linear basis, load-bearing: mu(I) = sum_k theta_k phi_k(I).

Dictionary modes behind one interface (Section 7):
- Mode C: phi_1 = 1.
- Mode P: mu(I) = mu_s + sum_g c_g f_g(I), f_g(I) = I / (I + I_0g), I_0g on a log grid in [1e-3, 1], c_g >= 0 with L1 selection. f_g(0) = 0, f_g(inf) = 1, so mu_s rides the constant feature and sum c_g is the friction rise.
- Mode F: learned basis, Phase 3, docs/FUNCTION_ENCODER.md.

## 2. Weak form: full derivations

### 2.1 Master identity (acceleration form)

Patch Q = Omega_j x [t0, t1], test field w with w = 0 on the spatial patch boundary and div w = 0:

  INT_Q rho a . w = OINT (sigma n) . w - INT_Q sigma : grad w + INT_Q rho g . w

Compact support kills the boundary integral (all unknown tractions). sigma : grad w = -p (div w) + tau : D[w] = tau : D[w]. Hence

  INT_Q tau : D[w] = INT_Q rho (g - a) . w                          (M1)

Rows: A[j,k] = INT_Q phi_k(I) p (2 D / |gamma_dot|_eps) : D[w_j],  b[j] = INT_Q rho (g - a) . w_j. p and I are data; linearity in theta holds for fixed p.

### 2.2 Time-weak forms (no second derivatives of data)

Lagrangian (tracks), with w(., t0) = w(., t1) = 0 and m_p = rho V_p:

  INT_Q rho a . w = - SUM_p m_p INT v_p . [ dw/dt + (v_p . grad) w ](x_p(t), t) dt

Eulerian, fully weak, using incompressibility, constant rho, compact w, and outer product (v (x) v)_{ij} = v_i v_j:

  INT_Q rho a . w = - INT_Q rho [ v . dw/dt + (v (x) v) : grad w ]   (M2)

(x) denotes the tensor outer product, NOT a cross product. Both (M1 with measured a) and (M2 or the Lagrangian variant) are always implemented and reported; disagreement is a refusal component.

### 2.3 Momentum-closure diagnostic

For the true material (M1) holds for any admissible w. Near-rigid fields w = chi e_i (not divergence free) check the full identity including the closure pressure; strictly admissible bumps check the pressure-free identity. Per-patch relative closure error localizes bad kinematics, pressure, or masks. Built before any regression.

### 2.4 Quadrature

Particle quadrature INT f dV approx SUM_p V_p f(x_p); trapezoid in time. Convergence verified in G0 by halving particle spacing and frame interval; the observed rate sets all tolerances.

### 2.5 Production estimator on MPM data: grid-consistent (Bubnov-Galerkin) assembly

The analytic bump test functions of Sections 2-3 are correct and are the basis of G0 and the divergence-free pressure-free analysis, but on real warp-mpm particle data the directly-sampled bump form is patch-radius-dependent and biased (it returns mu in roughly [0.04, 0.16] on the constant-mu collapse vs truth 0.38). The estimator that actually recovers mu to ~1% is the GRID-CONSISTENT (Bubnov-Galerkin) assembly: project particle quantities to the background B-spline grid with the SAME shape functions the MPM transfer uses, build the test functions and their gradients on that grid, and integrate there (ident/weakform/grid_assembly.py:assemble_grid_consistent). Rationale and the bump-vs-grid comparison are in docs/identification_notes.md. This is the load-bearing production method for all oracle-MPM gates; the bump form is retained for G0/manufactured-solution and divergence-free analysis. When this document's bump prescription and the grid-consistent code disagree on an MPM dump, the grid-consistent code is the production method (and this note records why), not a violation of the code-follows-document rule.

## 3. Test functions (Config A, in-plane coordinates x, z)

Generator along the out-of-plane axis: w = curl(eta e_y). Componentwise:

  w_x = - d eta / d z,    w_z = + d eta / d x,    w_y = 0,    div w = 0 identically.

Separable C2 generator on the space-time patch:

  eta(x, z, t) = r_x r_z B(u) B(s) B(q),  u = (x - x_c)/r_x,  s = (z - z_c)/r_z,  q = (t - t_c)/r_t
  B(zeta)  = (1 - zeta^2)^3            for |zeta| < 1, else 0
  B'(zeta) = -6 zeta (1 - zeta^2)^2
  B''(zeta) = 6 (1 - zeta^2)(5 zeta^2 - 1)

Explicit components and ALL gradient entries (implement these verbatim; do not re-derive):

  w_x = - r_x B(u) B'(s) B(q)
  w_z = + r_z B'(u) B(s) B(q)

  d w_x / d x = - B'(u) B'(s) B(q)
  d w_x / d z = - (r_x / r_z) B(u) B''(s) B(q)
  d w_z / d x = + (r_z / r_x) B''(u) B(s) B(q)
  d w_z / d z = + B'(u) B'(s) B(q)

Divergence check by inspection: d w_x / d x + d w_z / d z = 0. D[w] = sym(grad w) from these entries. Time compactness (B(q) factor) serves the time-weak form. Per patch: full scale, half scale centered, and an offset full-scale copy, 3 rows. Unit tests: div w = 0 to machine precision at random points, support containment, C1 continuity at the support boundary by finite differences, and a sign-flip negative control (Section 9).

3D (Config B, deferred): w = curl(eta c f), c in {e_x, e_y, e_z}, f in {1, finer bump}; same tests. Rigid and tool fields as before for diagnostics and force rows.

## 4. Masks and patches

Flowing mask: |gamma_dot|_eps > gamma_min (0.5 / s) AND I > I_min (1e-4). Validity mask: distance to front > c_f d (10), distance to free surface and base > c_s d (5), flowing-layer thickness h_flow / d > c_h (8) along the local inward surface normal, density criterion under the active density model. Gate transient: drop t < max(2 sqrt(d/g_mag), measured gate-clearance time, closure-diagnostic spike interval). Patches: space-time boxes in the mask intersection, stratified across realized-I deciles (the conditioning control for Modes P and F).

## 5. Pressure closures (signs derived, not assumed)

Vertical momentum with z upward, g_z = -g_mag, neglecting the deviatoric-stress divergence and horizontal coupling (exactly the neglect G1P prices):

  rho a_z = - d p / d z + rho g_z      =>      d p / d z = rho (g_z - a_z)

Integrating downward from the free surface where p(h) = 0:

  P1:  p(x, z, t) = INT from z to h(x,t) of rho ( a_z(x, z', t) + g_mag ) dz'
  P0:  p(x, z, t) = rho g_mag ( h(x, t) - z )         (the a_z = 0 limit of P1)

Required unit tests: P0 returns rho * 9.81 * (h - z) on a static column; P1 reduces to P0 when a_z = 0; both implemented only via conventions helpers, never with locally written signs. P1 is the default CANDIDATE closure; G1P decides admissibility. P2 (frozen-I alternation) is an engineering diagnostic only.

Bias structure: with p_est = p_true (1 + delta), the constant-mu estimator satisfies, to first order, mu_hat / mu = row-weighted mean of p_true / p_est. Away from the crossover the dominant effect of pressure bias is multiplicative in the recovered friction; near the mu(I) transition pressure also relabels I through I proportional to p^(-1/2) and can alter apparent rate dependence. The region-split diagnostic (8.4) catches both and depends only on locality.

Sensitivity operator: theta(p) = (A^T A + lambda G)^{-1} A^T b; dA has two channels per entry, the multiplicative p factor and the basis argument with d phi_k / d p = (d phi_k / d I)(-I / (2 p)). NOTE: this requires d phi / d I in PHYSICAL I, not d phi / d log10 I; see the interface contract in Section 7. Closed form, finite-difference checked in G0, propagated to var(mu(I)) = phi(I)^T Sigma_theta phi(I), validated against the ensemble.

## 6. Perception and kinematic reconstruction (Config A)

### 6.1 Geometry

Undistort (radial-tangential model k1, k2, p1, p2, k3). Homography H from the glass plane (x, z world) to pixels by DLT on AprilTag corners (6 or more tags), Levenberg-Marquardt refinement, metric scale from tag edge. Ensemble axis: corner jitter sigma_px = 0.3 px, refit, propagate.

### 6.2 Masks, surface, V_n

SAM2 mask, open-close radius 2 px, largest component. Free surface S(t) = boundary not adjacent to base, walls, gate; subpixel refinement via intensity-gradient maximum within 3 px of the boundary along image columns. V_n from signed distance functions: V_n approx [SDF_t - SDF_{t+dt}] / dt on S(t), n = grad SDF / |grad SDF|. Volume V(t) = mask area x W; phi(t) = m / (rho_s V(t)) is the dilation diagnostic.

### 6.3 Tracking

CoTracker3: uniform query grid (8 to 12 px spacing) in the mask, re-seeded every 0.1 s, windowed offline mode, visibility logits to confidence c_p, discard segments with c_p < 0.5 or homography-mapped inconsistency. Cross-check: classical PIV (32 x 32 px windows, 50 percent overlap, FFT correlation, three-point Gaussian subpixel peak). The track-vs-PIV velocity discrepancy field is a perception-uncertainty estimate and an ensemble axis.

### 6.4 Track derivatives and noise

Per-track weighted quadratic fits over N samples at spacing dt: v_hat = b, a_hat = 2c. Leading-order variances under iid position noise sigma_x:

  var(v_hat) approx 12 sigma_x^2 / (N^3 dt^2)
  var(a_hat) approx 720 sigma_x^2 / (N^5 dt^4)

At 500 fps, sigma_x = 0.2 mm, N = 25: std(v) about 0.004 m/s, std(a) about 0.6 m/s^2, below the order-g signal. The time-weak form removes a_hat entirely and is always run in parallel.

### 6.5 Streamfunction field fit (x, z)

psi(x, z, t) on tensor cubic B-splines (spatial knots about 2x mean inter-track spacing, temporal knots every 4 frames); v = curl(psi e_y), so v_x = - d psi / d z, v_z = + d psi / d x, divergence free by construction. Objective: confidence-weighted track misfit + lambda_n (v . n - V_n)^2 on S(t) + lambda_b base-condition misfit (model variant) + lambda_s |Laplacian psi|^2 + lambda_t |d psi / d t|^2. Regularizers by GCV, GCV(lambda) = (1/m)|(Id - S_lambda) y|^2 / [(1/m) tr(Id - S_lambda)]^2, Hutchinson trace, then frozen. Derived fields analytic; field-consistent acceleration a = dv/dt + (v . grad) v = dv/dt + L v with L_{ij} = d v_i / d x_j (so the contraction is L @ v; a probe test in simulator_notes.md confirms the dump's convention by matching trajectory acceleration against L @ v vs L^T @ v).

### 6.6 Ensemble axes

Homography jitter, mask erosion/dilation 1 to 2 px, surface method (SDF vs column gradient), track bootstrap 80 percent, confidence threshold 0.4 to 0.6, velocity source (tracks vs PIV), regularizers x3 and /3, base condition, density model (phi_0 vs phi(t)), pressure closure (P0 vs P1), mask constants halved and doubled, patch reseeding. 50 to 200 members; every downstream solve is milliseconds.

## 7. Dictionary interface (binding contract; Mode F details in docs/FUNCTION_ENCODER.md)

Assembly and solve code NEVER sees log coordinates. The interface takes physical inertial number I:

  Dictionary.K -> int
  Dictionary.phi(I: ndarray) -> ndarray (len(I), K)
  Dictionary.dphi_dI(I: ndarray) -> ndarray (len(I), K)        # physical-I derivative, used by the pressure sensitivity operator
  Dictionary.dphi_dlogI(I: ndarray) -> ndarray (len(I), K)     # = dphi_dI * I * ln(10); used only for monotonicity constraint rows
  Dictionary.gram(weight) -> ndarray (K, K)
  Dictionary.metadata -> dict (mode, support, nonnegativity pattern)

Implementations: ConstantDict, PouliquenGridDict, FunctionEncoderDict (table-backed; converts to s = log10 I internally; cubic interpolation; analytic chain rule for dphi_dI = dphi_ds / (I ln 10)). Mixing d phi / d I, d phi / d log I, and d phi / d p is the designated bug class for this module; the unit tests cross-check the three against finite differences on every implementation.

Encoding-through-linear-functionals lemma (why Mode F composes with EUCLID): the weak-form observations L_j[mu] = b_j are linear in mu, so A_jk = L_j[phi_k] and coefficient recovery is the same constrained least squares for every mode. Identifiability = column rank of A, governed by the basis Gram matrix and realized (I, D, w) diversity; hence I-stratified patches and joint multi-aspect-ratio systems.

## 8. Solve, uncertainty, diagnostics

8.1 Estimator: minimize |A theta - b|^2 + lambda theta^T G theta subject to C theta >= h (admissibility: mu >= mu_min on a log-I grid; optional monotonicity via dphi_dlogI rows; Mode P nonnegativity pattern). theta^T G theta = ||sum theta_k phi_k||^2 in L2_w. lambda by GCV or L-curve. Rows scaled by inertia-gravity magnitude; report cond(A^T A), effective rank at 1e-8, row-survival fraction.

8.2 Conditional posterior: sigma_hat^2 = |r|^2 / (m - dof_lambda); Sigma_theta = sigma_hat^2 (A^T A + lambda G)^{-1} (heteroscedastic sandwich variant also implemented); bands var(mu(I)) = phi(I)^T Sigma_theta phi(I), drawn only over the observed I band; constraint-active cases by truncated-Gaussian sampling.

8.3 Ensemble posterior: mixture over members; envelope and quantiles; analytic pressure-sensitivity bands overlaid, agreement reported.

8.4 Region-split same-I consistency: R_front = flowing rows with front distance in [c_f d, 2 c_f d]; R_bulk = front distance > 4 c_f d and depth > 2 c_s d. Independent solves; agreement evaluated ONLY on the intersection of empirical I supports; require at least half a decade of overlap and at least 20 percent of each set's rows, else return INSUFFICIENT_OVERLAP. Disagreement beyond combined bands at matched I is the designated falsification signal for pressure closure or locality.

8.5 Sidewall column (Phase 2): f_w = -(2 mu_w p / W) v / |v|_eps; column A[j, wall] = INT (2 p / W)(v / |v|_eps) . w_j; mu_w fixed to tilt value or bounded, never free; principal-angle collinearity report per collapse and stacked; W/d material-variation tolerance fixed in advance: exceed the ensemble 90 percent band or 0.04 in friction coefficient, whichever larger, declares apparatus dependence.

8.6 Observability gating and dual-form agreement: reject rows whose ensemble variance exceeds half their magnitude; report survival fraction; acceleration-form vs time-weak mu(I) compared over the observed band, disagreement is a refusal component.

## 9. G0 protocol (expanded per review)

1. Trivial case FIRST: constant p, constant mu, simple analytic divergence-free v, one bump test function, both sides by dense quadrature. Must match to quadrature error.
2. Sign-flip NEGATIVE CONTROL: flipping the sign of b or of D[w] must fail by order one, not pass within tolerance. A G0 that cannot fail is not a gate.
3. Full manufactured solution: analytic v(x, z, t), p(x, z, t), mu(I); convergence under quadrature refinement sets all tolerances.
4. Derivative cross-checks: dictionary phi / dphi_dI / dphi_dlogI vs finite differences; sensitivity operator vs finite differences; test-function gradients vs finite differences.
5. Acceleration identity on oracle data: trajectory finite differences vs grid forces (if exposed) vs dv/dt + L @ v; the L-convention probe lives here.

## 10. Validation metrics (Stage 6)

Synthetic: weighted relative L2 of mu_hat over the realized band. Field: held-out-window velocity error. Observable: front trajectory x_f(t); deposit profile Hausdorff and area difference. Transfer: identify on a training subset of (a, W/d), validate on held-out (a, W/d); runout-vs-aspect-ratio against Lube and Lajeunesse scaling forms with fitted exponents and confidence intervals. Statics: low-I limit of mu_hat against the repose prior interval, consistency check only.

## 11. Symbol table

a material acceleration; A feature matrix; b load vector; B bump kernel (1 - zeta^2)^3; c_p track confidence; d grain diameter; D rate of deformation; D[w] sym(grad w); eps_gamma shear-rate regularization; eta test generator; g in-plane gravity vector (0, -g_mag); g_mag 9.81; G Gram matrix; H homography; h free-surface height; I inertial number; K basis size; L velocity gradient, L_{ij} = d v_i / d x_j; p pressure (3D trace convention); phi_k basis functions; pack_frac packing fraction (phi in prose); psi streamfunction; Q space-time patch; rho bulk density; rho_s grain density; S(t) free surface; sigma Cauchy stress; tau deviatoric stress; theta coefficients; V_n surface normal velocity; V_p particle volume; w test field; W channel width; x horizontal; y out-of-plane; z vertical upward.
