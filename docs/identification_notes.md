# Identification notes (Phase 1, G1 oracle): the weak-form discretization issue

Status as of the overnight session. Records the Step-4 diagnosis and the fix
plan so the work is resumable. Read together with docs/MATH_REFERENCE.md and
docs/simulator_notes.md.

## What works

- G0 manufactured solution passes (analytic fields on a regular grid recover
  theta under quadrature refinement; sign-flip negative control fails by
  order one; derivative and sensitivity cross-checks pass).
- The mu(I) MPM material is verified pointwise (sim/verify_mu_i.py): the
  granular column-collapse dump satisfies the local law to machine precision.
  Measured on the realized flowing region of column_constant_a2 (mu_s=0.38):
    |dev sigma| / (sqrt2 p) = 0.380  (median, p25, p75 all 0.380)
    coaxiality dev(sigma):D / (|.||.|) = 1.000  (frac > 0.8 is 0.98)
  So the data is a faithful realization of mu = 0.38, co-axial, on yield.
- The assembled stress-power column is internally exact: with the true
  deviatoric stress from the dump,
    INT tau_true : D[w]  /  A_assembled  =  0.3796  ==  mu_true
  for the same particles, quadrature and test functions. The assembly of A is
  correct.

## The Step-4 failure and its diagnosis

Solving A theta = b for constant mu on column_constant_a2 (n_grid=80, dx=4mm)
gives mu_hat that is badly biased and, decisively, DEPENDS ON THE
TEST-FUNCTION (PATCH) LENGTH SCALE r. Unscaled A-weighted least squares,
varying patch radius r in grid cells (rc = r/dx):

    rc= 3 (12mm): mu_acc 0.134   mu_tw 0.166
    rc= 4 (16mm): mu_acc 0.213   mu_tw 0.247
    rc= 6 (24mm): mu_acc 0.306   mu_tw 0.349
    rc= 8 (32mm): mu_acc 0.352   mu_tw 0.430
    rc=12 (48mm): mu_acc 0.456   mu_tw 0.466
    rc=16 (64mm): mu_acc 0.613   mu_tw 0.637

mu rises monotonically with r/dx and crosses the truth (0.38) near rc ~ 9.
Both the acceleration form (b uses trajectory a) and the time-weak form (b
uses v and dw/dt, no a) show the SAME scale dependence.

Ruled out:
- Acceleration error: the time-weak form avoids a entirely and shows the same
  bias, so a is not the cause. (Magnitudes are sane: |a| median 1.3, p99 8.6
  m/s^2; flow active t in [0.02, 0.45] s.)
- Particle-quadrature density: downsampling particles 12500 -> 3149 (volume
  rescaled to preserve mass) barely moves mu (0.352 -> 0.343 at rc=8). The
  bias is invariant to particle count.
- Gravity dominance / cancellation of large terms: INT rho g . w is itself
  near zero for divergence-free compact w (the integral of a curl vanishes
  for uniform density), so the balance is not a difference of large gravity
  and inertia terms.

Conclusion: the continuum weak form INT tau:Dw = INT rho(g-a).w, sampled at
particle locations with V_p quadrature and INDEPENDENT analytic divergence-free
bump test functions, does NOT reproduce the simulator's discrete momentum
balance, and the recovered mu depends on the bump scale with no clean limit.

Scale-study evidence (ident/gates/scale_study.py, comparing dx=4mm and
dx=2.67mm): plotted against PHYSICAL patch radius r the two grids nearly
collapse (r=32mm: 0.352 vs 0.351; the curves separate only mildly at small r),
so the bump-form bias is governed primarily by the patch size relative to the
FLOW length scale, not by r/dx. Combined with the particle-count invariance
(downsampling), this identifies the bump-form bias as a divergence-free
quadrature/cancellation effect of evaluating INT p(2D/|gd|):Dw at scattered
particles (the integrand is a near-curl with heavy cancellation, factor ~r/H,
that particle quadrature resolves poorly), NOT a pure grid-coupling artifact.

Either way the cure is the same and is EUCLID's: use test functions in the
simulator's own discrete (grid B-spline) space with its grad-N operator, i.e.
the Bubnov-Galerkin choice, which reproduces the exact MPM nodal force balance.
That form recovers mu=0.384 (1%) at dx=4mm and 0.379 (0.27%) at dx=2.67mm
(converging under grid refinement), versus the bump form's 0.13-0.46 spread.

## What the literature says (EUCLID, via the lit-review workflow)

EUCLID (Flaschel/Kumar/De Lorenzis; NN-EUCLID Thakolkaran et al.) is
Bubnov-Galerkin: the weight functions ARE the FEM nodal shape functions used
to interpolate the displacement field (Eq. 18 expands v in the same N^a basis
as u). Substituting and using arbitrariness of the nodal test values collapses
the weak form to a per-node internal-minus-external force balance
    f_i^a = INT P_ij grad_j N^a dV  -  (boundary tractions)  = 0  at free DoFs
which is literally the assembled FEM internal-force vector. The residual
operator and the stress-divergence operator are the SAME discrete operator
(same N^a, same grad N^a, same mesh quadrature), which is exactly why EUCLID
is unbiased at finite mesh size. In sparse-regression EUCLID this residual is
linear in theta, giving A theta = b just like ours.

Two caveats EUCLID makes that matter for us:
- EUCLID does NOT reuse the forward (data-generation) mesh. It denoises the
  field (kernel ridge regression) and projects onto a COARSER identification
  mesh, then enforces test = trial on THAT mesh. So the requirement is
  test-space = the IDENTIFICATION trial-space, on a reconstructed field, not
  necessarily the forward solver's grid. This is the bridge to real data
  (Phase 2): reconstruct kinematics on a chosen basis and use that basis for
  the test functions.
- EUCLID reports no residual-vs-mesh convergence study; its unbiasedness is by
  construction (Bubnov-Galerkin), not by refinement.

## RESOLVED: grid-consistent assembly passes Step 4

Implemented ident/weakform/grid_assembly.py (assemble_grid_consistent): test
functions are the MPM quadratic B-spline GRID basis, reconstructed exactly
from particle positions and dx, restricted to plane strain by summing over
the out-of-plane node layer; the internal force uses the same grad N_i
operator as P2G. Rows are emitted per frame per grid node per direction, for
nodes whose contributing particle mass is >= flow_frac_min flowing-at-yield.

Result on column_constant_a2 (n_grid=80, true pressure, ConstantDict),
mu_true = 0.38:
    mu_hat = 0.3838  (relative error 1.0 percent)  -> Step 4 PASSES (<2%)
    stride-robust (0.3838 at stride 4 vs 0.3845 at stride 8)
    threshold sensitivity: 0.370 / 0.384 / 0.393 at flow_frac_min 0.85/0.90/0.95
Compare the analytic-bump assembly on the same dump: mu spanned 0.13 to 0.61
depending on patch radius. The EUCLID Bubnov-Galerkin diagnosis was correct.

Residual notes: the per-row relative residual is ~0.38 (the particle-vs-grid
acceleration gap shows up as per-node scatter), but it averages out over
~17000 rows so the point estimate is accurate; the analytic posterior band is
correspondingly optimistic (assumes iid row noise). The ~few-percent
threshold sensitivity is the same particle-vs-grid-acceleration gap; dumping
the GRID acceleration / internal force from warp-mpm would make the residual
machine-consistent and remove it (the exact-consistency path, future work).

## Fix plan (status)

Route 2 (grid-consistent assembly) DONE and passing. Route 1 (grid-refinement
convergence study) in progress as confirmation (n_grid=120 running). The
real-data path still needs route-1 logic on reconstructed fields.

Two complementary routes; pursue both.

1. Convergence study (validates the continuum method is correct).
   Show mu_hat -> mu_true as dx -> 0 at a fixed physical patch size, the G1
   analogue of G0's quadrature refinement. Comparing n_grid=80 (dx=4mm) with
   n_grid=120 (dx=2.67mm) and finer. ident/gates/scale_study.py produces the
   mu-vs-r and mu-vs-(r/dx) curves per resolution. If finer grid shifts the
   mu(physical r) curve toward 0.38, the bias is confirmed discretization and
   the method is correct in the limit.

2. Grid-consistent (Bubnov-Galerkin) assembly (the production fix; gets to
   within tolerance at finite resolution).
   Build the test functions from the MPM grid B-spline basis N_i and assemble
   the internal force with the same grad N_i operator MPM uses. Per in-plane
   grid node column (summing the 3D basis over the out-of-plane y nodes, which
   sends grad_y -> 0 and keeps sum_y N = 1, consistent with plane strain),
   with sigma = -p I + sum_k theta_k p phi_k (2D/|gd|):

     A[(i,d),k] = sum_p V_p phi_k(I_p) p_p (2 D/|gd|)_p[d,j] grad_j N_i(x_p)
     b[(i,d)]   = sum_p m_p (g_d - a_d) N_i(x_p) + sum_p V_p p_p grad_d N_i(x_p)

   The pressure term does NOT drop (these test functions are not divergence
   free) and moves to b as data. Exact recovery additionally needs the GRID
   acceleration a_i rather than the particle trajectory a; if grid-basis test
   functions with particle-interpolated a do not reach 2 percent, dump grid
   acceleration / internal force from warp-mpm (a schema addition) to close
   the last gap. NOTE this route is oracle-specific (uses the grid); the
   real-data path (Phase 2) uses route 1's logic on reconstructed fields
   (streamfunction B-splines, MATH_REFERENCE 6.5) with test = that basis.

## Two weak forms, and the pressure-closure story (G1P)

A subtlety the grid-consistent fix exposed. There are two test-function
families with opposite strengths:

- Divergence-free (the bump w = curl(eta) functions): the pressure term
  -INT p div w vanishes identically, so recovery is INSENSITIVE to the
  pressure GRADIENT and the pressure enters only multiplicatively through
  tau = mu p (2D/|gd|). This is what makes the closure bias clean and
  interpretable (MATH_REFERENCE 5: mu_hat/mu = row-weighted mean p_true/p_est).
  Cost: on raw particle data it carries the grid-coupling bias (scale
  dependent), because bump functions are not in the simulator's discrete
  space.
- Grid-consistent (the MPM B-spline nodal functions): unbiased on true
  pressure (Step 4 = 1%), but NOT divergence free, so the pressure gradient
  -INT p div N is a first-class term. Feeding a closure pressure whose
  GRADIENT differs from truth corrupts recovery catastrophically.

Measured on column_constant_a2 (mu_true = 0.38):
- Grid-consistent + true pressure: mu_hat = 0.384.
- Grid-consistent + P1 / P0: mu_hat = 0.071 / 0.076 (the closure pressure
  gradient is wrong, the balance breaks). This is the WRONG tool for closures.
- Divergence-free multiplicative bias prediction (mu_hat/mu = <p_true/p_est>
  over flowing rows): P0 -> 1.78 (mu ~ 0.68), P1 -> 1.64 (mu ~ 0.62). In the
  flowing surge the TRUE pressure exceeds the hydrostatic / depth-integrated
  closures by ~1.7x (dynamic and collisional stress beyond a settling-column
  estimate), so closures UNDER-predict p and the divergence-free recovery
  OVER-estimates mu by ~65 to 78 percent.

Implication: with the divergence-free form (the only option for real data,
which has no true pressure gradient), the standard P0/P1 closures are poor in
the dynamic flowing region. This is exactly the bias the region-split
same-I consistency diagnostic (MATH_REFERENCE 8.4) is meant to catch, and it
motivates the true-pressure oracle as the accuracy upper bound. The clean
four-curve figure with a SINGLE consistent method needs the smooth-field
(streamfunction B-spline, MATH_REFERENCE 6.5) reconstruction that makes the
divergence-free weak form both pressure-insensitive AND grid-consistent; that
is the Phase-2 method and a worthwhile Phase-1 enhancement.

## Practical solver notes found along the way

- Do NOT use the inertia-gravity row scaling (.scaled()) for the constant-mu
  solve: it up-weights low-signal rows and worsened mu (0.039 scaled vs 0.21
  unscaled at rc=4). The raw A-weighted least squares (A.b)/(A.A) is better
  because it up-weights high-stress-power rows. Revisit scaling for Mode P.
- The validity-mask clearances must be floored at a couple of grid cells
  (resolution_dx), not pure grain diameters, because d (1mm) is sub-grid
  (dx=4mm) here; otherwise near-surface low-pressure particles leak in and
  blow up the realized I band (saw I_max ~ 26000 before the fix).
