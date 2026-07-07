# Speedup plan: restricted launches, implicit MPM, CUDA path

Status: agreed plan, July 2026. Execution order is Step 1 (restricted launches), Step 2
(quasi-static implicit), Step 3 (CUDA path). Steps 1 and 2 are fully developable and
testable on the Mac (CPU); Step 3 pays off only on a CUDA box, so it goes last.

## Baseline: where the time goes today

Measured and code-audited facts, not guesses:

1. Warp CPU kernels are single threaded (measured: 98 s real vs 98 s user on 16 cores).
   The engine's speed ceiling on the Mac is one core; the real headroom is on GPU.
2. The explicit step is bound by the acoustic CFL, dt proportional to dx / sqrt(K/rho).
   We already trade correctness for speed here: the bulk modulus is softened about 1e4x
   below real water so dt stays near 1e-4 s. Real-K water needs dt near 1e-6 s at our dx,
   which is infeasible explicitly.
3. Every substep sweeps the full dense grid several times: zero, normalize plus gravity,
   then one full-grid launch per collider in the grid_postprocess loop. Occupied nodes are
   typically 2 to 5 percent of the grid, and each collider covers a small box.
4. The fork wraps six phases per substep in ScopedTimer(synchronize=True). Harmless on
   CPU (CPU kernels run synchronously), but on CUDA each one stalls the pipeline.

## Step 1: AABB-restricted grid work (days; expect 2 to 5x on multi-collider CPU scenes)

### 1A. Colliders launch over their bounding boxes

Every grid_postprocess kernel gains a `lo: wp.vec3i` offset input; the node index becomes
`wp.tid() + lo`, and the launch dim becomes the box extent. A parallel list
`_collider_aabb[k]` holds a host callable returning the current (lo, dim) per collider:

- Box collider: point +/- size in grid units, one cell halo. Exact.
- SDF collider: the world box of the 8 SDF-grid corners under the current (center, quat).
  This provably covers every constrained node because the collide kernel already gates on
  the body-frame SDF grid box. The corners are already computed for the tunneling guard.
- Axis-aligned plane (the floor): the affected half-space intersected with the grid is a
  thin slab of layers along the normal axis; restrict to that slab. Planes that are not
  axis aligned keep the full-grid fallback.
- Point cloud: static bounding box of the occupied cells, computed once at add time.
- Bounding-box BC and legacy modifiers: full-grid fallback (lo = 0), unchanged behavior.

Launches with an empty box are skipped entirely (tool outside the domain).

### 1B. Zero and normalize over the live particle box, plus the missing bounds guard

Once per control tick (inside Solver.step, where host reads already happen), read the
particle AABB and max speed, pad by max_speed * dt * substeps * 1.5 plus 3 dx, and use
that box for the tick's zero and normalize launches. Add an atomic violation counter to
P2G: any particle whose stencil would leave the padded box increments it, checked once
per tick, warn then raise. This one mechanism delivers both the speedup and the
out-of-bounds write guard the code review flagged (P2G currently writes through
unclamped grid indices; a particle near the domain edge corrupts memory silently).

### Test gates

1. Equivalence test: floor + box + cup-SDF scene, 50 substeps, restricted vs full
   launches; particle positions bitwise equal (per-node BC writes are independent),
   force accumulators allclose (atomic reorder only).
2. All existing tests stay green.
3. Bounds-guard test: a particle driven at the domain edge trips the counter.
4. bench_step before and after, one and three colliders, report the multiplier.

## Step 2: quasi-static implicit MPM with SDF Dirichlet (2 to 3 weeks, phased)

A new solver path (src/warpmpm/implicit/quasistatic.py); the explicit path is untouched.
Target scenes: press, squeeze, gripper shaping, where the tool moves at cm/s and explicit
integration burns about 1e4 substeps per control tick resolving sound waves. Reference
implementation: GeoWarp (in-repo) quasi_static_solver_2d/3d, which already has active-DOF
flags, Dirichlet via boundary-flag arrays with diagonal conditioning, and warp kernels.
Check its LICENSE before borrowing; default is reimplement-with-reference and cite in
AUTHORS.md.

### Phases

- 2.0 Reference audit (half day): GeoWarp license, solver structure, BC mechanism.
- 2.1 Newton core (3 to 4 days): DOFs are velocities on active nodes (mass > eps);
  residual r_i = sum_p V_p sigma_p(F_p(du)) grad N_i(x_p) - f_grav,i with
  F_p(du) = (I + sum_i du_i outer grad N_i) F_p^n (updated Lagrangian, existing B-spline
  transfers reused). Matrix-free Newton-CG with a finite-difference Hessian-vector
  product, Jacobi preconditioner, line search, float64 solve arrays on CPU. Materials in
  v1: FCR elastic and von-Mises dough; viscoplastic laws enter as backward Euler in the
  increment (rate = du/dt), using the regularized smooth forms we already trust.
- 2.2 SDF contact and wrench (2 days): the same SDF query marks constrained nodes at the
  tool's end-of-increment pose and pins their velocity to the tool surface velocity;
  flagged DOFs are projected out of CG (textbook Dirichlet, GeoWarp's mechanism). The
  reaction wrench is the residual at constrained nodes (the Lagrange multiplier), summed
  as force and moment: still Newton-exact, no accumulator needed. Separable contact is a
  v2 stretch goal via an active set (release nodes whose normal reaction turns tensile,
  re-solve). The tunneling guard already checks sweep-per-step against the contact band
  and scales correctly to large dt.
- 2.3 Validation gates (3 days):
  - V1 elastic block settles under gravity, matches explicit steady state.
  - V2 plate press on elastic block, force-displacement matches a slow explicit run.
  - V3 (the money test) dough squeeze: force curve and final shape match explicit, and
    squeeze_force_identify on the implicit dump recovers the same (tau_y, eta) as the
    explicit dump. Cross-integrator agreement doubles as a consistency slice for the
    bias-aware certification story.
  - V4 SDF tool press (the cup as a press): constraint correctness plus wrench.
  - V5 wall-clock benchmark. Honest expectation: the step-count win is 100 to 1000x but
    Newton times CG eats part of it; realistic 5 to 30x wall clock on CPU in v1, more
    with exact tangents and preconditioning. The qualitative unlock matters more: dt
    decoupled from the bulk modulus makes real K runnable, retiring the soft-EOS
    compromises.
- 2.4 Wiring (1 day): same coupling contract (set_sdf_pose, wrench readout), dumps in the
  existing schema so identification code consumes them unchanged.

### Risks

Near-incompressible conditioning at large K (mitigate: Jacobi, line search, incremental
loading); FD-JVP noise in float32 (float64 solve arrays on CPU); yield-surface corners
(regularized forms); GeoWarp license (checked first).

## Step 3: CUDA path (days, on the GPU box, after Steps 1 and 2 are green)

### Why removing the per-phase syncs is safe

There are two different things called sync. Correctness ordering between kernels comes
from the stream: kernels on one stream execute in launch order and see each other's
writes, and every inter-phase dependency in p2g2p is a device array on that stream. The
six ScopedTimer(synchronize=True) blocks exist only so the per-phase stopwatch measures
execution rather than launch: they are instrumentation. Host reads (exports, force
readouts, .numpy()) synchronize implicitly inside warp's device-to-host copy. The one
host-in-the-loop piece, modify_bc, mutates host struct fields that are marshalled by
value at each launch, so there is no device race. Removal is therefore bitwise-safe;
a profile=True flag restores the timers when a breakdown is wanted. On CPU this change
is a no-op either way, which is why Steps 1 and 2 are testable on the Mac first.

### Work items

1. DONE (with Step 1): timers opt-in via `sim.profile = True`, default off. All eight
   per-phase syncs removed from the default path; profile=True restores exactly the old
   behavior and fills time_profile. Verified no-op on CPU (full suite bitwise).
2. CUDA graph capture of the substep sequence (wp.ScopedCapture). Two design constraints
   discovered while building Step 1, both of which break a naive whole-substep capture:
   - modify_bc pose integration runs on the HOST between launches, and collider structs
     are marshalled by value at launch; a captured graph bakes those values, so a
     replayed graph would never update the tool pose. Either move pose integration into
     a device kernel (pose stored in a wp.array the graph reads), or keep the BC segment
     outside the capture.
   - Step 1's restricted launch dims are baked into a graph. Options: capture only the
     fixed-shape inner phases (zero, p2g, normalize, g2p) and leave BC launches live;
     or re-capture when a box changes by more than the halo; or disable restriction
     inside the captured region on GPU, where the dense sweep is cheap anyway.
   The pragmatic v1 on the GPU box: capture the inner phases, keep BC + modify live.
3. bench_step before and after on the GPU box; gate: existing suite green on CPU and GPU.

## Regime coverage: which solver for which scene

The implicit umbrella is three related solvers sharing grid DOFs, SDF-as-Dirichlet
contact, and wrench-as-constraint-residual. They are not interchangeable:

| Scene | Right solver | Why |
| --- | --- | --- |
| Press, squeeze, shaping | Quasi-static Newton (Step 2) | Inertia negligible; 1 solve per load increment |
| Drop, bounce, impact | Explicit (fine today) or implicit dynamic | Motion timescale must be resolved anyway; implicit buys 10 to 50x by removing acoustic stiffness |
| Pouring, sloshing, liquids | Implicit dynamic with pressure (density) projection | See below |

Pouring specifically: quasi-static CANNOT do it, a falling stream has no equilibrium to
solve for. The correct implicit treatment for liquids is a pressure or density projection
(Chorin-style splitting; implicit density projection in the MPM literature), which
enforces incompressibility exactly with dt set by advection (particles crossing about one
cell per step, dt near 1e-3 s at our dx) rather than by sound speed. Two extra benefits:
real incompressibility without the softened EOS, and, if done as a density projection
against a state-based (grid) density, it restores rest packing after breakup, which is
the fix for the apparent-volume inflation (+22 percent) measured in the pour ledger.
That projection solver is a Step 4 candidate after Step 2 lands, since it reuses the same
Poisson/CG machinery. Until then, SPH (Genesis) remains the pragmatic choice for
water-scale pouring visuals; the warp engine remains the right tool for the dense,
viscoplastic, and granular scenes that identification actually uses.

## Milestones

1. Step 1 landed: equivalence + bounds-guard tests green, bench multiplier recorded.
2. Step 2 V1-V5 green: implicit squeeze identification matches explicit.
3. Step 3 on GPU: suite green both devices, bench recorded.
4. (Candidate Step 4) density-projection liquid solver: pour ledger apparent-volume
   error under 2 percent at the 97k-particle benchmark.
