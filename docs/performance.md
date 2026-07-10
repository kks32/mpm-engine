# Performance notes

How the explicit solver was profiled and optimized, what each change bought, and the
equivalence guarantee behind each one. Provenance and BibTeX for borrowed designs are
in AUTHORS.md. Benchmarks quoted below come from two machines: an Apple M3 Max (warp
CPU, single threaded) and a TACC Vista GH200 node (driver 590.48.01, July 2026). The
GPU workload is the 192^3 honey pour of examples/pour_franka.py: 340k particles, 432
substeps per frame.

## Measured results

192^3 pour, simulation time per frame on the GH200:

| configuration | ms/frame |
| --- | --- |
| before this work | 782 |
| mass-gated colliders, shelled walls, restricted launches | 319 |
| plus the fused particle pass | 278 |

CPU results depend on the regime. Collider-heavy fluid scenes gain 3.0 to 3.2x from
the grid-side work; a particle-bound dough press gains almost nothing on CPU because
warp CPU kernels are single threaded and the particle arithmetic is the whole budget.

Every optimization below ships with an equivalence test. The default pipeline
produces bitwise-identical particle trajectories, deformation gradients, and stress
against the pre-optimization solver, and identical identification results downstream
(tests/test_fused_pipeline.py, tests/test_restricted_launch.py, tests/test_pour.py).

## Where the time goes

Facts that shaped the work, measured before optimizing:

1. Warp CPU kernels are single threaded regardless of core count (98 s real equals
   98 s user on 16 cores). The Mac is a correctness machine; speed lives on the GPU.
2. The explicit step is bound by the acoustic CFL, dt proportional to dx/sqrt(K/rho).
   The bulk modulus is already softened far below real water to keep dt near 1e-4 s.
   Removing this constraint requires an implicit solver, and no kernel tuning changes it.
3. Before this work, every substep swept the full dense grid several times (zero,
   normalize plus gravity, then one full-grid launch per collider), while occupied
   nodes are typically 2 to 5 percent of the grid.
4. At 192^3 the first GPU profile put the collider kernels at 70 percent of the
   substep, ahead of P2G. Profiles beat intuition; run `--profile` before tuning.

## What landed

### AABB-restricted grid launches

Grid kernels take a `lo` offset and launch over a bounding box instead of the full
grid. Colliders use per-collider boxes (a box collider uses its extent, an SDF
collider the world box of its grid corners under the live pose, an axis-aligned
plane a thin slab); zero and normalize use the particle bounding box, padded by
particle speed, and zeroing runs over the union with the previous box so departing
nodes are cleared exactly once. Solver.step raises if particles come within two
cells of the grid edge, which also closes the out-of-bounds P2G write a review had
flagged. Bench on a 72^3 plane-plus-two-SDF scene: 15.4 to 4.5 ms per substep.
Restricted and full launches produce bitwise-identical positions.

### Mass-gated colliders and shelled walls

Every collider kernel returns immediately on nodes with zero mass. This is exact,
because a massless node lies outside every particle stencil: G2P never reads its
velocity and it contributes nothing to any wrench. On the GH200 this cut collider
cost at 192^3 from 1.25 to 0.24 ms per substep and, combined with registering the
domain walls as six thin face shells instead of one full-grid launch, brought the
pour from 782 to 344 ms per frame. Colliders whose `modify_bc` is None cannot move
between substeps, so their launch boxes are cached until a pose setter invalidates
them; this is a structural property of the BC list, valid for any scene.

### Fused particle pass (G2P2G)

The default pipeline runs one particle kernel per interior substep, which gathers
from the grid, advects, return-maps stress, and scatters to the next grid state, in
place of the previous three passes. Over a tick of S substeps that is S+1 particle
passes instead of 3S. The port follows claymore's g2p2g design (Wang et al., ACM TOG
39(4), 2020; MIT license; see AUTHORS.md), reimplemented in warp on the fork's
existing grid double buffer: the fused kernel reads grid_v_out (state n) and
scatters into grid_v_in and grid_m (state n+1), which are disjoint arrays.

The fused output stays bitwise-identical to the split pipeline because the zeroing is
split around the fused pass: grid_m and grid_v_in are cleared before it, grid_v_out
after it.
Normalize writes grid_v_out only where mass exceeds 1e-15, so sub-threshold stencil
nodes must read an explicitly zeroed grid_v_out on the next gather; clearing it
early would instead erase the state the fused kernel still needs. The fused path
gains 13 percent on the GPU pour (319 to 278 ms per frame) and about 7 percent on
CPU. Solver falls back to the split pipeline per tick when rigid bodies, particle
modifiers, or sparse mode are active; `fused=False` restores it globally and is the
only path with CUDA graph capture.

### CUDA graph capture

The fixed-shape substep segments (zero, stress, P2G, normalize; then G2P) are
captured and replayed; collider launches and host-side pose integration stay live,
because collider structs are marshalled by value at launch and a captured graph
would freeze the tool pose. Graphs win 1.73x on small scenes where launch overhead
dominates and lose slightly at 192^3, where capture bakes full-grid sweep dims while
live mode gets the AABB restriction. One operational contract came out of this:
state imports must copy into the existing warp arrays in place, since replacing an
array invalidates the pointers baked into a captured graph (observed as CUDA error
700). A pointer-based graph signature forces recapture as a backstop and
WARPMPM_NO_CUDA_GRAPH=1 disables capture in the field.

### Active-block sparse compute

`sparse=True` marks the 4^3 grid blocks touched by particle stencils, dilates by one
block, and runs the grid sweeps over the compacted active list. Storage stays dense.
This helps when occupancy is not box-shaped (diagonal or scattered scenes); on the
pour, whose occupancy is a compact box, it matches the AABB restriction exactly.

### Periodic particle block sort

`sort_interval=K` sorts particle storage by 4^3-block key every K ticks and permutes
all particle arrays in place, keeping array pointers stable. On the GPU pour it did
not pay: the meshgrid initialization is already block-coherent, and without a
shared-memory scatter kernel the sort only adds overhead (320 vs 278 ms per frame at
K=8). It stays off by default, also because sorting changes particle index identity
and would break index-paired trajectory dumps. It remains the prerequisite for a
shared-memory P2G, where bucket order is required rather than merely helpful.

## Claymore: what was ported and what was left

The reference is claymore (github.com/penn-graphics-research/claymore, MIT), read
via Justin Bonus's fork (ClaymoreUW), against the paper Wang, Qiu et al., "A
Massively Parallel and Scalable Multi-GPU Material Point Method", ACM TOG 39(4),
2020. Design ported and reimplemented in warp; no source code copied. Their
architecture reduces to five ideas:

1. Fused g2p2g with double-buffered grids. Ported; described above.
2. Block-centric launches with shared-memory arenas: one CUDA block per active 4^3
   grid block, cooperative load of the block-plus-halo arena, block-local atomics,
   one global writeback. This is what removes global atomic contention in P2G, and
   it is where their reported 3 to 10x P2G speedups come from. Not portable to warp
   kernels as written (needs wp.tile or native CUDA); deferred.
3. Per-block particle buckets rebuilt every step. Ported in relaxed form as the
   periodic block sort; our global-atomic kernels only want locality, so every K
   ticks suffices.
4. AoSoA particle bins (channels interleaved per 32-particle bin). Not implemented:
   after a block sort our SoA arrays are already block-contiguous, which restores
   intra-warp coalescing; the remaining delta only pays together with idea 2.
5. Material-templated kernels to bound register pressure. Our kernels branch on a
   per-particle material id inside one fat kernel; if register pressure ever hurts
   the fused kernel, splitting by material family is the fallback.

Ideas 2 and 4 belong together as one CUDA-native follow-up, worth doing only if a
profile shows P2G dominant after the implicit-solver work below.

## Running on GPU clusters

Two failure modes found on GH200 nodes, both with simple rules:

1. GL and heavy CUDA in one process fault the driver (dmesg Xid 31 graphics MMU
   fault with mjr_readPixels hanging uninterruptibly, or Xid 109 context-switch
   timeouts). Either alone is fine. Run simulations with `--record`, which dumps
   per-frame render state and re-invokes the script as a GL-only subprocess
   (`--render-only`), with frames split across parallel workers. Diagnose apparent
   hangs with `dmesg -T | grep -i xid`, since the Python traceback points at the
   wrong layer.
2. Warp array pointers are part of the engine contract (captured graphs, cached
   views). Anything that replaces a state array instead of copying into it will
   surface as CUDA error 700 at the next graph launch.

`profile=True` (or `pour_franka.py --profile`) prints a per-phase table with time
shares; `Solver.profile_report()` returns it programmatically. The per-phase timers
synchronize the stream, so profiling changes wall time and is off by default.

## Which solver for which scene, and what comes next

Kernel-side tuning at 192^3 is close to exhausted: after fusion the single particle
pass is 55 percent of the substep and every grid phase is below 15 percent. The
remaining large lever is the substep count, which the acoustic CFL fixes. That is an
integrator change, planned in two parts and both keeping the explicit engine as the
reference oracle:

1. Quasi-static implicit solver, for press, squeeze, and shaping scenes where the
   tool moves at cm/s and explicit integration spends its substeps resolving sound
   waves. Newton-CG on grid velocities, matrix-free, with SDF contact as Dirichlet
   constraints and the reaction wrench read from the constraint residual. GeoWarp's
   quasi-static solvers are the reference implementation (check its license before
   borrowing; reimplement and cite). Validation gates: elastic settling and a plate
   press against slow explicit runs, then a dough squeeze whose implicit dump must
   yield the same identified (tau_y, eta) as the explicit dump. Realistic
   expectation on CPU is 5 to 30x wall clock; the qualitative gain is that dt
   decouples from the bulk modulus, so real stiffness becomes runnable.
2. A density-projection implicit dynamic solver for liquids, as a separate path.
   Quasi-statics cannot pour, since a falling stream has no equilibrium. A
   Chorin-style density projection enforces incompressibility with dt at the
   advective scale (about 1e-3 s at our dx), retires the softened EOS, and, done
   against a grid density, restores rest packing after breakup, which is the fix
   for the apparent-volume inflation the pour ledger measures at +22 percent.

A note on scene and solver fit until then: press and squeeze scenes are the
quasi-static target, drop and impact scenes are fine explicitly, and water-scale
pouring visuals are better served by SPH while the projection solver does not exist.
The dense, viscoplastic, and granular scenes that identification uses are exactly
where the explicit engine is already fast and validated.
