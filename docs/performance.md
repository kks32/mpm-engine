# Performance notes

This file records the explicit-solver profiles, optimization measurements, and
regression tests. Provenance and BibTeX for borrowed designs are in AUTHORS.md. The
benchmarks use an Apple M3 Max with single-threaded Warp CPU and a TACC Vista GH200
node with driver 590.48.01 in July 2026. The GPU workload is the 192^3 honey pour from
`examples/pour_franka.py`, with 340k particles and 432 substeps per frame.

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

The default pipeline produces bitwise-identical particle trajectories, deformation
gradients, and stress relative to the pre-optimization solver. Downstream
identification results are also unchanged. The regression coverage is in
`tests/test_fused_pipeline.py`, `tests/test_restricted_launch.py`, and
`tests/test_pour.py`.

## Where the time goes

The initial profiles showed:

1. Warp CPU kernels are single threaded regardless of core count: a 16-core run took
   98 seconds of both real and user time. Additional CPU cores did not reduce runtime.
2. The explicit step is bound by the acoustic CFL, dt proportional to dx/sqrt(K/rho).
   The bulk modulus is already softened far below real water to keep dt near 1e-4 s.
   Removing this constraint requires an implicit solver, and no kernel tuning changes it.
3. Before this work, every substep swept the full dense grid several times (zero,
   normalize plus gravity, then one full-grid launch per collider), while occupied
   nodes are typically 2 to 5 percent of the grid.
4. At 192^3 the first GPU profile put the collider kernels at 70 percent of the
   substep, ahead of P2G. Use `--profile` before tuning a different scene.

## Implemented changes

### AABB-restricted grid launches

Grid kernels take a `lo` offset and launch over a bounding box instead of the full
grid. Colliders use per-collider boxes (a box collider uses its extent, an SDF
collider the world box of its grid corners under the live pose, an axis-aligned
plane a thin slab); zero and normalize use the particle bounding box, padded by
particle speed, and zeroing runs over the union with the previous box so departing
nodes are cleared exactly once. Solver.step raises if particles come within two
cells of the grid edge, which also closes the out-of-bounds P2G write a review had
flagged. A 72^3 plane-plus-two-SDF scene improved from 15.4 to 4.5 ms per substep.
`tests/test_restricted_launch.py` covers the restricted and full launch paths.

### Mass-gated colliders and shelled walls

Every collider kernel returns immediately on nodes with zero mass. Skipping such a
node does not change the result: it lies outside every particle stencil, G2P never
reads its velocity, and it contributes nothing to a wrench. On the GH200 this reduced
collider cost at 192^3 from 1.25 to 0.24 ms per substep and, combined with registering the
domain walls as six thin face shells instead of one full-grid launch, brought the
pour from 782 to 344 ms per frame. Colliders whose `modify_bc` is None cannot move
between substeps, so their launch boxes are cached until a pose setter invalidates
them; this is a structural property of the BC list, valid for any scene.

### CPIC thin-boundary colliders (CDF)

CPIC colliders (`add_cdf_collider`, colored distance fields per Hu et al. 2018, see
AUTHORS.md) sever particle-node transfers across an open oriented sheet, so a wall
holds at any thickness: the zero-thickness dam test passes with 0 crossings under a
1 m/s impact, where an SDF needs about two cells of solid. That is their regime.
For bulky tools the SDF stays the right collider, and the pour A/B quantifies why.
On the default glass (20 mm wall, 2.7 cells at 96^3) with the CPIC sheet on the
cavity surface (`glass_cavity_profile`), the 120-frame 96^3 A/B gives: SDF cup
spill 1.6 percent with 6/18 particles projected at CFL 0.28/0.42; CDF glass spill
4.2/4.0 percent with max 364/495 particles inside the solid at any instant. The
CDF glass is watertight through 39 degrees of tilt (2 spilled particles); the gap
opens as fluid crests the rim, for two reasons inherent to the method. CPIC contact
is soft, so pressed fluid sits a few millimetres inside the sheet band and the
solid-interior audit counts it as embedded even though the sheet holds it, and the
rim carries an untagged exclusion tube of one CDF band (16 mm at 96^3, shrinking
linearly with dx) that makes the spout sloppier than the SDF's fully resolved rim.
At 192^3 on a GH200 the CDF pour transfers cleanly (receiver 40 percent of the
fill, final spill 254 of 340k particles, 0.07 percent, confirming the rim gap
shrinks with dx), but do not read wrenches through it: the receiving SDF glass
works as a scale while the CDF wrench under-reads the transferred weight by an
order of magnitude (contact routes through the incompatible weight fraction;
the readout is documented as approximate).

Cost at 96^3 on CPU with two CDF lanes: the fused pass goes from 25.7 to 37.3 ms
per substep (the in-kernel tag vote and ghost-velocity path, paid whenever CDF
is active) plus host-side stamping launches. The stamp kernels take live poses
by value and stay outside CUDA graph capture, which made them the dominant CDF
cost on GPU (1.3 to 2.2 s per frame at 192^3 on a GH200, against 344 ms for the
SDF pour). Tags are a pure function of pose, so lanes restamp only when their
pose changes, and the zero/stamp/copy is restricted to the dirty lanes' boxes: a
resting collider costs no per-substep host work at all, a static receiver stops
paying for a moving source, and the skip is pinned bitwise against always-stamp
in both pipelines (tests/test_cdf_stamp_skip.py). On CPU this removes 5.0 of the
5.3 ms per substep in static phases (pour settle 227 to 201 s); the GPU number
after the skip is a Vista gate.

### Fused particle pass (G2P2G)

The default pipeline runs one particle kernel per interior substep, which gathers
from the grid, advects, return-maps stress, and scatters to the next grid state, in
place of the previous three passes. Over a tick of S substeps that is S+1 particle
passes instead of 3S. The port follows claymore's g2p2g design (Wang et al., ACM TOG
39(4), 2020; MIT license; see AUTHORS.md), reimplemented in warp on the fork's
existing grid double buffer: the fused kernel reads grid_v_out (state n) and
scatters into grid_v_in and grid_m (state n+1), which are disjoint arrays.

The equivalence guarantee above depends on splitting the zeroing around the fused
pass: grid_m and grid_v_in are cleared before it, and grid_v_out is cleared after it.
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

The split pipeline captures at full grid dims, which is why it loses at 192^3. The
fused pipeline captures differently: its interior segment (split zero, fused pass,
second zero, normalize, damping) is captured over the particle box padded by four
cells and replayed until the live box escapes the pad, so the graph removes the
per-substep launch overhead without baking a full-grid sweep. The padding does not
change the output because nodes outside the box are already zero, normalize skips
massless nodes, and damping scales zeros. The post-fusion profile attributes roughly
14 percent of runtime to host overhead, which this capture targets. The graph-vs-live
test, `test_fused_graph_replay_bitwise_on_cuda`, needs a GPU. It is the Vista validation
gate and remains unverified on-device as of this writing.

### Substep count (CFL)

The substep count per control tick is set by the acoustic CFL. Lowering it reduces
work without changing kernels, while an implicit integrator removes the constraint.
`pour_franka --cfl` exposes the target, which defaults to 0.28. MLS-MPM with
quadratic B-splines usually holds near 0.4 to
0.45. A 96^3 smoke A/B (120 frames): 0.42 ran 144 substeps per frame against 216 at
0.28 (1.5x fewer), stayed stable, and left the ledger essentially unchanged (spill
and air 1.6 percent in both; leak audit 18 vs 6 projected particles of 40k). That run
is short enough that the stream never reaches the receiver, so it measures stability
and substep count, not transfer fidelity. The gate before changing the default is a
192^3 full-transfer run on the GH200.

### Active-block sparse compute

`sparse=True` marks the 4^3 grid blocks touched by particle stencils, dilates by one
block, and runs the grid sweeps over the compacted active list. Storage stays dense.
This helps when occupancy is not box-shaped (diagonal or scattered scenes); on the
pour, whose occupancy is a compact box, it matches the AABB restriction exactly.

### Periodic particle block sort

`sort_interval=K` sorts particle storage by 4^3-block key every K ticks and permutes
all particle arrays in place, keeping array pointers stable. It was slower on the GPU
pour: the meshgrid initialization is already block-coherent, and without a
shared-memory scatter kernel the sort only adds overhead (320 vs 278 ms per frame at
K=8). It stays off by default, also because sorting changes particle index identity
and would break index-paired trajectory dumps. It remains the prerequisite for a
shared-memory P2G, where bucket order is required rather than merely helpful.

## Claymore: what was ported and what was left

The reference is [Claymore](https://github.com/penn-graphics-research/claymore) (MIT),
read via Justin Bonus's ClaymoreUW fork and checked against Wang, Qiu et al., "A
Massively Parallel and Scalable Multi-GPU Material Point Method", ACM TOG 39(4),
2020. The design was reimplemented in Warp without copying source code. The relevant
parts of the architecture are:

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

Two failure modes were observed on GH200 nodes:

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

## Solver selection and planned work

After fusion at 192^3, the single particle pass takes 55 percent of each substep and
every grid phase takes less than 15 percent. Further large reductions require fewer
substeps, which means changing the integrator. Two such paths are planned, with the
explicit engine retained as the reference:

1. Quasi-static implicit solver, for press, squeeze, and shaping scenes where the
   tool moves at cm/s and explicit integration spends its substeps resolving sound
   waves. Newton-CG on grid velocities, matrix-free, with SDF contact as Dirichlet
   constraints and the reaction wrench read from the constraint residual. GeoWarp's
   quasi-static solvers are the reference implementation (check its license before
   borrowing; reimplement and cite). Validation gates: elastic settling and a plate
   press against slow explicit runs, then a dough squeeze whose implicit dump must
   yield the same identified (tau_y, eta) as the explicit dump. The estimated CPU
   speedup is 5 to 30 times. Decoupling dt from the bulk modulus would also permit
   materially realistic stiffness.
2. A density-projection implicit dynamic solver for liquids, as a separate path.
   Quasi-statics cannot pour, since a falling stream has no equilibrium. A
   Chorin-style density projection enforces incompressibility with dt at the
   advective scale (about 1e-3 s at our dx), retires the softened EOS, and, done
   against a grid density, restores rest packing after breakup, which is the fix
   for the apparent-volume inflation the pour ledger measures at +22 percent.

Until then, press and squeeze scenes are candidates for the quasi-static solver, while
drop and impact scenes remain explicit. SPH is a better fit for water-scale pouring
visuals until the projection solver exists. Dense viscoplastic and granular
identification scenes remain on the explicit engine.
