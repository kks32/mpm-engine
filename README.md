# warpmpm

A modular Warp MPM engine for robot manipulation of deformable and granular media:
dough pressing and shaping, glass-to-glass pouring, granular flows. It couples to
MuJoCo today and is built with NVIDIA Isaac Lab in mind (Warp to PyTorch is zero-copy
on CUDA).

The core is the validated warp-mpm fork (bit-for-bit the kernels behind the TrackEUCLID
results) wrapped behind a small typed `Solver`, with a heavily optimized explicit MLS/APIC
step: a claymore-style fused G2P2G particle pass (the default), AABB-restricted grid
launches, mass-gated colliders, an active-block sparse mode, CUDA graph capture, and
per-phase profiling. Every optimization is equivalence-tested; the default pipeline is
bitwise-identical to the original solver.

## Design

```
robot sim (MuJoCo / Isaac) ── link poses, velocities ──▶ warpmpm
                            ◀── per-link 6D wrench, contact obs ──
```

The MPM owns the material; the robot sim owns the robot; the coupling exchanges only
compact wrenches (never particles). 13 materials (Newtonian/Bingham/Herschel-Bulkley
dough, mu(I) sand incl. tabulated laws, elastic, von Mises, weakly compressible fluid).

What exists:

- `core/solver.py`: device-auto `Solver` (cuda:0 when present, cpu otherwise) with
  load / material / collider / step / export, `GridConfig`, and the pipeline flags
  described below. Kinematic box collider (`add_box` / `set_box`) as the robot
  end-effector proxy.
- `kernels/`: the vendored fork. Explicit MLS/APIC transfers, quadratic B-splines,
  fused G2P2G, revolved-SDF glasses with Newton-exact wrench accumulators, mesh-SDF
  colliders, restricted launches, active-block compute, CUDA graphs.
- `materials/`: composable presets (`newtonian`, `granular`, `elastic`, `vonmises`,
  `dough`, tabulated laws).
- `geometry/`: watertight mesh to SDF (winding-number sign), cached; cup meshes.
- `colliders/glass.py`: analytic revolved glass profile, cavity/solid masks, leak
  projection.
- `coupling/`: grid-impulse (Newton-exact) and stress-integral wrench readouts;
  two-way force feedback (the arm halts on the dough).
- `adapters/mujoco_adapter.py`: Franka arm, scripted press and pour kinematics,
  composite offscreen rendering of arm + glasses + particles.
- `src/ident/`, `src/common/`: the warp-free EUCLID identification stack (weak-form
  constitutive recovery); an import-boundary test keeps it free of warp/torch.
- `examples/`: one curated demo per capability (pour, force-feedback press, gripper
  shaping, squeeze identification, wrist-FT cross-check, shear-cell rheology, surface
  render), sharing helpers through `examples/common.py`; `examples/recovery/` holds
  the constitutive-recovery examples. See `examples/README.md`.
- `experiments/`: the paper studies and figure scripts, kept runnable for
  reproducibility and mapped to the results they produce in `experiments/README.md`.

## Performance

Measured on the 192^3 honey pour (340k particles, 432 substeps/frame, GH200):

| stage | sim ms/frame |
| --- | --- |
| before this optimization pass | 782 |
| + mass-gated colliders, shelled walls | 319 |
| + fused G2P2G (now the default) | 278 |

CPU (Apple Silicon, collider-heavy fluid scene): 3.0 to 3.2x over the same baseline.
All of it with bitwise-identical trajectories, deformation gradients, and stress against
the pre-optimization engine, and identical identification results downstream.

Solver flags (all default-off except `fused`):

- `fused=True` (default): one fused g2p+stress+p2g particle pass per interior substep
  instead of three passes. Bitwise-equal; falls back per tick for rigid bodies,
  particle modifiers, or sparse mode. `fused=False` restores the three-pass path
  (the only one with CUDA graph capture).
- `sparse=True`: active-block grid sweeps for scenes whose occupancy is not box-shaped.
- `sort_interval=K`: claymore-style block sort of particle storage every K ticks
  (GPU locality; changes particle index identity, so keep 0 for index-paired dumps).
- `profile=True`: per-phase substep timing table via `Solver.profile_report()`.
- `guard_interval=K`: amortize the grid-edge guard readback.

## Quickstart

```bash
uv pip install -e ".[dev,mujoco,render]"
pytest                              # equivalence, conservation, coupling, identification
python examples/pour_franka.py --fast --record   # pour + render in a separate GL process
python examples/pour_franka.py --skip-video --frames 60 --profile   # where time goes
ruff check . && ty check
```

On clusters where GL and CUDA must not share a process (GH200 nodes), use `--record`:
the simulation dumps per-frame render state and re-invokes itself as a GL-only
subprocess (`--render-only`), with frames split across parallel workers.

## Roadmap

1. DONE: primitive-SDF and mesh-SDF moving colliders, `set_robot_kinematics`.
2. DONE: per-link wrench readout (Newton-exact grid impulse) + cross-validation.
3. DONE: MuJoCo arm, dough manipulation, pouring, composite render.
4. DONE: restricted launches, sparse active-block grid, CUDA graphs, fused G2P2G,
   block sort, mass-gated colliders, profiling.
5. NEXT: implicit quasi-static solver (press/squeeze scenes; GeoWarp-style Newton-CG
   with SDF Dirichlet contact), then an implicit density-projection liquid solver as a
   separate dynamic path (real incompressibility for pouring). The explicit engine
   stays the reference oracle for both.
6. Learned constitutive residual (trainable seam); terrain on the same core.

## License, provenance, acknowledgments

MIT ([LICENSE](LICENSE)) for the group's code. The vendored core in
`src/warpmpm/kernels/` derives from the upstream UCLA warp-mpm (no license file
upstream); boundary in [AUTHORS.md](AUTHORS.md).

Full provenance with BibTeX in [AUTHORS.md](AUTHORS.md). In short:

- The numerical core began as the UCLA **warp-mpm** of Zeshun Zong and collaborators
  (Chenfanfu Jiang's group; cite Zong et al. SIGGRAPH Asia 2023 and PhysGaussian),
  extended by our group (UT Austin) with the robotics coupling, wrench readouts,
  granular/viscoplastic materials, and the optimization stack.
- The transfer-pipeline optimizations (fused G2P2G, per-block particle binning and
  sorting) port the architecture of **claymore** (MIT license), Wang, Qiu, et al.,
  "A Massively Parallel and Scalable Multi-GPU Material Point Method", ACM TOG 39(4),
  2020, read from Justin Bonus's fork (ClaymoreUW). Design ported and reimplemented in
  Warp; no source code copied. The shared-memory arena kernel and AoSoA bins remain
  claymore ideas earmarked for the CUDA-native follow-up.
- The planned implicit quasi-static solver references **GeoWarp** (license check before
  any borrowing; reimplement-with-reference and cite).
