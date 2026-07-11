# warpmpm

A Warp MPM engine for robot manipulation of deformable and granular materials. The
examples cover dough pressing and shaping, glass-to-glass pouring, and granular flow.
The engine currently couples to MuJoCo. Its array interface is also intended for NVIDIA
Isaac Lab; Warp-to-PyTorch transfers are zero-copy on CUDA.

The numerical core is a fork of warp-mpm, whose kernels were used for the TrackEUCLID
results. A typed `Solver` wraps its explicit MLS/APIC step. The default transfer path
uses a fused G2P2G particle pass based on Claymore's design. Other options restrict grid
launches to AABBs, skip empty collider nodes, activate sparse grid blocks, capture CUDA
graphs, and collect per-phase timings. Measurements and regression tests are in
[Performance](#performance) and [docs/performance.md](docs/performance.md).

## Design

```text
robot simulator         coupling adapter             warpmpm
---------------         ----------------             -------
tool pose + velocity ─▶ collider command ──────────▶ particles + grid contact
robot controller     ◀─ contact force + torque ◀──── grid impulse / stress integral
```

The robot simulator advances the robot, while `warpmpm` advances the material. They
exchange link kinematics and reaction wrenches; particle state remains in the MPM
solver. Contact is resolved inside `warpmpm` by collider boundary conditions on the
grid. The coupling adapter converts those contact results to a wrench for the robot
controller. The package contains 13 material models, including Newtonian, Bingham, and
Herschel-Bulkley dough; mu(I) sand with tabulated laws; elastic and von Mises solids;
and a weakly compressible fluid.

Code structure overview:

- `core/solver.py`: device-auto `Solver` (cuda:0 when present, cpu otherwise) with
  load / material / collider / step / export, `GridConfig`, and the pipeline flags
  described below. Kinematic box collider (`add_box` / `set_box`) as the robot
  end-effector proxy.
- `kernels/`: the vendored fork. Explicit MLS/APIC transfers, quadratic B-splines,
  fused G2P2G, revolved-SDF glasses with grid-impulse wrench accumulators, mesh-SDF
  colliders, restricted launches, active-block compute, CUDA graphs.
- `materials/`: composable presets (`newtonian`, `granular`, `elastic`, `vonmises`,
  `dough`, tabulated laws).
- `geometry/`: watertight mesh to SDF (winding-number sign), cached; cup meshes.
- `colliders/glass.py`: analytic revolved glass profile, cavity/solid masks, leak
  projection.
- `coupling/`: contact-wrench readouts based on accumulated grid impulse or a stress
  integral. The grid-impulse readout sums the impulses applied by collider boundary
  conditions at grid nodes. Force-feedback controllers use the quasi-static
  stress-integral signal to stop a press at a specified reaction force.
- `splats/`: Gaussian-splat scenes, PhysGaussian style. PLY load/save, interior fill,
  per-particle covariance advection and polar-rotation SH; see `examples/splat_sim.py`.
- `adapters/mujoco_adapter.py`: Franka arm, scripted press and pour kinematics,
  composite offscreen rendering of arm + glasses + particles.
- `src/ident/`, `src/common/`: the warp-free EUCLID identification stack (weak-form
  constitutive recovery); an import-boundary test keeps it free of warp/torch.
- `examples/`: scripts for pouring, force-feedback pressing, gripper shaping, squeeze
  identification, a wrist-FT cross-check, shear-cell rheology, and surface rendering.
  Shared helpers are in `examples/common.py`, and constitutive-recovery examples are in
  `examples/recovery/`. See `examples/README.md`.
- `experiments/`: the paper studies and figure scripts, kept runnable for
  reproducibility and mapped to the results they produce in `experiments/README.md`.

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


## Performance

Measured on the 192^3 honey pour (340k particles, 432 substeps/frame, GH200):

| stage | sim ms/frame |
| --- | --- |
| before this optimization pass | 782 |
| + mass-gated colliders, shelled walls | 319 |
| + fused G2P2G | 278 |

On Apple Silicon CPU, collider-heavy fluid scenes ran 3.0 to 3.2 times faster than the
same baseline. The equivalence guarantee, test coverage, and CPU caveats are documented
in [docs/performance.md](docs/performance.md).

`fused` defaults on; the other solver flags default off:

- `fused=True`: one fused g2p+stress+p2g particle pass per interior substep instead of
  three passes. It falls back per tick for rigid bodies, particle modifiers, or sparse
  mode. `fused=False` restores the three-pass path, which supports CUDA graph capture.
- `sparse=True`: active-block grid sweeps for scenes whose occupancy is not box-shaped.
- `sort_interval=K`: claymore-style block sort of particle storage every K ticks
  (GPU locality; changes particle index identity, so keep 0 for index-paired dumps).
- `profile=True`: per-phase substep timing table via `Solver.profile_report()`.
- `guard_interval=K`: amortize the grid-edge guard readback.

## License, provenance, acknowledgments

The group's code is MIT-licensed; see [LICENSE](LICENSE). The vendored core in
`src/warpmpm/kernels/` derives from the upstream UCLA warp-mpm, which has no license
file. [AUTHORS.md](AUTHORS.md) records the licensing boundary, full provenance, and
BibTeX entries. A summary follows:

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
