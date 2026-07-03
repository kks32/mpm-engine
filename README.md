# warpmpm

A modular **Warp-MPM** engine for robot manipulation of **deformable and granular media** —
dough first, terrain / rovers next. Built to couple to **MuJoCo now** and **NVIDIA Isaac Lab
later** (Warp ↔ PyTorch is zero-copy on CUDA, so this is the Isaac-native path).

**Today** the core is a **dense, explicit** MLS/APIC step wrapping the validated warp-mpm
fork — bit-for-bit the kernels behind the TrackEUCLID results. A **sparse active-block grid**
and an **implicit** quasi-static solver are the planned fast path (roadmap step 4), not yet
present.

## Design

```
robot sim (MuJoCo / Isaac) ──link poses, velocities──▶ warpmpm
                            ◀──per-link 6D wrench, contact obs──
```
The MPM owns the material; the robot sim owns the robot; the coupling exchanges only
compact wrenches (never particles). Core is the warp-mpm fork (12 materials incl.
Newtonian/Bingham dough, mu(I) sand) wrapped behind a small typed `Solver`.

What exists today:
- `core/solver.py` — CUDA-default `Solver` (`cuda:0`; pass `device="cuda:1"` for the
  second GPU or `device="cpu"` for CPU fallback) (load / material / collider / step / export) +
  `GridConfig`. The kinematic box collider lives here as `Solver.add_box` / `set_box`
  (the robot end-effector proxy); the kinematic 6-DoF glass collider as `Solver.add_cup` /
  `set_cup` / `cup_wrench` (analytic revolved SDF, separable Coulomb contact + anti-tunneling
  core, Newton-exact reaction force AND torque).
- `materials/` — composable presets: `newtonian`, `granular`, `elastic`, `dough` (each
  `.resolve()`s to the fork's (name, params)).
- `colliders/` — host-side collider geometry: `glass.py` (GlassProfile, reference SDF,
  containment/leak masks, cavity fill sampler, leak-projection rescue net).
- `coupling/wrench.py` — `box_contact_wrench`: stress-integral reaction wrench (Newton's
  third law), the cross-validation baseline.
- `adapters/mujoco_adapter.py` — `FrankaArm`: scripted Panda descent + composite render;
  `PandaPour`: the Dogma95-ported pour kinematics (FK bit-identical to the Genesis panda).
- `scenes.py` — `block`, `dough` scene builders.
- `examples/pour_franka.py` — Franka pours MPM honey glass-to-glass (192^3), action +
  geometry identical to the Dogma95 Genesis/SPH pouring study (cross-simulator
  comparable); volume fidelity of the bed is measured, first-order in dx.
- `tests/`, `benchmarks/` — conservation + sanity (incl. cup containment / pour); ms/step.

Planned (stubs today): capsule/sphere SDF colliders and a baked mesh-SDF collider for
non-revolved tools, a unified `coupling.WarpMPMBackend` 6-DoF tool surface, `render/`.
See the Roadmap.

## Quickstart

```bash
uv pip install -e ".[dev,mujoco,render]"
python benchmarks/bench_step.py     # baseline ms/step
pytest                              # conservation + sanity
ruff check . && ty check            # lint + types
```

The Warp MLS-MPM kernels live in-package at `src/warpmpm/kernels/` (solver, constitutive
kernels, structs); everything imports them from `warpmpm.kernels`. See AUTHORS.md for provenance.

## Roadmap

0. Package + explicit dense baseline + benchmark + tests  ← here
1. Primitive-SDF moving colliders + `set_robot_kinematics`
2. Per-link wrench readout (Newton's third law) + squeeze cross-validation
3. MuJoCo arm + dough manipulation demo + render
4. Sparse active-block grid (GPUMPM) + implicit Newton-CG (GeoWarp) — the fast version
5. Learned constitutive residual (trainable seam)
6. Terrain / rover navigation on the same core
