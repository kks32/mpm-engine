# Examples

One demo per engine capability. Each runs on CPU at a smoke size with no arguments
(`--device cuda:0` to pin a GPU) and writes to `out/`. Shared helpers live in
`common.py`: the kernel-matched shear-rate measure, the Chamfer metric, the ffmpeg
frame encoder, the Franka descent calibration, and particle-cloud surfacing.

Manipulation and coupling:

- `dough_franka_press.py`: two-way force-feedback press. An admittance law descends the
  gripper until the measured dough reaction reaches a target force; the material decides
  the stopping depth.
- `squeeze_plate_franka.py`: arm-mounted plate squeeze that recovers the dough's
  (tau_y, eta) from the plate reaction force alone, cross-validated against the 2D
  squeeze-flow result.
- `wrist_ft_franka.py`: the MPM grid-impulse reaction fed to a dynamic MuJoCo Franka
  equals the wrist force-torque sensor reading, validating the two-way readout.
- `gripper_shape.py`: press to identify the material, then plan 2-finger grips (CEM over
  grip positions and widths) that sculpt the dough to a target shape. Modes: `demo`,
  `plan`, `video`, `tshape`.

Pouring:

- `pour_franka.py`: A Franka pours honey glass to glass with per-frame
  metrics, wrench readouts on both glasses, a leak audit, and a record/render process
  split for GPU clusters (see the header and docs/performance.md).
- `pour_glass.py`: minimal mesh-to-SDF pour, the API demo for arbitrary watertight mesh
  colliders.

Identification:

- `vonmises_identify.py`: (G, yield) of a von-Mises dough from one squeeze probe, via a
  two-window power balance on the plate force.
- `shear_cell_fe.py`: wide-shear-rate 2D shear cell; recovers the apparent-viscosity
  curve eta_app(gd) on a learned basis and validates it by re-simulating a held-out
  shear speed. Needs `fe-weights/viscous.npz`.

Rendering:

- `dough_surface_render.py`: surface the particle cloud with marching cubes and render
  the dough as a continuous body. Needs the `surface` extra (scipy, scikit-image).
- `splat_sim.py`: Gaussian-splat scene, PhysGaussian style. Drops a splat body, fills its
  interior, then presses it with a scripted box collider; covariances advect and the SH
  rotate with the deformation. `--ply` loads a real checkpoint; `--record-splats` writes
  viewer frames; `--bake OUT.npz` compresses the run into temporal B-splines and
  `--from-bake IN.npz` re-exports viewer frames at any frame count from the bake.
  Needs the `splats` extra (plyfile, scipy).

`recovery/` holds the constitutive-recovery examples (elastic and plastic drops,
sequential identification, sample complexity); see its README. Paper studies and figure
scripts live in `../experiments/`.
