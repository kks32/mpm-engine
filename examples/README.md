# Examples

The scripts run at CPU smoke-test size with no arguments and write to `out/`. Pass
`--device cuda:0` to select a GPU. `common.py` contains the shared shear-rate measure,
Chamfer metric, ffmpeg frame encoder, Franka descent calibration, and particle-cloud
surfacing code.

Manipulation and coupling:

- `dough_franka_press.py`: two-way force-feedback press. An admittance law descends the
  gripper and stops when the measured dough reaction reaches a target force.
- `squeeze_plate_franka.py`: arm-mounted plate squeeze that recovers the dough's
  (tau_y, eta) from the plate reaction force alone, cross-validated against the 2D
  squeeze-flow result.
- `wrist_ft_franka.py`: compares the MPM grid-impulse reaction on a dynamic MuJoCo
  Franka with the wrist force-torque sensor reading.
- `gripper_shape.py`: press to identify the material, then plan 2-finger grips (CEM over
  grip positions and widths) that sculpt the dough to a target shape. Modes: `demo`,
  `plan`, `video`, `tshape`.

Pouring:

- `pour_franka.py`: a Franka pours honey between glasses. It records per-frame metrics,
  wrench readouts for both glasses, and a leak audit. Simulation and rendering can run
  in separate processes on GPU clusters; see the script header and docs/performance.md.
  `--glass cdf` swaps the analytic SDF glasses for CPIC cavity sheets (thin-boundary
  colliders, watertight at any wall thickness); the leak audit compares the two.
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

Floods and rigid bodies:

- `flood_vehicle.py`: a water surge hits a splat-captured vehicle held as one rigid
  body; grid momentum becomes body force and torque, so pushing, floating, and
  overturning come out of the coupling (at 25 cm depth and 3 m/s a 150 kg/m^3 truck
  floats up and rolls onto its side). Records displacement from the spawn center
  and yaw/pitch/roll per frame; `--vehicle` takes any 3DGS PLY or watertight mesh,
  `--depth` and `--velocity` set the surge. The same scene is importable
  (`warpmpm.vehicle.FloodScene`) for parameter studies.

`recovery/` holds the constitutive-recovery examples (elastic and plastic drops,
sequential identification, sample complexity); see its README. Paper studies and figure
scripts live in `../experiments/`.
