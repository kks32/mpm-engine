# Experiments

Paper studies and figure scripts. Unlike `../examples/`, these are kept for
reproducibility of specific results rather than as maintained demos: most encode one
study's protocol, several consume caches written by an earlier script, and some need
heavy extras (torch, CoTracker). Run them from the repository root with the project
environment active.

Rheology recovery (the shear-cell studies; all need `fe-weights/viscous.npz`):

- `shear_cell_3d.py`: 3D lift of the 2D shear cell (`examples/shear_cell_fe.py`); FE
  recovery plus a held-out rollout, writes `out/shear_cell_3d/rollout_3d.npz`.
- `strong_vs_weak.py`: strong-form (pointwise stress oracle) versus weak-form
  (wall-force power balance) recovery on the same sweeps.
- `dough_fe_viscous.py`: FE versus Bingham recovery of a shear-thinning dough at three
  press speeds.
- `correct_model_check.py`, `rollout_snapshots.py`, `rollout_error_contours.py`,
  `shear_rollout_video.py`: figures and videos built from `rollout_3d.npz`; run
  `shear_cell_3d.py` first.

Volume transfer and real-data protocol:

- `predict_volume_franka.py`: learn the law on one squeeze, predict plate force on a
  different-volume blob.
- `predict_volume_rollout.py`: predict deformation of an unseen volume, validated
  through the render-and-CoTracker pipeline.
- `realdata_pipeline.py`: the real-data-shaped ingest (textured video plus force CSV to
  rheology) exercised on synthetic renders at two volumes.
- `volume_holdout_check.py`: the 2x2 cross-volume held-out matrix; run
  `realdata_pipeline.py` first.

Manipulation planning and baselines:

- `shape_planning.py`: CEM shape planner over the MPM engine (Chamfer and EMD losses).
- `transfer_identify_plan.py`: identify on a small block, plan shaping on a large one;
  the size-transfer study.
- `three_prong.py`, `dough_franka_threeprong.py`: three-jaw shaping, standalone and
  arm-mounted.
- `gripper_render_dough.py`: PyVista render of the `examples/gripper_shape.py` plan.
- `gns_baseline.py`: a graph-network simulator baseline (torch) trained on engine
  rollouts, planned with the same CEM, against the identified-MPM forward model.

Perception and rendering studies:

- `rollout_franka_cotracker.py`: arm squeeze rendered as speckle, deformation extracted
  with CoTracker3, rollout error computed.
- `rollout_force_video.py`: side-by-side render with a live force trace; needs
  `rollout_franka_cotracker.py`'s output.
- `speckle_particle_videos.py`, `surface_track_test.py`: speckle-render inputs for
  tracking, and smooth-versus-textured surface trackability.

Design:

- `pressure_covariance_sweep.py`: information-matrix and covariance study of the
  (tau_y, eta) weak form across press loads.

Floods:

- `flood_sweep.py`: truck displacement and yaw over a grid of flood depths and surge
  velocities, using `warpmpm.vehicle.FloodScene` (the study form of
  `examples/flood_vehicle.py`).

Implicit solver (Phase 4b, docs/implicit_plan.md):

- `qs_prototype.py`: numpy prototype of the quasi-static implicit formulation
  (matrix-free Newton-Krylov, Hencky stress, quadratic B-splines); gate 1, the
  analytic equilibrium column.
- `qs_press_ab.py`: gate 2, the same elastic column pressed by a plate in the
  implicit solver and in the explicit engine (slow press, then a damped hold for
  the static reaction); forces agree to 2.5 percent.
