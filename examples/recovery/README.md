# Constitutive recovery examples (EUCLID weak form, no differentiable simulator)

These examples recover a material law from observed kinematics with a convex weak-form
momentum-residual solve that is linear in theta. The solve does not differentiate the
simulator. Instead, the inertia term from the observed trajectory supplies the absolute
force scale, so the moduli can be estimated without a force sensor. `warpmpm` supplies the
forward simulations, and the recovery uses NumPy least squares.

All of these run on the warp engine. The hyperelastic Mooney/Yeoh recovery lives elsewhere
(it uses the separate JAX `jmpm` engine for its forward model) and is intentionally not here.

## Running

Run from the `mpm_engine/` directory with the engine venv active:

```
cd mpm_engine
python -m examples.recovery.elastic_drop
```

Dumps, figures, and videos are written under `mpm_engine/out/`. Some examples consume
earlier dumps. Run `elastic_drop` and its `shape` mode before `sample_complexity`,
`elastic_identify_sequential`, and `elastic_render`. Run `plastic_drop gate` before
`plastic_identify_sequential`.

## Dependencies

Beyond the core engine, these need the `render` and `surface` extras (matplotlib for the star
shape and the plots, pyvista + scikit-image for the rendered bounce videos):

```
uv pip install -e '.[render,surface]'
```

## The examples

### elastic_drop.py
Drop a fixed-corotated (neo-Hookean) blob, observe the bounce, recover the elastic moduli
(mu, lambda -> E, nu) by the convex weak form. The first Piola-Kirchhoff stress is linear in
the moduli, so the solve is least squares over (test function x frame) rows.

- `python -m examples.recovery.elastic_drop` recovers the law from a sphere and then
  re-simulates it. E is recovered to about 0.3 percent, with 1.6 mm RMS error over 451
  frames.
- `python -m examples.recovery.elastic_drop shape` learns the moduli on a rectangle and
  predicts a held-out star bounce, with about 3.4 mm RMS error.
- `python -m examples.recovery.elastic_drop errors` -> reconstruction vs generalization error.

The bounce weakly excites the bulk mode (nu), so its estimate is loose.
`sample_complexity.py` measures this loss of information.

### plastic_drop.py
Plasticine (von-Mises) drop: recover (G, lambda, yield_stress) by the same weak form, with the
plastic gate. Yield is identifiable only once the material actually yields.

- `python -m examples.recovery.plastic_drop` recovers G and yield to about 0.1 percent
  from a single yielded drop.
- `python -m examples.recovery.plastic_drop gate` runs the same drop at two yield
  stresses. The yielded case identifies the yield stress; the elastic case returns
  `REFUSED` with a lower bound only.
- `python -m examples.recovery.plastic_drop gatefig` writes the gate figure.

### elastic_identify_sequential.py
Bayesian sequential identifiability: posterior mean and 95 percent credible band of E as more
frames are observed. Confidence is gated by the deformation actually seen (it tightens at
impact). `rollout` mode reports rollout RMSE vs number of observed frames.

- `python -m examples.recovery.elastic_identify_sequential rollout`
- `python -m examples.recovery.elastic_identify_sequential sphere` (or box) for the band figure.

### plastic_identify_sequential.py
This example updates (G, lambda, yield) frame by frame with recursive least squares. It
refuses the yield estimate until the deviatoric strain saturates. Run `plastic_drop gate`
first to produce the input dumps. G and yield are recovered to about 4 percent, and the
yield estimate stabilizes after the material yields at about t = 0.25 s.

### sample_complexity.py
This script checks the Gauss-Markov scaling of the weak-form recovery. The shear-mode
(mu) error falls as 1/sqrt(N) with the number of rows. The bulk-mode (lambda) error stays
high because the bounce provides little compression. Run `elastic_drop shape` first to
produce `box_truth.npz`.

### elastic_render.py
Render the bounce as a shaded, strain-coloured solid: truth law vs recovered law on the star,
and the rectangle training drop. Writes `out/nclaw_compare/nclaw_star_bounce.mp4` and
`nclaw_box_train.mp4`. Needs the dumps from `elastic_drop` and `elastic_drop shape`.

### nclaw_geom_render.py
Shared pyvista surfacing and scene helpers used by elastic_render (density-field marching
cubes, strain colouring, camera setup). Not a standalone example.
