# Constitutive recovery examples (EUCLID weak form, no differentiable simulator)

These examples recover a material law from observed kinematics by a convex, linear-in-theta
weak-form momentum-residual solve. The simulator is never differentiated: the inertia term from
the observed trajectory supplies the absolute force scale, so the moduli are pinned without a
force sensor. The forward simulator is the warp MPM engine in this repo (`warpmpm`); the
recovery is plain numpy least squares.

All of these run on the warp engine. The hyperelastic Mooney/Yeoh recovery lives elsewhere
(it uses the separate JAX `jmpm` engine for its forward model) and is intentionally not here.

## Running

Run from the `mpm_engine/` directory with the engine venv active:

```
cd mpm_engine
python -m examples.recovery.elastic_drop
```

Outputs (dumps, figures, videos) are written under `mpm_engine/out/`. Some examples consume
dumps produced by another, so the suggested order is: `elastic_drop` (and its `shape` mode),
then `sample_complexity`, `elastic_identify_sequential`, `elastic_render`; and `plastic_drop`
(with `gate`) before `plastic_identify_sequential`.

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

- `python -m examples.recovery.elastic_drop` -> sphere: recover then re-simulate the recovered
  law. Verified: E recovered to ~0.3 percent; re-sim 1.6 mm RMS over 451 frames.
- `python -m examples.recovery.elastic_drop shape` -> learn the moduli on a RECTANGLE, predict a
  held-out STAR bounce. Verified: ~3.4 mm RMS on the unseen geometry.
- `python -m examples.recovery.elastic_drop errors` -> reconstruction vs generalization error.

The bulk mode (nu) is weakly excited by a bounce and is recovered loosely; that starvation is
the subject of sample_complexity.py.

### plastic_drop.py
Plasticine (von-Mises) drop: recover (G, lambda, yield_stress) by the same weak form, with the
plastic gate. Yield is identifiable only once the material actually yields.

- `python -m examples.recovery.plastic_drop` -> recover from a single yielded drop. Verified:
  G and yield to ~0.1 percent.
- `python -m examples.recovery.plastic_drop gate` -> the gate: same drop at two yield stresses;
  yield is identified in the yielded case and REFUSED (lower bound only) in the elastic case.
- `python -m examples.recovery.plastic_drop gatefig` -> the gate figure.

### elastic_identify_sequential.py
Bayesian sequential identifiability: posterior mean and 95 percent credible band of E as more
frames are observed. Confidence is gated by the deformation actually seen (it tightens at
impact). `rollout` mode reports rollout RMSE vs number of observed frames.

- `python -m examples.recovery.elastic_identify_sequential rollout`
- `python -m examples.recovery.elastic_identify_sequential sphere` (or box) for the band figure.

### plastic_identify_sequential.py
The plasticine analogue in a recursive-least-squares (RLS) form: (G, lambda, yield) updated
frame by frame, with the yield refused until the deviatoric strain saturates (the plastic gate
in time). Needs `plastic_drop gate` to have produced the dumps first. Verified: G and yield to
~4 percent, yield locks in once the material yields (around t = 0.25 s).

### sample_complexity.py
Gauss-Markov validation of the weak-form recovery: the shear-mode (mu) error falls as
1/sqrt(N) with the number of weak-form rows, while the bulk mode (lambda) stays high because a
bounce does not compress enough to excite it (the weak form refusing an uninformative mode).
Needs `elastic_drop shape` to have produced `box_truth.npz`.

### elastic_render.py
Render the bounce as a shaded, strain-coloured solid: truth law vs recovered law on the star,
and the rectangle training drop. Writes `out/nclaw_compare/nclaw_star_bounce.mp4` and
`nclaw_box_train.mp4`. Needs the dumps from `elastic_drop` and `elastic_drop shape`.

### nclaw_geom_render.py
Shared pyvista surfacing and scene helpers used by elastic_render (density-field marching
cubes, strain colouring, camera setup). Not a standalone example.
