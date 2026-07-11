# Function-encoder weights

Each `.npz` contains a frozen tabulation of `K` learned basis functions on a fixed input
grid. Training code is under `ident/features/function_encoder_training/`.
`FunctionEncoderDict` and the engine's tabulated materials load the files at inference.
Training requires Torch, but loading the tables is NumPy-only and does not require
retraining.

Total size ~84 KB. Schema math: `docs/FUNCTION_ENCODER.md`. The trainers write here by default.

## The weights

| file | family | grid (input) | K | held-out worst rel-L2 | report |
|------|--------|--------------|---|-----------------------|--------|
| `granular_mu_i.npz` | granular mu(I) | `s_grid` = log10 I in [-4, 0] | 8 | 4.2e-5 | `fe_report.json` |
| `viscous.npz` | Newtonian / Bingham / Herschel-Bulkley | `s_grid` = log10 shear-rate in [-1, 2] | 8 | 1.2e-4 | `fe_viscous_report.json` |
| `hyperelastic_1inv.npz` | neo-Hookean / Yeoh / Gent | `x_grid` = Ibar1 - 3 | 6 | 2.7e-3 | `hyperelastic_fe_train.json` |
| `hyperelastic_2inv.npz` | Mooney-Rivlin / generalized Rivlin | `x1, x2` = (Ibar1-3, Ibar2-3), 14x14 | 8 | 4.1e-3 | `hyperelastic_fe2_train.json` |

Keys per file:
- `granular_mu_i.npz`: `s_grid[256]`, `table[256,8]`, `weight[256]`, `K`, `theta_mean[8]`, `theta_cov[8,8]` (the family-coefficient Gaussian prior).
- `viscous.npz`: `s_grid[256]`, `table[256,8]`, `K`.
- `hyperelastic_1inv.npz`: `x_grid[256]`, `table[256,6]`, `K`.
- `hyperelastic_2inv.npz`: `x1[196]`, `x2[196]`, `table[196,16]` (stacked W1;W2 channels, shared K=8 coefficients), `K`, `n` (grid edge = 14).

## Training corpora and ranges

- granular (`corpus.py`, 2000 materials): Pouliquen (mu_s 0.18-0.45, delta_mu 0.05-0.55, I0 10^-2.5..10^0), constant (mu_s 0.2-0.55), power-law (mu_s 0.2-0.45, a 0.1-0.6, n 0.3-1.0), double-sigmoid.
- viscous (`viscous_corpus.py`, 2500 materials): Newtonian, power-law, Carreau, Bingham, Herschel-Bulkley.
- hyperelastic (`hyperelastic_train.py`): 1-inv over neo-Hookean/Yeoh/Gent (2400), 2-inv over Mooney/generalized Rivlin (3000).

## Consumers

- `ident/features/function_encoder.py` (`FunctionEncoderDict`) and `ident/gates/{g1_oracle,fe_corpus,fe_joint_prior}.py` load `granular_mu_i.npz`.
- `perception/*` (collapse3d, incline, joint_curve, nn_euclid, out_of_family_refusal, ...) load `granular_mu_i.npz` (and `viscous.npz` for the fluid path).
- `sim/hyperelastic.py` (`recover_fe`, `recover_fe2`) load the two hyperelastic tables.
- `mpm_engine` examples (`shear_cell_fe`, `shear_cell_3d`, `dough_fe_viscous`) load `viscous.npz` and re-simulate it via the `tabulated_viscous` material.

## Regenerate

```bash
python -m ident.features.function_encoder_training.train            # -> granular_mu_i.npz + fe_report.json
python -m ident.features.function_encoder_training.prior            # -> adds theta_mean/theta_cov to granular_mu_i.npz
python -m ident.features.function_encoder_training.viscous_train    # -> viscous.npz
python -m ident.features.function_encoder_training.hyperelastic_train      # -> hyperelastic_1inv.npz
python -m ident.features.function_encoder_training.hyperelastic_train i2   # -> hyperelastic_2inv.npz
```

The trainers write to this directory by default. They use `seed=0` on CPU and take a
few minutes per basis.

## Load

```python
import numpy as np
from ident.features.function_encoder import FunctionEncoderDict
d = np.load("mpm_engine/fe-weights/granular_mu_i.npz")           # paths are repo-root relative
fe = FunctionEncoderDict(s_grid=d["s_grid"], table=d["table"])   # phi / dphi_dI / dphi_dlogI
```

These weights live inside the `mpm_engine` git repo (not gitignored), so they are committed
with the package. Consumers in `ident/`, `perception/`, and `sim/` load them by the repo-root
relative path `mpm_engine/fe-weights/<name>.npz`.
