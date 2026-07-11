# Function encoder for TrackEUCLID

This is a Phase 3 component. Phases 1 and 2 depend only on the dictionary interface
defined in MATH_REFERENCE.md Section 7; the training implementation stays outside the
Phase 1 path.

## 1. Setting

The material family F = { mu_m } lies in L2_w on s = log10 I for s in [-4, 0].
The weight w(s) is a 50/50 mixture of the corpus flowing-I histogram and a density
that is uniform in log I. The basis network Phi: R -> R^K takes s as input, uses
three 64-unit SiLU hidden layers, and outputs phi_k(s). K = 8 by default, with
ablations at K in {3, 5, 8, 12}.

Deployment uses a frozen 256-point tabulation with cubic interpolation, wrapped by
FunctionEncoderDict. The wrapper exposes phi(I), dphi_dI(I) = dphi_ds / (I ln 10),
and dphi_dlogI(I). Torch or JAX may be used only under
ident/features/function_encoder_training/; the rest of ident/ remains NumPy-only.

## 2. Training by closed-form projection

For material m with grid samples (s_i, mu_m(s_i)) and diagonal weights W_m from w(s):

  theta_m = ( Phi^T W_m Phi + eps Id )^{-1} Phi^T W_m mu_m,   eps = 1e-6 relative

Batch loss:

  L = SUM_m || mu_m - Phi theta_m ||^2_{W_m} / || mu_m ||^2_{W_m}  +  beta || G - Id ||_F^2

G_kl = INT w(s) phi_k phi_l ds on the grid, with beta about 1e-2. The conditioning
of G directly affects the downstream solve. Backpropagation updates only the network;
theta_m is differentiated through the explicit solve, which is inexpensive at K = 8.
Training uses Adam at 1e-3,
cosine decay, about 50k steps, and an 80/20 material split. Acceptance requires a
worst-family weighted relative L2 below 2 percent on held-out materials. Outputs are
the tabulation, Gram report, and per-family span-coverage table.

## 3. Encoding through linear functionals

At test time, mu is observed through weak-form functionals L_j[mu] = b_j that are
linear in mu. Therefore L_j[sum theta_k phi_k] = sum_k A_jk theta_k, where
A_jk = L_j[phi_k]. Coefficients use the same constrained least-squares solve as
Modes C and P, applied through the physics operator instead of pointwise samples.
Identifiability is the column rank of A and depends on the Gram matrix and the
realized (I, D, w) diversity. This motivates I-stratified patches and joint
multi-aspect-ratio systems.

## 4. Constraints, posterior, refusal calibration

Admissibility rows enforce sum theta_k phi_k(I_i) >= mu_min on a log-I grid.
Monotonicity through dphi_dlogI rows is optional and is reported both with and without
the constraint. The solve is a convex QP in OSQP. The conditional posterior and
ensemble follow MATH_REFERENCE.md Section 8.

The refusal score r = |A theta_hat - b| / |b| is conformally calibrated on held-out,
in-span corpus materials processed through the full pipeline at matched noise. For
sorted scores r_(1..n), the threshold is the ceil((n+1)(1 - alpha)) order statistic,
with alpha = 0.1 and the standard exchangeability guarantee. Report the flag as a
pipeline-consistency result covering model mismatch, kinematic inconsistency, and
insufficient excitation, not as a pure material-span certificate. Before reporting
results on new families, a cohesive or out-of-family probe must trigger the flag or
the cohesion warning rather than silently fit.

## 5. Phase 3 definitions of done

Phase 3 is complete when it provides:

- a basis-acceptance table for each family;
- FunctionEncoderDict derivative checks matching Modes C and P;
- Mode F recovery on the G1 oracle within posterior bands wherever Mode P succeeds;
- a conformal calibration curve and the specified out-of-family response; and
- unchanged Mode C and P results when Mode F is enabled; Mode F does not modify their
  assembly or solve paths.
