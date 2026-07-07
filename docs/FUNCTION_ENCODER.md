# Function Encoder for TrackEUCLID (docs/FUNCTION_ENCODER.md)

Phase 3 material. Nothing in Phases 1 or 2 may import or depend on this beyond the dictionary interface defined in MATH_REFERENCE.md Section 7. This document exists so the training mathematics does not sit in the Phase 1 coding path.

## 1. Setting

Material family F = { mu_m } in L2_w on s = log10 I, s in [-4, 0], weight w(s) a 50/50 mixture of the corpus-realized flowing-I histogram and the uniform-in-log density. Basis network Phi: R -> R^K, input s, three hidden layers of 64, SiLU, K = 8 default (ablate K in {3, 5, 8, 12}); outputs interpreted as phi_k(s). The deployed artifact is NOT the network: it is a frozen 256-point tabulation with cubic interpolation, wrapped as FunctionEncoderDict, which exposes phi(I), dphi_dI(I) = dphi_ds / (I ln 10), and dphi_dlogI(I) per the interface contract. Torch (or jax) is permitted ONLY under ident/features/function_encoder_training/ and never elsewhere in ident/.

## 2. Training by closed-form projection

For material m with grid samples (s_i, mu_m(s_i)) and diagonal weights W_m from w(s):

  theta_m = ( Phi^T W_m Phi + eps Id )^{-1} Phi^T W_m mu_m,   eps = 1e-6 relative

Batch loss:

  L = SUM_m || mu_m - Phi theta_m ||^2_{W_m} / || mu_m ||^2_{W_m}  +  beta || G - Id ||_F^2

G_kl = INT w(s) phi_k phi_l ds on the grid, beta about 1e-2 (conditioning of G directly conditions the downstream solve). Backpropagate into the network only; theta_m is an explicit function of the network through the solve, so differentiate through it (cheap at K = 8). Adam 1e-3, cosine decay, about 50k steps, 80/20 material split. Acceptance: worst-family weighted relative L2 under 2 percent on held-out materials. Deliverables: tabulation file, Gram report, per-family span-coverage table.

## 3. Encoding through linear functionals

At test time mu is observed only through the weak-form functionals L_j[mu] = b_j, which are linear in mu. Hence L_j[sum theta_k phi_k] = sum theta_k L_j[phi_k] = sum_k A_jk theta_k: the feature matrix is A_jk = L_j[phi_k] and coefficient recovery is the identical constrained least squares used by Modes C and P. The function-encoder encoding step and the EUCLID identification step are the same projection, performed through the physics operator instead of pointwise samples. Identifiability is the column rank of A, governed jointly by the Gram matrix and the realized (I, D, w) diversity; this is the formal basis for I-stratified patches and joint multi-aspect-ratio systems.

## 4. Constraints, posterior, refusal calibration

Admissibility rows: sum theta_k phi_k(I_i) >= mu_min on a log-I grid; optional monotonicity via dphi_dlogI rows (report with and without). Convex QP, OSQP. Conditional posterior and ensemble as in MATH_REFERENCE.md Section 8. Refusal: relative projection residual r = |A theta_hat - b| / |b| conformally calibrated on held-out in-span corpus materials processed through the FULL pipeline at matched noise; with sorted calibration scores r_(1..n), threshold = the ceil((n+1)(1 - alpha)) order statistic, alpha = 0.1, with the standard exchangeability guarantee. The flag is always reported as pipeline consistency (model mismatch, kinematic inconsistency, insufficient excitation), never as a pure material-span certificate. Out-of-family probe required before any headline use: a cohesive or out-of-family material must trigger the flag or activate the cohesion tripwire with a warning, never silently fit.

## 5. Phase 3 definitions of done

Basis acceptance table per family; FunctionEncoderDict passing the same derivative cross-check tests as Modes C and P; Mode F recovery on G1 oracle within posterior bands wherever Mode P succeeds; conformal calibration curve; out-of-family probe behaving as specified; no regression in Modes C and P results when Mode F is enabled (assembly and solve untouched by construction).
