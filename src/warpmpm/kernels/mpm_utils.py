import warp as wp
from warpmpm.kernels.warp_utils import *  # noqa: F401,F403
import numpy as np
import math


# compute stress from F
@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)

@wp.func
def kirchoff_stress_water(
    J: float, bulk: float
):
    gamma = 1.1 # gamma is set to be a liitle greater than 1 for weakly compressible fluids
    pressure = -bulk * (wp.pow(J, -gamma) - 1.)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    cauchy_stress = id * pressure
    return J * cauchy_stress

@wp.func
def kirchoff_stress_newtonian(
    J: float, bulk: float, L: wp.mat33, eta: float, tau_y: float,
    pk: float, pn: float
):
    # weakly-compressible generalized-Newtonian / Bingham / Herschel-Bulkley fluid:
    # Cauchy = -p I + 2 eta_app dev(D),
    #   eta_app = eta + tau_y / |gd|_eps + pk * |gd|_eps^(pn - 1)
    # with p from the water EOS, D = sym(L), and |gd|_eps = sqrt(2 dev(D):dev(D)
    # + eps^2) the regularized deviatoric shear rate. tau_y=0, pk=0 -> Newtonian;
    # pk>0, pn<1 -> shear-thinning power law. Kirchhoff = J * Cauchy.
    eps = 0.05
    gamma = 1.1
    pressure = -bulk * (wp.pow(J, -gamma) - 1.0)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    D = 0.5 * (L + wp.transpose(L))
    trd = (D[0, 0] + D[1, 1] + D[2, 2]) / 3.0
    D_dev = D - id * trd
    dd = (D_dev[0, 0] * D_dev[0, 0] + D_dev[0, 1] * D_dev[0, 1] + D_dev[0, 2] * D_dev[0, 2]
          + D_dev[1, 0] * D_dev[1, 0] + D_dev[1, 1] * D_dev[1, 1] + D_dev[1, 2] * D_dev[1, 2]
          + D_dev[2, 0] * D_dev[2, 0] + D_dev[2, 1] * D_dev[2, 1] + D_dev[2, 2] * D_dev[2, 2])
    gd = wp.sqrt(2.0 * dd + eps * eps)
    eta_app = eta + tau_y / gd + pk * wp.pow(gd, pn - 1.0)
    cauchy = id * pressure + 2.0 * eta_app * D_dev
    return J * cauchy

@wp.func
def kirchoff_stress_tabulated(
    J: float, bulk: float, L: wp.mat33,
    table: wp.array(dtype=float), smin: float, smax: float, n: int
):
    # Same weakly-compressible generalized-Newtonian form as kirchoff_stress_newtonian,
    # but eta_app(gd) is read from a TABLE on s = log10(gd) in [smin, smax] (n uniform
    # samples) with clamped linear interpolation, instead of the parametric HB formula.
    # This lets an FE-recovered eta_app(gd) curve be re-simulated directly. eps and the
    # EOS match the newtonian kernel exactly so the only difference is the eta_app source.
    eps = 0.05
    gamma = 1.1
    pressure = -bulk * (wp.pow(J, -gamma) - 1.0)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    D = 0.5 * (L + wp.transpose(L))
    trd = (D[0, 0] + D[1, 1] + D[2, 2]) / 3.0
    D_dev = D - id * trd
    dd = (D_dev[0, 0] * D_dev[0, 0] + D_dev[0, 1] * D_dev[0, 1] + D_dev[0, 2] * D_dev[0, 2]
          + D_dev[1, 0] * D_dev[1, 0] + D_dev[1, 1] * D_dev[1, 1] + D_dev[1, 2] * D_dev[1, 2]
          + D_dev[2, 0] * D_dev[2, 0] + D_dev[2, 1] * D_dev[2, 1] + D_dev[2, 2] * D_dev[2, 2])
    gd = wp.sqrt(2.0 * dd + eps * eps)
    s = wp.log(gd) * 0.43429448190325176          # log10(gd)
    # map s -> fractional index in [0, n-1], clamped (flat extrapolation at both ends)
    idx = (s - smin) / (smax - smin) * float(n - 1)
    idx = wp.clamp(idx, 0.0, float(n - 1) - 1.0e-6)
    i0 = int(wp.floor(idx))
    frac = idx - float(i0)
    eta_app = table[i0] * (1.0 - frac) + table[i0 + 1] * frac
    cauchy = id * pressure + 2.0 * eta_app * D_dev
    return J * cauchy

@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    b = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3(
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
    )
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * wp.vec3(
        1.0, 1.0, 1.0
    )
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_StVK(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (
        1.0 / sig[0]
    )
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (
        1.0 / sig[1]
    )
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (
        1.0 / sig[2]
    )
    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)


@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening[mat] > 0.5:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi[mat] * delta_gamma
            )
        return F_elastic
    else:
        return F_trial

@wp.func
def von_mises_return_mapping_with_damage(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        if model.yield_stress[p] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        model.yield_stress[p] = model.yield_stress[p] - model.softening[mat] * wp.length(
            (delta_gamma / epsilon_hat_norm) * epsilon_hat
        )
        if model.yield_stress[p] <= 0:
            model.mu[p] = 0.0
            model.lam[p] = 0.0
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening[mat] > 0.5:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi[mat] * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


# for toothpaste
@wp.func
def viscoplasticity_return_mapping_with_StVK(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int, dt: float
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    b_trial = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - wp.vec3(
        trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
    )
    s_trial = 2.0 * model.mu[p] * epsilon_hat
    s_trial_norm = wp.length(s_trial)
    y = s_trial_norm - wp.sqrt(2.0 / 3.0) * model.yield_stress[p]
    if y > 0:
        mu_hat = model.mu[p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
        s_new_norm = s_trial_norm - y / (
            1.0 + model.plastic_viscosity[mat] / (2.0 * mu_hat * dt)
        )
        s_new = (s_new_norm / s_trial_norm) * s_trial
        epsilon_new = 1.0 / (2.0 * model.mu[p]) * s_new + wp.vec3(
            trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
        )
        sig_elastic = wp.mat33(
            wp.exp(epsilon_new[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        return F_elastic
    else:
        return F_trial


@wp.func
def kirchoff_stress_hencky(
    U: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    # Kirchhoff stress for Hencky elasticity, coaxial with the left stretch:
    # tau = U diag(2 mu eps_i + lam tr(eps)) U^T with eps = log(sig)
    sig_c = wp.vec3(
        wp.max(sig[0], 1e-6), wp.max(sig[1], 1e-6), wp.max(sig[2], 1e-6)
    )
    eps = wp.vec3(wp.log(sig_c[0]), wp.log(sig_c[1]), wp.log(sig_c[2]))
    tr_eps = eps[0] + eps[1] + eps[2]
    tau0 = 2.0 * mu * eps[0] + lam * tr_eps
    tau1 = 2.0 * mu * eps[1] + lam * tr_eps
    tau2 = 2.0 * mu * eps[2] + lam * tr_eps
    tau_diag = wp.mat33(tau0, 0.0, 0.0, 0.0, tau1, 0.0, 0.0, 0.0, tau2)
    return U * tau_diag * wp.transpose(U)


@wp.func
def mu_i_return_mapping(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int, dt: float
):
    # Local mu(I) rheology (TrackEUCLID), Dunatunga-Kamrin style return:
    # Hencky elastic predictor, scalar plastic correction on the yield
    # surface tau_bar = mu(I) p with I = gamma_dot_p d sqrt(rho_s / p).
    # The return is deviatoric (non dilatant): J and the pressure are
    # preserved, so the dumped Cauchy stress trace equals the pressure the
    # update consumed. See docs/simulator_notes.md.
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    sig_c = wp.vec3(
        wp.max(wp.abs(sig[0]), 1e-6),
        wp.max(wp.abs(sig[1]), 1e-6),
        wp.max(wp.abs(sig[2]), 1e-6),
    )
    eps = wp.vec3(wp.log(sig_c[0]), wp.log(sig_c[1]), wp.log(sig_c[2]))
    tr_eps = eps[0] + eps[1] + eps[2]

    F_elastic = F_trial

    if tr_eps >= 0.0:
        # volumetric expansion: cohesionless free separation, stress free,
        # no tension memory
        F_elastic = U * wp.transpose(V)
    else:
        G = model.mu[p]
        lam = model.lam[p]
        J = sig_c[0] * sig_c[1] * sig_c[2]

        mean_eps = tr_eps / 3.0
        dev0 = eps[0] - mean_eps
        dev1 = eps[1] - mean_eps
        dev2 = eps[2] - mean_eps
        dev_norm = wp.sqrt(dev0 * dev0 + dev1 * dev1 + dev2 * dev2)

        p_K = -(2.0 * G / 3.0 + lam) * tr_eps  # > 0 in compression
        tau_bar_K = 2.0 * G * dev_norm / wp.sqrt(2.0)
        p_C = p_K / J

        mu_s = model.muI_mu_s[mat]
        dmu = model.muI_delta_mu[mat]
        I0 = model.muI_I0[mat]

        if tau_bar_K > mu_s * p_K:
            # plastic: bisect g(gdp) = tau_bar_K - G dt gdp - mu(I) p_K = 0,
            # strictly decreasing, sign change on [0, tau_bar_K / (G dt)]
            I_coef = model.muI_d[mat] * wp.sqrt(model.muI_rho_s[mat] / p_C)
            lo = float(0.0)
            hi = tau_bar_K / (G * dt)
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                I_mid = mid * I_coef
                mu_mid = mu_s + dmu * I_mid / (I_mid + I0)
                g_mid = tau_bar_K - G * dt * mid - mu_mid * p_K
                if g_mid > 0.0:
                    lo = mid
                else:
                    hi = mid
            gdp = 0.5 * (lo + hi)
            tau_bar_new = tau_bar_K - G * dt * gdp
            scale = tau_bar_new / wp.max(tau_bar_K, 1e-20)
            e0 = dev0 * scale + mean_eps
            e1 = dev1 * scale + mean_eps
            e2 = dev2 * scale + mean_eps
            sig_new = wp.mat33(
                wp.exp(e0), 0.0, 0.0, 0.0, wp.exp(e1), 0.0, 0.0, 0.0, wp.exp(e2)
            )
            F_elastic = U * sig_new * wp.transpose(V)

    return F_elastic


@wp.func
def mu_i_tabulated_return_mapping(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int, dt: float
):
    # IDENTICAL to mu_i_return_mapping (Dunatunga-Kamrin deviatoric return) except the
    # friction law mu(I) is read from a TABLE on s = log10(I) in [smin, smax] (n uniform
    # samples, clamped linear interpolation, flat extrapolation) instead of the parametric
    # Pouliquen formula mu_s + dmu I/(I+I0). This lets an NN-EUCLID- (or FE-) recovered
    # mu(I) curve be re-simulated directly. The mu-table is stored in model.eta_table
    # (re-used; a granular tabulated material and a viscous tabulated material are never
    # active in the same run). Everything else -- pressure, bisection, stress -- matches
    # material 9 exactly, so the only difference is the source of mu(I).
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    sig_c = wp.vec3(
        wp.max(wp.abs(sig[0]), 1e-6),
        wp.max(wp.abs(sig[1]), 1e-6),
        wp.max(wp.abs(sig[2]), 1e-6),
    )
    eps = wp.vec3(wp.log(sig_c[0]), wp.log(sig_c[1]), wp.log(sig_c[2]))
    tr_eps = eps[0] + eps[1] + eps[2]

    F_elastic = F_trial

    if tr_eps >= 0.0:
        F_elastic = U * wp.transpose(V)
    else:
        G = model.mu[p]
        lam = model.lam[p]
        J = sig_c[0] * sig_c[1] * sig_c[2]

        mean_eps = tr_eps / 3.0
        dev0 = eps[0] - mean_eps
        dev1 = eps[1] - mean_eps
        dev2 = eps[2] - mean_eps
        dev_norm = wp.sqrt(dev0 * dev0 + dev1 * dev1 + dev2 * dev2)

        p_K = -(2.0 * G / 3.0 + lam) * tr_eps
        tau_bar_K = 2.0 * G * dev_norm / wp.sqrt(2.0)
        p_C = p_K / J

        smin = model.eta_table_smin
        smax = model.eta_table_smax
        nt = model.eta_table_n

        # mu at zero shear (table value at smin) sets the static yield check
        mu_static = model.eta_table[0]
        if tau_bar_K > mu_static * p_K:
            I_coef = model.muI_d[mat] * wp.sqrt(model.muI_rho_s[mat] / p_C)
            lo = float(0.0)
            hi = tau_bar_K / (G * dt)
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                I_mid = mid * I_coef
                # mu(I_mid) by clamped linear interpolation on s = log10(I)
                s = wp.log(wp.max(I_mid, 1e-12)) * 0.43429448190325176
                idx = (s - smin) / (smax - smin) * float(nt - 1)
                idx = wp.clamp(idx, 0.0, float(nt - 1) - 1.0e-6)
                i0 = int(wp.floor(idx))
                frac = idx - float(i0)
                mu_mid = model.eta_table[i0] * (1.0 - frac) + model.eta_table[i0 + 1] * frac
                g_mid = tau_bar_K - G * dt * mid - mu_mid * p_K
                if g_mid > 0.0:
                    lo = mid
                else:
                    hi = mid
            gdp = 0.5 * (lo + hi)
            tau_bar_new = tau_bar_K - G * dt * gdp
            scale = tau_bar_new / wp.max(tau_bar_K, 1e-20)
            e0 = dev0 * scale + mean_eps
            e1 = dev1 * scale + mean_eps
            e2 = dev2 * scale + mean_eps
            sig_new = wp.mat33(
                wp.exp(e0), 0.0, 0.0, 0.0, wp.exp(e1), 0.0, 0.0, 0.0, wp.exp(e2)
            )
            F_elastic = U * sig_new * wp.transpose(V)

    return F_elastic


@wp.func
def j_cs_of_I(model: MPMModelStruct, mat: int, I: float):
    # critical-state STRESS-FREE volume ratio the material relaxes toward at inertial
    # number I. Phi_c(I) = Phi_init*(1 - chi I/(I+I0)) (looser at high I), and
    # J_cs = Phi_init/Phi_c = 1/(1 - chi I/(I+I0)) >= 1 (dilates with shear rate).
    I0 = model.muI_I0[mat]
    chi = model.muI_phi_chi[mat]
    denom = wp.max(1.0 - chi * I / (I + I0), 1e-3)
    return 1.0 / denom


@wp.func
def mu_i_phi_pressure(model: MPMModelStruct, mat: int, J: float, Jp_ref: float):
    # compaction pressure from the elastic compression relative to the (slowly
    # relaxing) stress-free reference volume Jp_ref: p = K (Jp_ref/J - 1)_+ . This is
    # the elastic EOS shifted by the critical-state reference, so it is SMOOTH and
    # MONOTONE in J (stable, like the mu(I) elastic EOS) and the rate dependence
    # enters only through the slow drift of Jp_ref -- not an instantaneous I feedback.
    K = model.E[mat] / (3.0 * (1.0 - 2.0 * model.nu[mat]))
    p = K * (Jp_ref / wp.max(J, 1e-6) - 1.0)
    return wp.max(p, 0.0)


@wp.func
def mu_i_phi_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int, mat: int, dt: float
):
    # Compressible mu(I)-Phi(I) (TrackEUCLID, material 11). Deviatoric mu(I) yield
    # against the COMPACTION pressure p = K(Phi/Phi_c(I)-1)_+ (not the elastic EOS),
    # so the solid fraction relaxes toward the rate-dependent critical state Phi_c(I)
    # under a free surface -- an emergent O(chi) observable density signal. The
    # volumetric strain is PRESERVED in the return (the volume is physical, set by the
    # flow); the pressure comes from Phi via the compaction law, applied in the stress.
    # The realized inertial number I is stored in particle_Jp for the stress and the
    # next step's Phi_c lag.
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)
    sig_c = wp.vec3(
        wp.max(wp.abs(sig[0]), 1e-6),
        wp.max(wp.abs(sig[1]), 1e-6),
        wp.max(wp.abs(sig[2]), 1e-6),
    )
    eps = wp.vec3(wp.log(sig_c[0]), wp.log(sig_c[1]), wp.log(sig_c[2]))
    J = sig_c[0] * sig_c[1] * sig_c[2]
    tr_eps = eps[0] + eps[1] + eps[2]
    mean_eps = tr_eps / 3.0
    dev0 = eps[0] - mean_eps
    dev1 = eps[1] - mean_eps
    dev2 = eps[2] - mean_eps
    dev_norm = wp.sqrt(dev0 * dev0 + dev1 * dev1 + dev2 * dev2)

    G = model.mu[p]
    # stress-free reference volume Jp_ref (particle_Jp); the solver inits it to 0,
    # which we read as the natural value 1.0 (undeformed reference).
    Jp_ref = state.particle_Jp[p]
    if Jp_ref < 0.5:
        Jp_ref = 1.0
    p_comp = mu_i_phi_pressure(model, mat, J, Jp_ref)     # compaction pressure (Cauchy)
    beta = 8.0                                            # dilatancy rate PER UNIT PLASTIC STRAIN

    F_elastic = F_trial
    state.particle_Jp[p] = Jp_ref                         # default: reference frozen (no plastic strain)

    if p_comp <= 0.0:
        # at or below the reference volume: loose / stress free, only relax deviatoric;
        # preserve the volume (do not inject volume by resetting J to 1).
        s_iso = wp.exp(mean_eps)
        F_elastic = U * wp.mat33(s_iso, 0.0, 0.0, 0.0, s_iso, 0.0, 0.0, 0.0, s_iso) * wp.transpose(V)
    else:
        p_K = p_comp * J                                  # Kirchhoff pressure scale
        tau_bar_K = 2.0 * G * dev_norm / wp.sqrt(2.0)
        mu_s = model.muI_mu_s[mat]
        dmu = model.muI_delta_mu[mat]
        I0 = model.muI_I0[mat]
        if tau_bar_K > mu_s * p_K:
            I_coef = model.muI_d[mat] * wp.sqrt(model.muI_rho_s[mat] / p_comp)
            lo = float(0.0)
            hi = tau_bar_K / (G * dt)
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                I_mid = mid * I_coef
                mu_mid = mu_s + dmu * I_mid / (I_mid + I0)
                g_mid = tau_bar_K - G * dt * mid - mu_mid * p_K
                if g_mid > 0.0:
                    lo = mid
                else:
                    hi = mid
            gdp = 0.5 * (lo + hi)
            I_real = gdp * I_coef
            # DILATANCY (critical state) tied to PLASTIC STRAIN, not wall-clock: the
            # stress-free reference volume relaxes toward J_cs(I) by an amount
            # proportional to the plastic shear increment gdp*dt. Static / jittering
            # material (gdp~0) does NOT drift, so it compacts and holds hydrostatic
            # pressure; only sustained flow dilates. dgamma capped at 1 for stability.
            dgamma = wp.min(beta * gdp * dt, 1.0)
            j_cs = j_cs_of_I(model, mat, I_real)
            state.particle_Jp[p] = Jp_ref + dgamma * (j_cs - Jp_ref)
            tau_bar_new = tau_bar_K - G * dt * gdp
            scale = tau_bar_new / wp.max(tau_bar_K, 1e-20)
            e0 = dev0 * scale + mean_eps
            e1 = dev1 * scale + mean_eps
            e2 = dev2 * scale + mean_eps
            sig_new = wp.mat33(
                wp.exp(e0), 0.0, 0.0, 0.0, wp.exp(e1), 0.0, 0.0, 0.0, wp.exp(e2)
            )
            F_elastic = U * sig_new * wp.transpose(V)
    return F_elastic


@wp.func
def kirchoff_stress_mu_i_phi(
    U: wp.mat33, sig: wp.vec3, G: float, p_comp: float
):
    # deviatoric Hencky (mu(I)-limited via the returned F) + compaction isotropic
    # tau = U diag(2 G dev_eps_i) U^T  -  p_comp * J * I   (Cauchy pressure = p_comp)
    sig_c = wp.vec3(wp.max(sig[0], 1e-6), wp.max(sig[1], 1e-6), wp.max(sig[2], 1e-6))
    eps = wp.vec3(wp.log(sig_c[0]), wp.log(sig_c[1]), wp.log(sig_c[2]))
    J = sig_c[0] * sig_c[1] * sig_c[2]
    mean_eps = (eps[0] + eps[1] + eps[2]) / 3.0
    iso = -p_comp * J
    tau0 = 2.0 * G * (eps[0] - mean_eps) + iso
    tau1 = 2.0 * G * (eps[1] - mean_eps) + iso
    tau2 = 2.0 * G * (eps[2] - mean_eps) + iso
    tau_diag = wp.mat33(tau0, 0.0, 0.0, 0.0, tau1, 0.0, 0.0, 0.0, tau2)
    return U * tau_diag * wp.transpose(U)


@wp.func
def sand_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int, mat: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    epsilon = wp.vec3(
        wp.log(wp.max(wp.abs(sig[0]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[1]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[2]), 1e-14)),
    )
    sigma_out = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    epsilon_hat_norm = wp.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * model.lam[p] + 2.0 * model.mu[p])
        / (2.0 * model.mu[p])
        * tr
        * model.alpha[mat]
    )

    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))

        F_elastic = U * wp.diag(s_new) * wp.transpose(V)
    return F_elastic


@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    mat = state.particle_material[p]
    model.mu[p] = model.E[mat] / (2.0 * (1.0 + model.nu[mat]))
    model.lam[p] = model.E[mat] * model.nu[mat] / ((1.0 + model.nu[mat]) * (1.0 - 2.0 * model.nu[mat]))

@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx


@wp.func
def update_cov(state: MPMStateStruct, p: int, grad_v: wp.mat33, dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    state.particle_cov[p * 6] = cov_np1[0, 0]
    state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    p = wp.tid()
    if state.particle_selection[p] == 0:
        stress = state.particle_stress[p]
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.matrix_from_cols(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.matrix_from_cols(
            fx - wp.vec3(1.5),
            -2.0 * (fx - wp.vec3(1.0)),
            fx - wp.vec3(0.5),
        )

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    dpos = (
                        wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    ) * model.dx
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    C = state.particle_C[p]
                    # if model.rpic = 0, standard apic
                    C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (
                        C - wp.transpose(C)
                    )
                    if model.rpic_damping < -0.001:
                        # standard pic
                        C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    elastic_force = -state.particle_vol[p] * stress * dweight
                    v_in_add = (
                        weight
                        * state.particle_mass[p]
                        * (state.particle_v[p] + C * dpos)
                        + dt * elastic_force
                    )
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                    )


# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * (
            1.0 / state.grid_m[grid_x, grid_y, grid_z]
        )
        # add gravity
        v_out = v_out + dt * model.gravitational_accelaration
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out


@wp.kernel
def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        if state.particle_material[p] == 7 or state.particle_material[p] == 8:  # stationary/rigid handled separately
            return
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.matrix_from_cols(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.matrix_from_cols(
            fx - wp.vec3(1.5),
            -2.0 * (fx - wp.vec3(1.0)),
            fx - wp.vec3(0.5),
        )
        new_v = wp.vec3(0.0, 0.0, 0.0)
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = new_v + grid_v * weight
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        state.particle_v[p] = new_v
        state.particle_x[p] = state.particle_x[p] + dt * new_v
        state.particle_C[p] = new_C
        # new_F is the discrete velocity gradient L with L_ij = dv_i/dx_j
        # (sum_node v_node[i] * d w_node/dx_j). Stored for the TrackEUCLID dump.
        state.particle_L[p] = new_F
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_tmp = (I33 + new_F * dt) * state.particle_F[p]
        state.particle_F_trial[p] = F_tmp

        if model.update_cov_with_F:
            update_cov(state, p, new_F, dt)


# compute (Kirchhoff) stress = stress(returnMap(F_trial))
@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        mat = state.particle_material[p]

        # apply return mapping
        if mat == 1:  # metal
            state.particle_F[p] = von_mises_return_mapping(
                state.particle_F_trial[p], model, p, mat
            )
        elif mat == 2:  # sand
            state.particle_F[p] = sand_return_mapping(
                state.particle_F_trial[p], state, model, p, mat
            )
        elif mat == 3:  # visplas, with StVk+VM, no thickening
            state.particle_F[p] = viscoplasticity_return_mapping_with_StVK(
                state.particle_F_trial[p], model, p, mat, dt
            )
        elif mat == 5:  # plasticine
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p, mat
            )
        elif mat == 9:  # local mu(I) rheology (TrackEUCLID)
            state.particle_F[p] = mu_i_return_mapping(
                state.particle_F_trial[p], model, p, mat, dt
            )
        elif mat == 13:  # tabulated mu(I) (NN-EUCLID / FE-recovered curve, TrackEUCLID)
            state.particle_F[p] = mu_i_tabulated_return_mapping(
                state.particle_F_trial[p], model, p, mat, dt
            )
        elif mat == 11:  # compressible mu(I)-Phi(I) dilatancy (TrackEUCLID)
            state.particle_F[p] = mu_i_phi_return_mapping(
                state.particle_F_trial[p], state, model, p, mat, dt
            )
        elif mat == 6 or mat == 10 or mat == 12:  # fluid / newtonian / tabulated (viscous)
            J = wp.determinant(state.particle_F_trial[p])
            Jcbr = J**(1.0 / 3.0)
            state.particle_F[p] = wp.mat33(Jcbr, 0.0, 0.0, 0.0, Jcbr, 0.0, 0.0, 0.0, Jcbr)
        elif mat == 7 or mat == 8:  # stationary / rigid — no deformation
            state.particle_F[p] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        else:  # jelly (0), snow (4), or custom
            state.particle_F[p] = state.particle_F_trial[p]

        # also compute stress here
        J = wp.determinant(state.particle_F[p])
        U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sig = wp.vec3(0.0)
        stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        wp.svd3(state.particle_F[p], U, sig, V)
        if mat == 0 or mat == 5:
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )
        if mat == 1:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if mat == 2:
            stress = kirchoff_stress_drucker_prager(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if mat == 3:
            # temporarily use stvk, subject to change
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if mat == 6:  # fluid
            stress = kirchoff_stress_water(
                J, model.bulk[mat]
            )
        if mat == 10:  # newtonian / Bingham / power-law fluid (EOS + 2 eta_app dev D)
            stress = kirchoff_stress_newtonian(
                J, model.bulk[mat], state.particle_L[p], model.plastic_viscosity[mat],
                model.yield_stress[p], model.hardening[mat], model.softening[mat]
            )
        if mat == 12:  # tabulated eta_app(gd) fluid (FE-recovered curve, EOS + 2 eta_app dev D)
            stress = kirchoff_stress_tabulated(
                J, model.bulk[mat], state.particle_L[p], model.eta_table,
                model.eta_table_smin, model.eta_table_smax, model.eta_table_n
            )
        if mat == 9 or mat == 13:  # local mu(I) (parametric or tabulated): Hencky stress of returned F
            stress = kirchoff_stress_hencky(
                U, sig, model.mu[p], model.lam[p]
            )
        if mat == 11:  # compressible mu(I)-Phi(I): deviatoric Hencky + compaction pressure
            Jp_ref = state.particle_Jp[p]
            if Jp_ref < 0.5:
                Jp_ref = 1.0
            p_comp = mu_i_phi_pressure(model, mat, J, Jp_ref)
            stress = kirchoff_stress_mu_i_phi(U, sig, model.mu[p], p_comp)

        stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry
        state.particle_stress[p] = stress


@wp.kernel
def compute_cov_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    init_cov = wp.mat33(0.0)
    init_cov[0, 0] = state.particle_init_cov[p * 6]
    init_cov[0, 1] = state.particle_init_cov[p * 6 + 1]
    init_cov[0, 2] = state.particle_init_cov[p * 6 + 2]
    init_cov[1, 0] = state.particle_init_cov[p * 6 + 1]
    init_cov[1, 1] = state.particle_init_cov[p * 6 + 3]
    init_cov[1, 2] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 0] = state.particle_init_cov[p * 6 + 2]
    init_cov[2, 1] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 2] = state.particle_init_cov[p * 6 + 5]

    cov = F * init_cov * wp.transpose(F)

    state.particle_cov[p * 6] = cov[0, 0]
    state.particle_cov[p * 6 + 1] = cov[0, 1]
    state.particle_cov[p * 6 + 2] = cov[0, 2]
    state.particle_cov[p * 6 + 3] = cov[1, 1]
    state.particle_cov[p * 6 + 4] = cov[1, 2]
    state.particle_cov[p * 6 + 5] = cov[2, 2]


@wp.kernel
def compute_R_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    # polar svd decomposition
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    # compute rotation matrix
    R = U * wp.transpose(V)
    state.particle_R[p] = wp.transpose(R)

@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: float):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_v_out[grid_x, grid_y, grid_z] = (
        state.grid_v_out[grid_x, grid_y, grid_z] * scale
    )


# NOTE: E and nu are now per-type (indexed by material type, not particle).
# To use different E/nu for a region, define a new material type and use
# set_parameters_for_particles() instead.
@wp.kernel
def apply_additional_params(
    state: MPMStateStruct,
    model: MPMModelStruct,
    params_modifier: MaterialParamsModifier,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if (
        pos[0] > params_modifier.point[0] - params_modifier.size[0]
        and pos[0] < params_modifier.point[0] + params_modifier.size[0]
        and pos[1] > params_modifier.point[1] - params_modifier.size[1]
        and pos[1] < params_modifier.point[1] + params_modifier.size[1]
        and pos[2] > params_modifier.point[2] - params_modifier.size[2]
        and pos[2] < params_modifier.point[2] + params_modifier.size[2]
    ):
        state.particle_density[p] = params_modifier.density


@wp.kernel
def selection_add_impulse_on_particles(state: MPMStateStruct, impulse_modifier: Impulse_modifier):
    p = wp.tid()
    offset = state.particle_x[p] - impulse_modifier.point
    if (
        wp.abs(offset[0]) < impulse_modifier.size[0]
        and wp.abs(offset[1]) < impulse_modifier.size[1]
        and wp.abs(offset[2]) < impulse_modifier.size[2]
                ):
        impulse_modifier.mask[p] = 1 
    else:
        impulse_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_translation(state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
                ):
        velocity_modifier.mask[p] = 1 
    else:
        velocity_modifier.mask[p] = 0

        
@wp.kernel
def selection_enforce_particle_velocity_cylinder(state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point

    vertical_distance = wp.abs(wp.dot(offset, velocity_modifier.normal))

    horizontal_distance = wp.length(offset - wp.dot(offset, velocity_modifier.normal) * velocity_modifier.normal)
    if (
        vertical_distance < velocity_modifier.half_height_and_radius[0]
        and horizontal_distance < velocity_modifier.half_height_and_radius[1]
                ):
        velocity_modifier.mask[p] = 1 
    else:
        velocity_modifier.mask[p] = 0


# ---------------------------------------------------------------------------
# Rigid body kernels (material == 8)
# ---------------------------------------------------------------------------

@wp.kernel
def rigid_g2p_accumulate(
    state: MPMStateStruct,
    model: MPMModelStruct,
    rigid_x_cm: wp.array(dtype=wp.vec3),
    rigid_linear_mom: wp.array(dtype=wp.vec3),
    rigid_angular_mom: wp.array(dtype=wp.vec3),
):
    """For each rigid particle gather grid momentum and accumulate per-body
    linear and angular momentum contributions."""
    p = wp.tid()
    if state.particle_selection[p] == 0 and state.particle_material[p] == 8:
        bid = state.particle_rigid_id[p]

        # standard quadratic B-spline weights (same as g2p)
        grid_pos = state.particle_x[p] * model.inv_dx
        base_x = wp.int(grid_pos[0] - 0.5)
        base_y = wp.int(grid_pos[1] - 0.5)
        base_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(wp.float(base_x), wp.float(base_y), wp.float(base_z))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.matrix_from_cols(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )

        v_interp = wp.vec3(0.0)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weight = w[0, i] * w[1, j] * w[2, k]
                    v_interp = v_interp + state.grid_v_out[base_x + i, base_y + j, base_z + k] * weight

        mass_p = state.particle_mass[p]
        r = state.particle_x[p] - rigid_x_cm[bid]
        wp.atomic_add(rigid_linear_mom, bid, v_interp * mass_p)
        wp.atomic_add(rigid_angular_mom, bid, wp.cross(r, v_interp * mass_p))


@wp.kernel
def rigid_body_integrate(
    rigid_x_cm: wp.array(dtype=wp.vec3),
    rigid_v_cm: wp.array(dtype=wp.vec3),
    rigid_omega: wp.array(dtype=wp.vec3),
    rigid_orientation: wp.array(dtype=wp.mat33),
    rigid_mass: wp.array(dtype=float),
    rigid_inv_inertia_body: wp.array(dtype=wp.mat33),
    rigid_linear_mom: wp.array(dtype=wp.vec3),
    rigid_angular_mom: wp.array(dtype=wp.vec3),
    dt: float,
):
    """Integrate rigid body EOM for body b (one thread per body).
    Updates v_cm, omega, x_cm, and orientation."""
    b = wp.tid()
    R = rigid_orientation[b]
    M = rigid_mass[b]

    # --- linear ---
    v_cm_new = rigid_linear_mom[b] / M

    # --- angular ---
    # I_world = R * I_body * R^T,  I_world_inv = R * I_body_inv * R^T
    I_world_inv = R * rigid_inv_inertia_body[b] * wp.transpose(R)
    omega_new = I_world_inv * rigid_angular_mom[b]

    # --- position ---
    x_cm_new = rigid_x_cm[b] + v_cm_new * dt

    # --- orientation: first-order update then polar-decomp re-orthogonalisation ---
    ox = omega_new[0]
    oy = omega_new[1]
    oz = omega_new[2]
    skew_w = wp.mat33(0.0, -oz, oy,  oz, 0.0, -ox,  -oy, ox, 0.0)
    R_new = R + skew_w * R * dt
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(R_new, U, sig, V)
    R_orth = U * wp.transpose(V)

    rigid_v_cm[b] = v_cm_new
    rigid_omega[b] = omega_new
    rigid_x_cm[b] = x_cm_new
    rigid_orientation[b] = R_orth


@wp.kernel
def rigid_particle_update(
    state: MPMStateStruct,
    rigid_x_cm: wp.array(dtype=wp.vec3),
    rigid_v_cm: wp.array(dtype=wp.vec3),
    rigid_omega: wp.array(dtype=wp.vec3),
    rigid_orientation: wp.array(dtype=wp.mat33),
):
    """Set per-particle position, velocity, C, and F from rigid body state."""
    p = wp.tid()
    if state.particle_selection[p] == 0 and state.particle_material[p] == 8:
        bid = state.particle_rigid_id[p]
        R = rigid_orientation[bid]
        x_cm = rigid_x_cm[bid]
        v_cm = rigid_v_cm[bid]
        omega = rigid_omega[bid]

        x_p = x_cm + R * state.particle_x_ref[p]
        r = x_p - x_cm

        # velocity: v_cm + omega x r
        state.particle_v[p] = v_cm + wp.cross(omega, r)
        state.particle_x[p] = x_p

        # C = skew(omega) so APIC scatter uses the correct linear velocity field
        ox = omega[0]
        oy = omega[1]
        oz = omega[2]
        state.particle_C[p] = wp.mat33(0.0, -oz, oy,  oz, 0.0, -ox,  -oy, ox, 0.0)

        # rigid — no deformation
        state.particle_F[p] = wp.mat33(1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0)
