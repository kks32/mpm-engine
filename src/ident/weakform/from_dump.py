"""Build in-plane weak-form FrameData from a validated oracle dump.

Bridges ident/io/schema.py (raw 3D dump) to ident/weakform/assembly.py
(in-plane FrameData). Pure ident-internal: no warp/torch/sim imports.

Conventions:
  - in-plane axes (x, z) are read from the dump metadata; the in-plane
    velocity-gradient block L[(x,z),(x,z)] gives D = sym(L_ip) and
    |gamma_dot|_eps = sqrt(2 D:D + eps^2), matching the assembly. For ideal
    plane strain (D_yy = D_xy = D_zy = 0) this equals the 3D norm, so the
    in-plane restriction of the mu(I) law is exact; the realized departure
    from plane strain is reported as a diagnostic.
  - pressure is the TRUE 3D Cauchy trace (pressure_from_cauchy_3d_trace).
  - material acceleration is the per-particle trajectory finite difference of
    the dumped velocity (particles are material points), the identification
    default. Central differences interior, one-sided at the ends.
  - density per frame is mass / current volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.conventions import (
    EPS_GAMMA_DEFAULT,
    equivalent_shear_rate,
    inertial_number,
    pressure_from_cauchy_3d_trace,
    sym,
)
from ident.io.schema import Dump
from ident.masks.flowing import (
    flowing_mask,
    gate_transient_cutoff,
    validity_mask_quasi2d,
)
from ident.weakform.assembly import FrameData


@dataclass
class FramesBundle:
    frames: list[FrameData]
    times: np.ndarray
    eps_gamma: float
    plane_strain_residual: float   # median |D_yy| / |gamma_dot|_3D over flowing
    gate_cut: float
    n_rows_valid_total: int        # total valid particle-frames after masking
    I_observed: tuple[float, float]


def build_inplane_frames(
    dump: Dump,
    eps_gamma: float = EPS_GAMMA_DEFAULT,
    gate_clearance_time: float = 0.0,
    closure_spike_time: float = 0.0,
) -> FramesBundle:
    meta = dump.meta
    ax, az = meta.in_plane_axes
    out_ax = meta.out_of_plane_axis
    d, rho_s = meta.grain_diameter, meta.rho_s
    frame_dt = meta.frame_dt
    F = meta.n_frames
    cfg = meta.extra.get("config", {})
    resolution_dx = (
        cfg["grid_lim"] / cfg["n_grid"] if "grid_lim" in cfg and "n_grid" in cfg else None
    )

    x_ip = dump.x[..., [ax, az]]               # (F, P, 2)
    v_ip = dump.v[..., [ax, az]]               # (F, P, 2)
    L_ip = dump.L[:, :, [ax, az]][:, :, :, [ax, az]]   # (F, P, 2, 2)
    D_ip = sym(L_ip)
    gd = equivalent_shear_rate(D_ip, eps_gamma)        # (F, P)
    p = pressure_from_cauchy_3d_trace(dump.stress)     # (F, P)
    I = inertial_number(gd, p, d, rho_s)               # (F, P)
    rho = dump.mass[None, :] / np.maximum(dump.volume, 1e-30)

    # trajectory (material) acceleration by finite difference of dumped v
    a_ip = np.zeros_like(v_ip)
    if F >= 3:
        a_ip[1:-1] = (v_ip[2:] - v_ip[:-2]) / (2.0 * frame_dt)
    a_ip[0] = (v_ip[1] - v_ip[0]) / frame_dt
    a_ip[-1] = (v_ip[-1] - v_ip[-2]) / frame_dt

    gate_cut = gate_transient_cutoff(d, gate_clearance_time, closure_spike_time)

    # plane-strain residual: |D_yy| relative to the 3D shear-rate norm
    L3 = dump.L
    D3 = 0.5 * (L3 + np.swapaxes(L3, -1, -2))
    gd3 = equivalent_shear_rate(D3, eps_gamma)
    flow_any = flowing_mask(gd3, I) & dump.active
    if np.any(flow_any):
        ps_res = float(np.median(np.abs(D3[..., out_ax, out_ax][flow_any]) / gd3[flow_any]))
    else:
        ps_res = float("nan")

    frames: list[FrameData] = []
    n_valid_total = 0
    I_collect: list[np.ndarray] = []
    for f in range(F):
        flow = flowing_mask(gd[f], I[f])
        valid = validity_mask_quasi2d(
            x_ip[f, :, 0], x_ip[f, :, 1], flow, d, resolution_dx=resolution_dx
        )
        mask = dump.active[f] & flow & valid & (p[f] > 0.0) & np.isfinite(I[f])
        if dump.times[f] < gate_cut:
            mask = np.zeros_like(mask)
        n_valid_total += int(mask.sum())
        if np.any(mask):
            I_collect.append(I[f][mask])
        frames.append(
            FrameData(
                t=float(dump.times[f]),
                x=x_ip[f],
                v=v_ip[f],
                D=D_ip[f],
                a=a_ip[f],
                p=p[f],
                I=I[f],
                vol=dump.volume[f],
                rho=rho[f],
                mask=mask,
            )
        )
    # well-sampled I band: 5th-90th percentile over valid particle-frames. The
    # upper tail (fast, dilute, low-pressure surface material) is sparsely
    # sampled and unreliable for identification, so it is excluded from the
    # reported coverage and from the eval/constraint band used downstream
    if I_collect:
        allI = np.concatenate(I_collect)
        I_lo = float(np.percentile(allI, 5))
        I_hi = float(np.percentile(allI, 90))
    else:
        I_lo, I_hi = 0.0, 0.0
    return FramesBundle(
        frames=frames,
        times=dump.times,
        eps_gamma=eps_gamma,
        plane_strain_residual=ps_res,
        gate_cut=gate_cut,
        n_rows_valid_total=n_valid_total,
        I_observed=(I_lo, I_hi),
    )


def in_plane_sigma_frames(dump: Dump) -> list[np.ndarray]:
    """In-plane Cauchy 2x2 blocks per frame, for the closure diagnostic."""
    ax, az = dump.meta.in_plane_axes
    blk = dump.stress[:, :, [ax, az]][:, :, :, [ax, az]]
    return [blk[f] for f in range(dump.meta.n_frames)]
