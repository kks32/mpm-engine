"""Per-link reaction wrench from the MPM material (Newton's third law).

v1 estimator: integrate the Cauchy traction the material exerts on a horizontal contact
layer just beneath a box end-effector (the validated squeeze-flow plate-force method:
F = -sum_contact (sigma . e_z) * vol / T_layer). Force on the gripper opposes the press
(down-press -> up reaction). A later kernel-level readout (grid impulse capture) will
replace this with a per-substep exact wrench; this stress-integral version is the
cross-validation baseline.
"""
from __future__ import annotations

import numpy as np


def box_contact_wrench(x, cauchy, vol, box_center, box_half, layer_cells: float = None,
                       dx: float = None) -> dict:
    """Reaction wrench on a box end-effector from the dough it presses.

    x        : (N,3) particle positions
    cauchy   : (N,3,3) Cauchy stress
    vol      : (N,) particle volume
    box_center, box_half : the kinematic box (centre, half-extents)
    Returns force[3], torque[3] (about box_center), contact count, normal force.
    """
    cx, cy, cz = box_center
    hx, hy, hz = box_half
    h = float(np.mean(vol) ** (1.0 / 3.0))
    T = (layer_cells * dx) if (layer_cells and dx) else 3.0 * h
    bottom = cz - hz
    under = (np.abs(x[:, 0] - cx) < hx) & (np.abs(x[:, 1] - cy) < hy)
    contact = under & (x[:, 2] > bottom - T) & (x[:, 2] < bottom + 0.5 * T)
    if contact.sum() < 3:
        return {"force": np.zeros(3), "torque": np.zeros(3), "n_contact": 0, "Fz": 0.0}
    # traction the material exerts on a downward-facing box bottom is -(sigma . e_z);
    # reaction on the box = -that, integrated as (vol / T) over the contact layer
    trac = cauchy[contact][:, :, 2]                      # (n,3) = sigma . e_z
    f_per = -trac * (vol[contact, None] / T)             # force the box feels, per particle
    force = f_per.sum(0)
    r = x[contact] - np.array(box_center)
    torque = np.cross(r, f_per).sum(0)
    return {"force": force, "torque": torque, "n_contact": int(contact.sum()),
            "Fz": float(force[2])}
