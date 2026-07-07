"""warpmpm: a modular Warp-MPM engine for robot manipulation of deformable and granular
media (dough first; terrain / rovers next). Today: a dense, explicit MLS/APIC core wrapping
the validated warp-mpm fork, a kinematic box collider as the robot end-effector proxy, a
stress-integral reaction-wrench readout, and a MuJoCo coupling adapter. Sparse active-block
grid + implicit Newton-CG are the planned fast path (see README roadmap)."""
from __future__ import annotations

from warpmpm.colliders.glass import GlassProfile, cup_fill
from warpmpm.core.solver import GridConfig, Solver
from warpmpm.coupling.admittance import ForceAdmittance, Impedance1D
from warpmpm.coupling.backend import WarpMPMBackend
from warpmpm.coupling.wrench import box_contact_wrench
from warpmpm.materials import (
    Material,
    elastic,
    granular,
    newtonian,
    tabulated_viscous,
    vonmises,
)
from warpmpm.scenes import block, dough

__version__ = "0.0.1"
__all__ = [
    "ForceAdmittance",
    "GlassProfile",
    "GridConfig",
    "Impedance1D",
    "Material",
    "Solver",
    "WarpMPMBackend",
    "__version__",
    "block",
    "box_contact_wrench",
    "cup_fill",
    "dough",
    "elastic",
    "granular",
    "newtonian",
    "tabulated_viscous",
    "vonmises",
]
