"""Warp MLS-MPM kernels: the numerical core of the engine.

This subpackage holds the explicit MLS-MPM solver (`mpm_solver_warp`), the constitutive
stress and return-mapping kernels (`mpm_utils`), and the model/state structs (`warp_utils`).
It began as the UCLA warp-mpm of Zeshun Zong et al. (the base solver, the quadratic
B-spline transfer, and materials 0-8) and was extended by this group (UT Austin): moving
robot colliders with a velocity boundary, a Newton-exact grid-impulse contact force,
a 6-DoF kinematic revolved-SDF glass collider (separable Coulomb contact + wrench/torque
readout, for pouring), multi-material scenes, a point-cloud loader, and the mu(I) /
viscoplastic constitutive models (materials 9-13). See AUTHORS.md for attribution and
citation.

This is the single import surface for the kernels: the typed `core.Solver` wrapper and the
`sim/` scenes import `MPM_Simulator_WARP` (and the low-level structs) from here.
"""
from __future__ import annotations

from warpmpm.kernels.mpm_solver_warp import MATERIAL_NAME_TO_ID, MPM_Simulator_WARP
from warpmpm.kernels.warp_utils import MPMModelStruct, MPMStateStruct  # low-level structs (sim/ probes)

__all__ = [
    "MATERIAL_NAME_TO_ID",
    "MPMModelStruct",
    "MPMStateStruct",
    "MPM_Simulator_WARP",
]
