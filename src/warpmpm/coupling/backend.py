"""WarpMPMBackend: the stable coupling surface between a robot sim and the MPM material.

The robot sim owns the robot's dynamics; this backend owns the material and exchanges only
compact kinematics (in) and reaction wrenches (out) per contact tool -- never particles.
This is the contract MuJoCo plugs into now and Isaac Lab plugs into later, unchanged.

A "tool" is a kinematic box end-effector. It is driven by the START-of-tick centre plus a
per-tick velocity: the fork's modify_bc advances point += dt*velocity on every substep, so
over one control tick the box sweeps centre -> centre + dt_ctrl*velocity and lands exactly
on the commanded target. The imposed grid velocity is what presses the material; a tool at
rest (velocity 0) acts as a static no-slip boundary that holds the material. The reaction
wrench is the compressive-gated stress integral (coupling/wrench.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from warpmpm.core.solver import Solver
from warpmpm.coupling.wrench import box_contact_wrench


@dataclass
class WarpMPMBackend:
    """Owns one Solver; exposes attach_tool / set_tool_kinematics / step / get_tool_wrench."""

    solver: Solver
    _tools: dict[int, dict[str, Any]] = field(default_factory=dict)

    # --- tool lifecycle --------------------------------------------------------------
    def attach_tool(self, center, half, velocity=(0.0, 0.0, 0.0)) -> int:
        """Add a kinematic box end-effector; returns a tool id."""
        tid = self.solver.add_box(center, half, velocity)
        self._tools[tid] = {"half": tuple(map(float, half)), "center": tuple(map(float, center))}
        return tid

    def set_tool_kinematics(self, tool_id: int, center, velocity) -> None:
        """Command a tool with its START-of-tick centre and per-tick velocity (the contract
        in set_box: modify_bc integrates centre -> centre + dt_ctrl*velocity over the step)."""
        self.solver.set_box(tool_id, center=center, velocity=velocity)
        self._tools[tool_id]["center"] = tuple(map(float, center))

    # --- exchange --------------------------------------------------------------------
    def step(self, dt: float, substeps: int = 1) -> None:
        self.solver.step(dt, substeps)

    def get_tool_wrench(self, tool_id: int, at_center=None, **kw) -> dict:
        """Quasi-STATIC contact wrench from the stress integral over the box, for the force
        CONTROLLER (admittance/impedance) feedback. This and the grid impulse measure DIFFERENT
        things and must not be conflated: the grid impulse (reset_tool_force/get_tool_reaction)
        is the Newton-exact DYNAMIC reaction, F = sum m (v_free - v_imposed)/dt, which is the
        calibrated lever for the moving-contact IDENTIFICATION (the squeeze, the shear cell);
        but at a quasi-static halt v_free -> v_imposed so that impulse collapses, while the
        static contact force the controller must regulate against is still large and is what
        this stress integral reports. Hence: grid impulse for moving-contact identification,
        this wrench for static-contact force control. Uncalibrated (biases the stress estimate),
        but the right signal for a halt; evaluate at the post-step centre (at_center)."""
        t = self._tools[tool_id]
        c = tuple(map(float, at_center)) if at_center is not None else t["center"]
        return box_contact_wrench(
            self.solver.x(), self.solver.cauchy(), self.solver.vol(), c, t["half"], **kw
        )

    def reset_tool_force(self, tool_id: int) -> None:
        """Zero the tool's exact reaction-impulse accumulator (call before step())."""
        self.solver.reset_tool_force(tool_id)

    def get_tool_reaction(self, tool_id: int, dt: float):
        """Newton-EXACT reaction force on the tool from the collider grid impulse accumulated
        since the last reset_tool_force, over elapsed time dt. Returns force[3] (compression
        -> +z). Calibrated: no contact band, no T_layer, no gating."""
        return self.solver.tool_force(tool_id, dt)
