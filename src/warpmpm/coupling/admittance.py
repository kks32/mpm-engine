"""Force-feedback controllers that close the robot <-> material loop.

These turn a measured reaction force into the next tool motion, so the MATERIAL decides
where the end-effector stops instead of a scripted endpoint. This is what makes the
coupling two-way: press harder than the material can resist and it yields; press to a
target force and the tool halts at the depth where the reaction balances the command.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForceAdmittance:
    """First-order vertical admittance: regulate the tool to a target CONTACT force.

        v_down = clip((f_target - f_react) / damping, -v_max, +v_max)

    f_react is the upward reaction the material exerts on the tool (>=0 in contact). In free
    space f_react = 0 -> descend at the cap; as the material resists, v_down falls; the tool
    HALTS where f_react == f_target and then holds that force (v ~ 0 -> static boundary). If
    the material over-resists, v_down goes negative and the tool retracts, regulating back to
    f_target. The tool never reaches the floor as long as the material can supply f_target
    first; a hard z-clamp in the loop is the safety net, not the stopping mechanism.

    damping has units N/(m/s): the contact force error that produces full descent speed is
    f_target at v_max when damping = f_target / v_max.
    """

    f_target: float = 40.0
    v_max: float = 0.4
    damping: float | None = None    # N/(m/s); default f_target/v_max -> full v_max at zero force
    allow_retract: bool = True

    def _damp(self) -> float:
        # default: full descent speed at zero reaction, linearly to zero at f_target
        return self.damping if self.damping is not None else self.f_target / self.v_max

    def velocity_down(self, f_react: float) -> float:
        """Commanded downward speed (m/s, positive = descending) given the reaction force."""
        v = (self.f_target - float(f_react)) / self._damp()
        lo = -self.v_max if self.allow_retract else 0.0
        return float(np.clip(v, lo, self.v_max))


@dataclass
class Impedance1D:
    """Second-order vertical impedance for a contact tool (height z, up positive):

        m*z'' + b*z' + k*(z - z_ref) = f_react

    A finite stiffness k makes the tool YIELD to the material: with a target z_ref below the
    surface the tool presses in and settles where k*(z_eq - z_ref) = f_react(z_eq), stopping
    short of z_ref. Semi-implicit integration with implicit damping (unconditionally stable
    in b). Use when you want a compliant position target rather than a regulated force.
    """

    m: float = 1.0
    b: float = 120.0
    k: float = 4000.0
    z: float = 0.0
    zd: float = 0.0
    z_floor: float | None = None

    def step(self, dt: float, f_react: float, z_ref: float) -> tuple[float, float]:
        a = (float(f_react) - self.k * (self.z - z_ref)) / self.m
        self.zd = (self.zd + dt * a) / (1.0 + dt * self.b / self.m)  # implicit damping
        self.z += dt * self.zd
        if self.z_floor is not None and self.z < self.z_floor:
            self.z = self.z_floor
            self.zd = max(self.zd, 0.0)
        return self.z, self.zd
