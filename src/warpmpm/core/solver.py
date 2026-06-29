"""CPU-default Solver: a small typed wrapper over the warp-mpm fork.

Centralizes device handling (the fork hardcodes cuda in many signatures; on Apple
Silicon we run CPU), Warp init, and the common load/material/collider/step/export calls,
so scenes, tests, and the coupling backend never touch the raw fork or sys.path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from warpmpm.kernels import MPM_Simulator_WARP

_INITED = False


def _ensure_warp() -> None:
    global _INITED
    if not _INITED:
        wp.config.quiet = True
        wp.init()
        _INITED = True


@dataclass
class GridConfig:
    n_grid: int = 64
    grid_lim: float = 0.4  # cubic domain edge, metres

    @property
    def dx(self) -> float:
        return self.grid_lim / self.n_grid


@dataclass
class Solver:
    """Thin owner of one MPM_Simulator_WARP instance, CPU by default."""

    grid: GridConfig = field(default_factory=GridConfig)
    device: str = "cpu"
    _sim: Any = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)
    _vol0: Any = field(default=None, init=False, repr=False)

    def load_particles(self, pos: np.ndarray, vol: np.ndarray) -> Solver:
        import torch

        _ensure_warp()
        self._vol0 = vol.astype(np.float32).copy()
        self._sim = MPM_Simulator_WARP(len(pos), device=self.device)
        self._sim.load_initial_data_from_torch(
            torch.from_numpy(pos.astype(np.float32)),
            torch.from_numpy(vol.astype(np.float32)),
            n_grid=self.grid.n_grid,
            grid_lim=self.grid.grid_lim,
            device=self.device,
        )
        return self

    def set_material(self, material, **overrides: float) -> Solver:
        """Accepts a composable warpmpm.materials.Material (preferred) or a fork material
        name string. Resolves to the fork's (name, params) and applies it."""
        if hasattr(material, "resolve"):
            name, params = material.resolve()
        else:
            name, params = str(material), {}
        params = {**params, **overrides}
        self._sim.set_parameters_dict(
            {"material": name, "g": [0.0, 0.0, -9.81], **params}, device=self.device
        )
        self._sim.finalize_mu_lam(device=self.device)
        return self

    def add_plane(self, point, normal, surface: str = "sticky", friction: float = 0.0) -> Solver:
        self._sim.add_surface_collider(tuple(point), tuple(normal), surface, friction=friction)
        return self

    def add_box(self, center, half_size, velocity=(0.0, 0.0, 0.0),
                start_time: float = 0.0, end_time: float = 1.0e9) -> int:
        """A kinematic box collider (axis-aligned box-SDF) imposing its velocity on the
        grid nodes it covers. Returns a handle; drive it each control tick with set_box.
        This is the robot end-effector proxy for the coupling layer."""
        self._sim.set_velocity_on_cuboid(
            point=tuple(center), size=tuple(half_size), velocity=tuple(velocity),
            start_time=start_time, end_time=end_time,
        )
        return len(self._sim.collider_params) - 1

    def set_box(self, handle: int, center=None, velocity=None) -> Solver:
        """Update a kinematic box's pose/velocity (called each control tick from the robot
        end-effector). The fork's modify_bc advances point += dt*velocity on EVERY substep,
        so over one tick the box sweeps center -> center + dt_ctrl*velocity. Drive it with
        the START-of-tick center and the per-tick velocity (vz = (target - prev)/dt_ctrl);
        the box then lands exactly on target by the end of the step. Passing the end-of-tick
        target as center double-applies the motion and leaves the box one tick ahead."""
        p = self._sim.collider_params[handle]
        if center is not None:
            p.point = wp.vec3(float(center[0]), float(center[1]), float(center[2]))
        if velocity is not None:
            p.velocity = wp.vec3(float(velocity[0]), float(velocity[1]), float(velocity[2]))
        return self

    def reset_tool_force(self, handle: int) -> Solver:
        """Zero a box collider's Newton-exact reaction-impulse accumulator. Call before the
        substeps you want to integrate the force over (typically before step())."""
        self._sim.collider_params[handle].force.zero_()
        return self

    def tool_force(self, handle: int, dt: float) -> np.ndarray:
        """Reaction force the material exerts on a box collider, from the EXACT grid impulse
        accumulated since the last reset: F = sum_substeps sum_nodes m*(v_free - v_imposed) /
        dt. dt is the elapsed time accumulated over (e.g. substeps*substep_dt). Returns
        force[3] (compression -> +z). This is the calibrated alternative to the stress
        integral; no contact band, no T_layer, no gating."""
        impulse = self._sim.collider_params[handle].force.numpy()[0]
        return np.asarray(impulse, dtype=float) / dt

    def step(self, dt: float, substeps: int = 1) -> Solver:
        for _ in range(substeps):
            self._sim.p2g2p(self._step, dt, device=self.device)
            self._step += 1
        return self

    # --- exports (numpy, off the hot path) -------------------------------------------
    def x(self) -> np.ndarray:
        return self._sim.export_particle_x_to_torch().cpu().numpy()

    def v(self) -> np.ndarray:
        return self._sim.export_particle_v_to_torch().cpu().numpy()

    def F(self) -> np.ndarray:
        return self._sim.export_particle_F_to_torch().cpu().numpy().reshape(-1, 3, 3)

    def stress(self) -> np.ndarray:
        return self._sim.export_particle_stress_to_torch().cpu().numpy().reshape(-1, 3, 3)

    def L(self) -> np.ndarray:
        """Per-particle velocity gradient L_ij = dv_i/dx_j from the most recent G2P. Use the
        symmetric part D = sym(L) for the strain rate (and |gamma_dot| = sqrt(2 D:D + eps^2))."""
        return self._sim.export_particle_L_to_torch().cpu().numpy().reshape(-1, 3, 3)

    def vol(self) -> np.ndarray:
        """Current particle volume V0 * det(F) (Cauchy stress = Kirchhoff / det F)."""
        J = np.abs(np.linalg.det(self.F()))
        return self._vol0 * J

    def cauchy(self) -> np.ndarray:
        """Cauchy stress per particle = Kirchhoff (exported) / det(F)."""
        J = np.clip(np.abs(np.linalg.det(self.F())), 1e-9, None)
        return self.stress() / J[:, None, None]

    @property
    def n_particles(self) -> int:
        return 0 if self._sim is None else self._sim.n_particles
