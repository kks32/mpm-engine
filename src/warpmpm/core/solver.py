"""Device-auto Solver: a small typed wrapper over the warp-mpm fork.

Centralizes device handling, Warp init, and the common load/material/collider/step/export
calls, so scenes, tests, and the coupling backend never touch the raw fork or sys.path.
The default ``device="auto"`` resolves to ``cuda:0`` when a CUDA GPU is present and to
``cpu`` otherwise (Apple Silicon); pass ``device="cuda:1"`` or ``device="cpu"`` to pin.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from warpmpm.kernels import MPM_Simulator_WARP

_INITED = False
_DEVICE_ANNOUNCED = False   # print the auto-resolved device once per process


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
    """Thin owner of one MPM_Simulator_WARP instance; device resolves at load time."""

    grid: GridConfig = field(default_factory=GridConfig)
    device: str = "auto"
    # cadence (in control ticks) of the particle-box update + grid-edge guard. Both need a
    # device-to-host readback of x and v, which is free on CPU but a pipeline sync on CUDA;
    # GPU runs (where the captured graphs sweep the full grid and ignore the box anyway) can
    # raise this to amortize the sync, at the cost of the edge guard firing that much later.
    guard_interval: int = 1
    # active-block sparse compute: grid sweeps run over 4^3 blocks containing material
    # (rebuilt each tick) instead of the dense grid or its bounding box. Wins when the
    # occupied region is not box-shaped (separated bodies, spread fluid, large empty
    # domains). Storage stays dense; takes precedence over the CUDA-graph fast path.
    sparse: bool = False
    # claymore-style fused particle pass (docs/claymore_notes.md): interior substeps of
    # a tick run ONE g2p+stress+p2g kernel instead of three particle passes (S+1 passes
    # per S substeps instead of 3S). DEFAULT: bitwise-equal to the three-pass pipeline
    # (verified across material families incl. wrench readout) and faster everywhere
    # measured (13% on the 192^3 GPU pour, ~4% CPU). Falls back silently per tick when
    # the scene uses features the fused path excludes (rigid bodies, particle
    # modifiers, sparse mode). fused=False restores the three-pass pipeline, which is
    # also the only path with CUDA graph capture (graphs only pay on small scenes).
    fused: bool = True
    # claymore-style block sort (5a): every `sort_interval` ticks, reorder particles by
    # their 4^3 grid block so P2G atomics from neighboring threads hit neighboring
    # nodes and G2P gathers coalesce (the locality AoSoA buys, in SoA layout). 0 = off.
    # WARNING: particle index identity changes at sort ticks; keep 0 for runs whose
    # dumps pair frames by particle index (trajectory-based identification).
    sort_interval: int = 0
    # per-phase substep profiling: syncs the device around every kernel phase and
    # accumulates timings (zero/stress/p2g/grid_update/BC/g2p). Forces live launches
    # (a captured graph cannot be timed per phase), so a profiled run is slower;
    # the SHARES are the signal. Read the result with profile_report().
    profile: bool = False
    _sim: Any = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)
    _tick: int = field(default=0, init=False, repr=False)
    _vol0: Any = field(default=None, init=False, repr=False)

    def load_particles(self, pos: np.ndarray, vol: np.ndarray) -> Solver:
        import torch

        _ensure_warp()
        if self.device == "auto":
            self.device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
            global _DEVICE_ANNOUNCED
            if not _DEVICE_ANNOUNCED:
                _DEVICE_ANNOUNCED = True
                print(f"warpmpm: device auto -> {self.device}")
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
        self._sim._bc_box_cache = {}
        if center is not None:
            p.point = wp.vec3(float(center[0]), float(center[1]), float(center[2]))
        if velocity is not None:
            p.velocity = wp.vec3(float(velocity[0]), float(velocity[1]), float(velocity[2]))
        return self

    # --- kinematic glass (revolved-SDF cup collider) ----------------------------------
    def add_cup(self, profile, center, quat=(1.0, 0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
                omega=(0.0, 0.0, 0.0), friction: float = 0.05, sticky_cells: float = 1.5,
                contact_cells: float = 0.5, start_time: float = 0.0,
                end_time: float = 1.0e9) -> int:
        """A kinematic open-top glass collider (analytic revolved SDF; profile is a
        colliders.glass.GlassProfile) at pose (center, wxyz quat), imposing its rigid
        velocity field on the grid: separable Coulomb-friction contact from contact_cells*dx
        OUTSIDE the surface (approach is stopped before material can creep into the wall)
        down to sticky_cells*dx inside it, full grab deeper (anti-tunneling backstop).
        Returns a handle; drive it each control tick with set_cup. Accumulates the
        Newton-exact reaction impulse AND torque; read with cup_wrench after step()."""
        from warpmpm.colliders.glass import quat_to_mat

        # the sticky core must survive inside the wall: cap the friction shell at just
        # under half the wall thickness so coarse grids keep an anti-tunneling backstop
        sticky_depth = min(sticky_cells * self.grid.dx, 0.45 * profile.wall_thickness)
        return self._sim.add_revolved_sdf_collider(
            point=tuple(center), rot=quat_to_mat(quat), velocity=tuple(velocity),
            omega=tuple(omega), outer_radius=profile.outer_radius,
            inner_radius=profile.inner_radius, half_height=profile.half_height,
            inner_floor_z=profile.inner_floor_z, fillet_radius=profile.fillet_radius,
            friction=friction, sticky_depth=sticky_depth,
            contact_band=contact_cells * self.grid.dx,
            start_time=start_time, end_time=end_time,
        )

    def set_cup(self, handle: int, center=None, quat=None, velocity=None, omega=None) -> Solver:
        """Update a cup's pose/velocities. Same contract as set_box, extended to rotation:
        pass the START-of-tick pose plus per-tick velocities (v, omega); modify_bc sweeps
        the cup to the commanded end-of-tick pose over the substeps."""
        from warpmpm.colliders.glass import quat_to_mat

        rot = None if quat is None else quat_to_mat(quat)
        self._sim.set_revolved_collider_pose(handle, point=center, rot=rot,
                                             velocity=velocity, omega=omega)
        return self

    def reset_cup_wrench(self, handle: int) -> Solver:
        """Zero a cup collider's reaction impulse + torque accumulators (call before the
        substeps you want to integrate the wrench over)."""
        p = self._sim.collider_params[handle]
        p.force.zero_()
        p.torque.zero_()
        return self

    def cup_wrench(self, handle: int, dt: float) -> dict:
        """Newton-exact reaction wrench the material exerts on a cup collider, from the
        grid impulse accumulated since the last reset over elapsed time dt: force[3] and
        torque[3] (about the cup centre). A static cup holding m kg of settled liquid
        reads force ~ (0, 0, -m*g) -- the liquid's weight pressing on the glass."""
        p = self._sim.collider_params[handle]
        return {
            "force": np.asarray(p.force.numpy()[0], dtype=float) / dt,
            "torque": np.asarray(p.torque.numpy()[0], dtype=float) / dt,
        }

    def add_domain_walls(self, start_time: float = 0.0, end_time: float = 1.0e9) -> Solver:
        """Zero outward grid velocity in a 3-cell band at the domain faces, so splashes
        can never advect particles out of [0, grid_lim]^3 (out-of-domain particles would
        index the grid out of bounds in p2g)."""
        self._sim.add_bounding_box(start_time=start_time, end_time=end_time)
        return self

    def add_sdf_collider(self, sdf, center, quat=(0.0, 0.0, 0.0, 1.0),
                         velocity=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0), band=None,
                         surface: str = "separable", friction: float = 0.4,
                         start_time: float = 0.0, end_time: float = 1.0e9) -> int:
        """Add a watertight mesh as a moving/rotating signed-distance-field collider. `sdf` is
        a warpmpm.geometry.SDFData (built from a mesh). Drive its pose each control tick with
        set_sdf_pose; read the reaction wrench with sdf_wrench. Returns a handle. This is the
        general (arbitrary-mesh, oriented) counterpart to add_box for the coupling layer."""
        return self._sim.add_sdf_collider(
            sdf.values, sdf.grads, sdf.origin, sdf.cell, center, quat=quat, velocity=velocity,
            omega=omega, band=band, surface=surface, friction=friction,
            start_time=start_time, end_time=end_time, device=self.device,
        )

    def set_sdf_pose(self, handle: int, center=None, quat=None, velocity=None, omega=None
                     ) -> Solver:
        """Update an SDF collider's pose/velocity/angular-velocity (called each control tick).
        Like set_box, pass the START-of-tick center/quat and the per-tick velocity/omega; the
        fork integrates center += dt*velocity and rotates the quat by omega on every substep."""
        self._sim.set_sdf_pose(handle, center=center, quat=quat, velocity=velocity, omega=omega)
        return self

    def reset_sdf_force(self, handle: int) -> Solver:
        """Zero an SDF collider's reaction force + torque accumulators (call before step())."""
        self._sim.collider_params[handle].force.zero_()
        self._sim.collider_params[handle].torque.zero_()
        return self

    def sdf_wrench(self, handle: int, dt: float) -> dict:
        """Newton-exact reaction WRENCH the material exerts on an SDF collider, from the grid
        impulse accumulated since the last reset: force = sum m*(v_free - v_new) / dt, torque =
        sum (x - center) x impulse / dt (about the collider centre, world frame). Returns
        {'force': (3,), 'torque': (3,)}. The general 6-DOF analogue of tool_force for the box."""
        f = np.asarray(self._sim.collider_params[handle].force.numpy()[0], dtype=float)
        t = np.asarray(self._sim.collider_params[handle].torque.numpy()[0], dtype=float)
        return {"force": f / dt, "torque": t / dt}

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
        if self._tick % max(1, self.guard_interval) == 0:
            self._update_grid_box(dt, substeps)
        if self.sort_interval and self._tick % self.sort_interval == 0:
            self._sort_particles()
        if self.sparse:
            self._sim.rebuild_active_blocks(self.device)
        self._sim.profile = self.profile
        self._tick += 1
        fused_ok = (self.fused and not self.sparse
                    and not self._sim.pre_p2g_operations
                    and not self._sim.particle_velocity_modifiers
                    and self._sim.n_rigid_bodies == 0)
        if fused_ok:
            self._sim.p2g2p_fused_tick(dt, substeps, device=self.device)
            self._step += substeps
        else:
            for _ in range(substeps):
                self._sim.p2g2p(self._step, dt, device=self.device)
                self._step += 1
        return self

    def _sort_particles(self) -> bool:
        """Block-key sort (stable): key = lexicographic 4^3-block index of the stencil
        base. Skips the permutation when already ordered (particles move well under a
        cell per tick, so most sort ticks after the first are no-ops)."""
        x = self.x()
        base = (np.floor(x / self.grid.dx - 0.5).astype(np.int64)) >> 2
        nb = (self.grid.n_grid >> 2) + 2
        keys = (base[:, 0] * nb + base[:, 1]) * nb + base[:, 2]
        if np.all(np.diff(keys) >= 0):
            return False
        perm = np.argsort(keys, kind="stable")
        self._sim.permute_particles(perm, device=self.device)
        self._vol0 = self._vol0[perm]  # host-side pair of the device arrays
        return True

    def profile_report(self) -> str:
        """Aggregate the per-phase timings collected while profile=True into a table
        (total seconds, ms per substep, share of timed device work)."""
        prof = getattr(self._sim, "time_profile", {}) or {}
        rows = [(k, sum(v) / 1000.0, len(v)) for k, v in prof.items() if v]
        if not rows:
            return "profile_report: no samples (set solver.profile = True and step)"
        timed = sum(t for _, t, _ in rows)
        rows.sort(key=lambda r: -r[1])
        lines = [f"substep profile over {self._step} substeps "
                 f"(timed device work {timed:.1f}s; live launches + per-phase sync):"]
        for name, tot, n in rows:
            lines.append(f"  {name:<28s} {tot:7.1f}s  {tot / max(n, 1) * 1000:8.3f} ms/substep"
                         f"  {tot / timed * 100:5.1f}%")
        return "\n".join(lines)

    def _update_grid_box(self, dt: float, substeps: int) -> None:
        """Once per control tick: (a) guard against particles reaching the grid edge,
        where the quadratic-stencil P2G scatter would write out of bounds (silent memory
        corruption); (b) set the live particle box, padded for this tick's motion, that
        the zero/normalize/damping sweeps launch over instead of the full dense grid."""
        x = self.x()
        v = self.v()
        dx = self.grid.dx
        lim = self.grid.grid_lim
        if x.min() < 1.5 * dx or x.max() > lim - 2.5 * dx:
            raise RuntimeError(
                f"particles within 2 cells of the grid edge (x in "
                f"[{x.min():.4f}, {x.max():.4f}] m, domain [0, {lim}] m, dx={dx:.4f}): "
                f"the P2G stencil would write out of bounds. Enlarge grid_lim or add a "
                f"bounding box / wall collider.")
        pad = 3.0 * dx + 1.5 * float(np.abs(v).max()) * dt * substeps
        self._sim.grid_launch_box = self._sim._grid_box(x.min(0) - pad, x.max(0) + pad,
                                                        halo=0)
    # --- imports (numpy, off the hot path; e.g. the leak-projection rescue net) -------
    def set_x(self, pos: np.ndarray) -> Solver:
        import torch

        self._sim.import_particle_x_from_torch(
            torch.from_numpy(np.ascontiguousarray(pos, dtype=np.float32)), device=self.device
        )
        return self

    def set_v(self, vel: np.ndarray) -> Solver:
        import torch

        self._sim.import_particle_v_from_torch(
            torch.from_numpy(np.ascontiguousarray(vel, dtype=np.float32)), device=self.device
        )
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

    def inverted_count(self) -> int:
        """Particles with non-positive det(F) (inverted or degenerate). vol() and cauchy()
        take |det F| so volume and stress stay finite, which hides inversion; call this to
        detect a nonphysical state (raise substeps or soften the contact if it is nonzero)."""
        return int((np.linalg.det(self.F()) <= 0.0).sum())

    def _warn_if_inverted(self, J: np.ndarray) -> None:
        n = int((J <= 0.0).sum())
        if n and not getattr(self, "_warned_inverted", False):
            self._warned_inverted = True
            warnings.warn(f"{n} particles have det(F) <= 0 (inverted); |det F| is used so "
                          "volume/stress stay finite but the state is nonphysical "
                          "(see Solver.inverted_count()).", RuntimeWarning, stacklevel=2)

    def vol(self) -> np.ndarray:
        """Current particle volume V0 * |det(F)| (Cauchy stress = Kirchhoff / det F)."""
        J = np.linalg.det(self.F())
        self._warn_if_inverted(J)
        return self._vol0 * np.abs(J)

    def cauchy(self) -> np.ndarray:
        """Cauchy stress per particle = Kirchhoff (exported) / |det(F)|."""
        J = np.linalg.det(self.F())
        self._warn_if_inverted(J)
        return self.stress() / np.clip(np.abs(J), 1e-9, None)[:, None, None]

    @property
    def n_particles(self) -> int:
        return 0 if self._sim is None else self._sim.n_particles
