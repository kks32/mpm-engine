"""SplatScene: the one-object API coupling a Gaussian-splat cloud to the MPM solver.

The scene drops near-transparent splats, fits the cloud into the grid, fills the interior
with solid particles, assigns those fillers an appearance, and loads everything into a
Solver with per-particle covariances. Positions advect with the material; each splat's
covariance deforms and its spherical harmonics rotate by the polar rotation of F, applied
at render time by inverse-rotating the view direction (the PhysGaussian trick). Opacity is
fixed; J modulation is a later option.
"""
from __future__ import annotations

import numpy as np

from warpmpm.core.solver import GridConfig, Solver

from .appearance import assign_filler_appearance, eval_sh
from .fill import fill_interior, particle_volumes
from .io import GaussianCloud
from .transforms import fit_to_grid


class SplatScene:
    def __init__(self, cloud: GaussianCloud, grid: GridConfig | None = None, material=None,
                 device: str = "auto", fill: bool = True, fill_kwargs: dict | None = None,
                 filler_appearance: str = "inherit", filler_kwargs: dict | None = None,
                 cov_mode: str = "step", opacity_threshold: float = 0.02,
                 floor="sticky", transform=None):
        self.grid = grid if grid is not None else GridConfig()
        self.cov_mode = cov_mode

        # 1. drop near-transparent splats
        keep = cloud.opacity.reshape(-1) >= opacity_threshold
        cloud = _index_cloud(cloud, keep)

        # 2. world -> sim transform
        self.transform = transform if transform is not None else fit_to_grid(cloud, self.grid)
        pos_sim = self.transform.to_sim(cloud.pos)
        cov_sim = self.transform.to_sim_cov(cloud.cov)
        sim_cloud = GaussianCloud(pos=pos_sim, cov=cov_sim, opacity=cloud.opacity,
                                  sh=cloud.sh, sh_degree=cloud.sh_degree)

        dx, n_grid = self.grid.dx, self.grid.n_grid

        # 3. interior fill (in sim space)
        if fill:
            filler_pos = fill_interior(pos_sim, cloud.opacity, cov_sim, n_grid, dx,
                                       **(fill_kwargs or {}))
        else:
            filler_pos = np.zeros((0, 3), dtype=np.float32)

        # 4. filler appearance (kNN over the sim-space originals)
        f_cov, f_op, f_sh = assign_filler_appearance(
            sim_cloud, filler_pos, mode=filler_appearance, **(filler_kwargs or {}))

        # 5. concatenate originals + fillers
        pos_all = np.concatenate([pos_sim, filler_pos], axis=0).astype(np.float32)
        cov_all = np.concatenate([cov_sim, f_cov], axis=0).astype(np.float32)
        # opacity fixed; J modulation is a later option
        op_all = np.concatenate([cloud.opacity, f_op], axis=0).astype(np.float32)
        sh_all = np.concatenate([cloud.sh, f_sh], axis=0).astype(np.float32)
        self.n_gaussians = pos_sim.shape[0]
        self.n_visible = (self.n_gaussians if filler_appearance == "invisible"
                          else pos_all.shape[0])

        # 6. volumes and solver load
        vol = particle_volumes(pos_all, n_grid, dx)
        self.solver = Solver(grid=self.grid, device=device).load_particles(
            pos_all, vol, cov=cov_all, cov_mode=cov_mode)
        if material is not None:
            self.solver.set_material(material)

        # 7. floor plane below the settled cloud
        if floor:
            min_z = float(pos_all[:, 2].min())
            self.floor_z = max(1.5 * dx, min_z - 1.0 * dx)
            surface = floor if isinstance(floor, str) else "sticky"
            self.solver.add_plane((0.0, 0.0, self.floor_z), (0.0, 0.0, 1.0), surface)

        # render-time appearance for the visible splats, on the solver device
        import torch
        dev = self.solver.device
        self._opacity_t = torch.as_tensor(op_all, dtype=torch.float32, device=dev)
        self._sh_t = torch.as_tensor(sh_all, dtype=torch.float32, device=dev)
        self.sh_degree = cloud.sh_degree

    def step(self, dt: float = 1e-4, substeps: int = 20) -> SplatScene:
        self.solver.step(dt, substeps)
        return self

    def state(self) -> dict:
        """Per-visible-splat render state as torch tensors on the solver device: pos and
        cov6 in world units (inverse transform applied on-device), plus R, opacity, sh."""
        import torch

        nv = self.n_visible
        dev = self.solver.device
        s = float(self.transform.s)
        sim_c = torch.as_tensor(self.transform.sim_center, dtype=torch.float32, device=dev)
        world_c = torch.as_tensor(self.transform.world_center, dtype=torch.float32, device=dev)

        pos_world = (self.solver.x_torch()[:nv] - sim_c) / s + world_c
        cov_world = self.solver.cov_torch()[:nv] / (s * s)
        return {
            "pos": pos_world,
            "cov6": cov_world,
            "R": self.solver.R_torch()[:nv],
            "opacity": self._opacity_t[:nv],
            "sh": self._sh_t[:nv],
        }

    def colors(self, camera_pos) -> np.ndarray:
        """RGB per visible splat from the current SH and view direction. The direction from
        camera to splat is inverse-rotated by the splat's polar rotation R (applying R^T),
        so the rest-frame SH coefficients evaluate correctly as the body rotates. This is
        the PhysGaussian view-direction trick."""
        st = self.state()
        x = st["pos"].detach().cpu().numpy().astype(np.float32)
        R = st["R"].detach().cpu().numpy().astype(np.float32)
        sh = st["sh"].detach().cpu().numpy().astype(np.float32)
        cam = np.asarray(camera_pos, dtype=np.float32)

        dirs = x - cam
        dirs = dirs / np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12, None)
        dirs_rot = np.einsum("nji,nj->ni", R, dirs)   # R^T @ dir per splat
        return eval_sh(self.sh_degree, sh, dirs_rot)


def _index_cloud(cloud: GaussianCloud, mask) -> GaussianCloud:
    scales = None if cloud.scales is None else cloud.scales[mask]
    quats = None if cloud.quats is None else cloud.quats[mask]
    return GaussianCloud(pos=cloud.pos[mask], cov=cloud.cov[mask],
                         opacity=cloud.opacity[mask], sh=cloud.sh[mask],
                         sh_degree=cloud.sh_degree, scales=scales, quats=quats)
