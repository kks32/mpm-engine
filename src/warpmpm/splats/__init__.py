"""Gaussian-splat scenes on the warpmpm engine.

Load or synthesize a splat cloud, fit it into the grid, fill the interior with solid
particles, and simulate: positions advect with the material, covariances deform, and the
spherical harmonics rotate by the polar rotation of the deformation gradient. plyfile and
scipy are needed only for PLY io and filler kNN, and are imported inside the functions
that use them, so this package imports without the splats extra installed.
"""
from __future__ import annotations

from .appearance import assign_filler_appearance, eval_sh
from .bake import Baked4DSplats, bake
from .export import (
    FrameRecorder,
    convert_to_sog,
    cov6_to_scale_quat,
    export_frame_ply,
    rotate_sh,
)
from .fill import fill_interior, particle_volumes
from .io import (
    GaussianCloud,
    load_gaussians_ply,
    make_synthetic_cloud,
    save_gaussians_ply,
)
from .scene import SplatScene
from .transforms import SimTransform, fit_to_grid

__all__ = [
    "Baked4DSplats",
    "FrameRecorder",
    "GaussianCloud",
    "SimTransform",
    "SplatScene",
    "assign_filler_appearance",
    "bake",
    "convert_to_sog",
    "cov6_to_scale_quat",
    "eval_sh",
    "export_frame_ply",
    "fill_interior",
    "fit_to_grid",
    "load_gaussians_ply",
    "make_synthetic_cloud",
    "particle_volumes",
    "rotate_sh",
    "save_gaussians_ply",
]
