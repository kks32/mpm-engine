"""Preview rendering for splat scenes.

preview_frame is a rasterizer-free matplotlib scatter, enough to watch the motion and the
colors evolve with no dependency beyond matplotlib. rasterize_inria is an optional hook to
the INRIA differentiable rasterizer, kept behind a guarded import: that package carries a
separate non-commercial license and is never a dependency of this repo.
"""
from __future__ import annotations

import numpy as np

from .appearance import eval_sh


def _to_numpy(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def _colors_from_state(state, camera_pos, sh_degree) -> np.ndarray:
    x = _to_numpy(state["pos"]).astype(np.float32)
    R = _to_numpy(state["R"]).astype(np.float32)
    sh = _to_numpy(state["sh"]).astype(np.float32)
    dirs = x - np.asarray(camera_pos, np.float32)
    dirs = dirs / np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12, None)
    dirs_rot = np.einsum("nji,nj->ni", R, dirs)
    return eval_sh(sh_degree, sh, dirs_rot)


def default_camera(state, elev: float = 18.0, azim: float = -60.0) -> dict:
    """A viewpoint derived from the current splat bounding box: camera placed off one
    corner, looking at the center."""
    x = _to_numpy(state["pos"])
    lo, hi = x.min(0), x.max(0)
    center = 0.5 * (lo + hi)
    span = float(np.max(hi - lo)) + 1e-6
    pos = center + np.array([1.6, -1.8, 1.2]) * span
    return {"pos": pos.astype(np.float32), "elev": elev, "azim": azim,
            "center": center.astype(np.float32), "span": span}


def preview_frame(state, camera: dict | None = None, path=None, colors=None,
                  sh_degree: int | None = None):
    """Matplotlib 3D scatter of splat centers, colored by their SH color for the camera and
    sized by mean sigma. Returns the RGB array plotted; writes a PNG when path is given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if camera is None:
        camera = default_camera(state)
    if sh_degree is None:
        sh_degree = round(_to_numpy(state["sh"]).shape[1] ** 0.5) - 1
    if colors is None:
        colors = _colors_from_state(state, camera["pos"], sh_degree)
    colors = np.clip(_to_numpy(colors), 0.0, 1.0)

    x = _to_numpy(state["pos"]).astype(np.float32)
    cov6 = _to_numpy(state["cov6"]).astype(np.float64)
    sigma = np.sqrt(np.clip((cov6[:, 0] + cov6[:, 3] + cov6[:, 5]) / 3.0, 0.0, None))
    span = camera.get("span", float(np.max(x.max(0) - x.min(0)) + 1e-6))
    sizes = np.clip(sigma / (0.02 * span), 0.5, 40.0) * 6.0

    fig = plt.figure(figsize=(6, 5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colors, s=sizes, alpha=0.85, linewidths=0)
    center = camera.get("center", 0.5 * (x.max(0) + x.min(0)))
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=camera.get("elev", 18.0), azim=camera.get("azim", -60.0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    if path is not None:
        fig.savefig(str(path))
    plt.close(fig)
    return colors


def rasterize_inria(state, camera, image_size=(512, 512), background=(0.0, 0.0, 0.0),
                    sh_degree: int | None = None):
    """Rasterize the current state with the INRIA diff_gaussian_rasterization kernel, if it
    is importable. That package is separately licensed (non-commercial) and is not a
    dependency of this repo; install it yourself to use this path.

    camera must provide the fields the rasterizer needs (world_view_transform,
    full_proj_transform, camera_center, tanfovx, tanfovy). Raises ImportError when the
    optional dependency is missing.
    """
    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError as exc:
        raise ImportError(
            "rasterize_inria needs diff_gaussian_rasterization, an optional and "
            "separately-licensed (non-commercial) dependency that this repo does not "
            "install; install it yourself to use this path."
        ) from exc

    import torch

    means3d = state["pos"]
    if sh_degree is None:
        sh_degree = round(_to_numpy(state["sh"]).shape[1] ** 0.5) - 1
    colors = _colors_from_state(state, _to_numpy(camera["camera_center"]), sh_degree)
    settings = GaussianRasterizationSettings(
        image_height=int(image_size[1]), image_width=int(image_size[0]),
        tanfovx=float(camera["tanfovx"]), tanfovy=float(camera["tanfovy"]),
        bg=torch.as_tensor(background, dtype=torch.float32, device=means3d.device),
        scale_modifier=1.0,
        viewmatrix=camera["world_view_transform"],
        projmatrix=camera["full_proj_transform"],
        sh_degree=0, campos=camera["camera_center"], prefiltered=False, debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    image, _ = rasterizer(
        means3D=means3d, means2D=torch.zeros_like(means3d),
        colors_precomp=torch.as_tensor(colors, dtype=torch.float32, device=means3d.device),
        opacities=state["opacity"], cov3D_precomp=state["cov6"],
        shs=None, scales=None, rotations=None)
    return image
