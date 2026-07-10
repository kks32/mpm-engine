"""GaussianCloud data container plus PLY load/save and a synthetic scene generator.

The PLY layout is the INRIA 3D Gaussian Splatting one: per vertex x, y, z, nx, ny, nz,
f_dc_0..2, f_rest_0..(3*((deg+1)^2 - 1) - 1), opacity, scale_0..2, rot_0..3. Stored
scales are log, opacity is a logit, the quaternion is (w, x, y, z) and normalized before
use, and f_rest is channel-major. plyfile is imported inside the functions that need it
so ``import warpmpm.splats`` works without the splats extra installed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .appearance import rgb_to_sh_dc


@dataclass
class GaussianCloud:
    """Plain data for a Gaussian-splat scene. No solver coupling lives here.

    pos:     (N, 3) f32 centers.
    cov:     (N, 6) f32 covariance, upper-triangular (xx, xy, xz, yy, yz, zz), world units.
    opacity: (N, 1) f32 in [0, 1], post-sigmoid.
    sh:      (N, K, 3) f32 spherical-harmonic coefficients, K = (sh_degree + 1)^2.
    sh_degree: int SH degree.
    scales:  optional (N, 3) f32 linear scales; quats optional (N, 4) f32 (w, x, y, z).
             Populated by the loader and the generator so save_gaussians_ply round-trips
             the covariance exactly through the standard scale/rotation fields.
    """

    pos: np.ndarray
    cov: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    sh_degree: int
    scales: np.ndarray | None = None
    quats: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.pos.shape[0]


def _quat_to_rotation(quats: np.ndarray) -> np.ndarray:
    """(N, 4) quaternions (w, x, y, z), normalized, to (N, 3, 3) rotation matrices."""
    q = np.asarray(quats, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _cov6_from_scale_quat(scales: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Sigma = R diag(scale^2) R^T, packed to (N, 6) upper-triangular."""
    R = _quat_to_rotation(quats)
    s2 = (np.asarray(scales, dtype=np.float64)) ** 2
    sigma = np.einsum("nij,nj,nkj->nik", R, s2, R)     # R S^2 R^T
    idx = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    return np.stack([sigma[:, i, j] for (i, j) in idx], axis=1).astype(np.float32)


def _sigmoid(a: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def load_gaussians_ply(path, sh_degree: int | None = None) -> GaussianCloud:
    """Load an INRIA-layout Gaussian-splat PLY into a GaussianCloud (world units).

    sh_degree defaults to None, which infers the degree from the f_rest_* fields
    present, so DC-only files (for example sh_mode="dc" exports) load without
    arguments. An explicit sh_degree must not exceed what the file carries."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply["vertex"]
    n = v.count
    pos = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opacity = _sigmoid(np.asarray(v["opacity"], dtype=np.float64)).astype(np.float32)[:, None]

    fields = {p.name for p in v.properties}
    n_rest_file = sum(1 for f in fields if f.startswith("f_rest_"))
    degree_file = math.isqrt(n_rest_file // 3 + 1) - 1
    if sh_degree is None:
        sh_degree = degree_file
    elif sh_degree > degree_file:
        raise ValueError(f"file carries SH degree {degree_file}, requested {sh_degree}")
    kdim = (sh_degree + 1) ** 2
    sh = np.zeros((n, kdim, 3), dtype=np.float32)
    sh[:, 0, :] = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    if kdim > 1:
        # the file's channel-major layout spans its own band count, so reshape with the
        # file's K and truncate bands afterwards
        kf = (degree_file + 1) ** 2
        rest = np.stack([np.asarray(v[f"f_rest_{i}"]) for i in range(n_rest_file)], axis=1)
        rest = rest.reshape(n, 3, kf - 1).transpose(0, 2, 1)
        sh[:, 1:, :] = rest[:, : kdim - 1, :]

    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)).astype(
        np.float32)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(
        np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    cov = _cov6_from_scale_quat(scales, quats)
    return GaussianCloud(pos=pos, cov=cov, opacity=opacity, sh=sh, sh_degree=sh_degree,
                         scales=scales, quats=quats)


def save_gaussians_ply(cloud: GaussianCloud, path) -> None:
    """Write a GaussianCloud to an INRIA-layout PLY. Uses cloud.scales/quats when present
    (exact covariance round-trip); otherwise recovers them from the covariance by
    eigendecomposition."""
    from plyfile import PlyData, PlyElement

    n = cloud.n
    scales, quats = cloud.scales, cloud.quats
    if scales is None or quats is None:
        scales, quats = _scale_quat_from_cov6(cloud.cov)

    log_scale = np.log(np.clip(np.asarray(scales, dtype=np.float64), 1e-12, None))
    logit_op = _logit(cloud.opacity.reshape(-1))

    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    n_rest = 3 * (cloud.sh.shape[1] - 1)
    fields += [f"f_rest_{i}" for i in range(n_rest)]
    fields += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

    data = np.zeros(n, dtype=[(f, "f4") for f in fields])
    data["x"], data["y"], data["z"] = cloud.pos[:, 0], cloud.pos[:, 1], cloud.pos[:, 2]
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = (
        cloud.sh[:, 0, 0], cloud.sh[:, 0, 1], cloud.sh[:, 0, 2])
    if n_rest > 0:
        rest = cloud.sh[:, 1:, :].transpose(0, 2, 1).reshape(n, n_rest)  # -> channel-major
        for i in range(n_rest):
            data[f"f_rest_{i}"] = rest[:, i]
    data["opacity"] = logit_op.astype(np.float32)
    for i in range(3):
        data[f"scale_{i}"] = log_scale[:, i].astype(np.float32)
    for i in range(4):
        data[f"rot_{i}"] = np.asarray(quats, dtype=np.float32)[:, i]

    el = PlyElement.describe(data, "vertex")
    PlyData([el]).write(str(path))


def _scale_quat_from_cov6(cov6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover (scales, quats) from covariances by symmetric eigendecomposition. Lossy in
    the scale/quat representation but reconstructs the covariance to numerical precision."""
    cov6 = np.asarray(cov6, dtype=np.float64)
    n = cov6.shape[0]
    sigma = np.zeros((n, 3, 3))
    idx = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    for c, (i, j) in enumerate(idx):
        sigma[:, i, j] = cov6[:, c]
        sigma[:, j, i] = cov6[:, c]
    evals, evecs = np.linalg.eigh(sigma)
    scales = np.sqrt(np.clip(evals, 0.0, None)).astype(np.float32)
    quats = _rotation_to_quat(evecs).astype(np.float32)
    return scales, quats


def _rotation_to_quat(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices to (N, 4) quaternions (w, x, y, z). Forces det = +1."""
    R = np.asarray(R, dtype=np.float64).copy()
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"R must have shape (N, 3, 3), got {R.shape}")
    if not np.isfinite(R).all():
        raise ValueError("R contains non-finite entries")
    flip = np.linalg.det(R) < 0
    R[flip, :, 2] *= -1.0
    q = np.empty((len(R), 4), dtype=np.float64)
    trace = np.trace(R, axis1=1, axis2=2)

    # The positive-trace formula is well conditioned away from pi.  Near pi,
    # select the largest diagonal component instead; unlike division by 4w,
    # these branches remain finite when the scalar quaternion component is zero.
    m = trace > 0.0
    s = 2.0 * np.sqrt(np.maximum(trace[m] + 1.0, 0.0))
    q[m, 0] = 0.25 * s
    q[m, 1] = (R[m, 2, 1] - R[m, 1, 2]) / s
    q[m, 2] = (R[m, 0, 2] - R[m, 2, 0]) / s
    q[m, 3] = (R[m, 1, 0] - R[m, 0, 1]) / s

    remaining = ~m
    mx = remaining & (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2])
    s = 2.0 * np.sqrt(np.maximum(1.0 + R[mx, 0, 0] - R[mx, 1, 1] - R[mx, 2, 2], 0.0))
    q[mx, 0] = (R[mx, 2, 1] - R[mx, 1, 2]) / s
    q[mx, 1] = 0.25 * s
    q[mx, 2] = (R[mx, 0, 1] + R[mx, 1, 0]) / s
    q[mx, 3] = (R[mx, 0, 2] + R[mx, 2, 0]) / s

    my = remaining & ~mx & (R[:, 1, 1] >= R[:, 2, 2])
    s = 2.0 * np.sqrt(np.maximum(1.0 + R[my, 1, 1] - R[my, 0, 0] - R[my, 2, 2], 0.0))
    q[my, 0] = (R[my, 0, 2] - R[my, 2, 0]) / s
    q[my, 1] = (R[my, 0, 1] + R[my, 1, 0]) / s
    q[my, 2] = 0.25 * s
    q[my, 3] = (R[my, 1, 2] + R[my, 2, 1]) / s

    mz = remaining & ~mx & ~my
    s = 2.0 * np.sqrt(np.maximum(1.0 + R[mz, 2, 2] - R[mz, 0, 0] - R[mz, 1, 1], 0.0))
    q[mz, 0] = (R[mz, 1, 0] - R[mz, 0, 1]) / s
    q[mz, 1] = (R[mz, 0, 2] + R[mz, 2, 0]) / s
    q[mz, 2] = (R[mz, 1, 2] + R[mz, 2, 1]) / s
    q[mz, 3] = 0.25 * s

    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("R contains a matrix that cannot be converted to a quaternion")
    return q / norm


def make_synthetic_cloud(shape: str = "box", n: int = 6000, size=(0.12, 0.10, 0.07),
                         center=(0.0, 0.0, 0.0), sh_degree: int = 0, color=None,
                         seed: int = 0) -> GaussianCloud:
    """A zero-download demo scene and test fixture: a jittered lattice inside a box or
    sphere, isotropic covariances with sigma = half the mean spacing, opacity 0.95, and
    DC-only SH from color (or a smooth position-based colormap)."""
    rng = np.random.default_rng(seed)
    size = np.asarray(size, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    # a lattice sized to yield roughly n points inside the shape, then jittered
    if shape == "box":
        vol_frac = 1.0
    elif shape == "sphere":
        vol_frac = np.pi / 6.0     # sphere volume / bounding-box volume
    else:
        raise ValueError(f"shape must be 'box' or 'sphere', got {shape!r}")
    per_axis = max(2, int(np.ceil((n / vol_frac) ** (1.0 / 3.0))))
    axes = [np.linspace(-0.5, 0.5, per_axis) * size[d] for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    spacing = float(np.min(size / (per_axis - 1)))
    grid = grid + rng.uniform(-0.25 * spacing, 0.25 * spacing, size=grid.shape)

    if shape == "sphere":
        r = size / 2.0
        inside = np.sum((grid / r) ** 2, axis=1) <= 1.0
        grid = grid[inside]
    if grid.shape[0] > n:
        grid = grid[rng.choice(grid.shape[0], n, replace=False)]
    pos = (grid + center).astype(np.float32)
    m = pos.shape[0]

    sigma = 0.5 * spacing
    cov = np.zeros((m, 6), dtype=np.float32)
    cov[:, 0] = cov[:, 3] = cov[:, 5] = sigma * sigma
    scales = np.full((m, 3), sigma, dtype=np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (m, 1))

    opacity = np.full((m, 1), 0.95, dtype=np.float32)

    kdim = (sh_degree + 1) ** 2
    sh = np.zeros((m, kdim, 3), dtype=np.float32)
    if color is not None:
        rgb = np.broadcast_to(np.asarray(color, np.float32), (m, 3))
    else:
        # smooth colormap from normalized position: maps each axis to a channel
        lo, hi = grid.min(0), grid.max(0)
        rgb = ((grid - lo) / np.clip(hi - lo, 1e-9, None)).astype(np.float32)
    sh[:, 0, :] = rgb_to_sh_dc(rgb)

    return GaussianCloud(pos=pos, cov=cov, opacity=opacity, sh=sh, sh_degree=sh_degree,
                         scales=scales, quats=quats)
