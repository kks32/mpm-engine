"""Per-frame splat export so a simulation plays in Cheng-Hsi's SplatViewer.

The viewer (github.com/chhsiao93/SplatViewer) preloads frame_{index:04}.sog files from a
folder and replays them with play, pause, and seek. Panes that share a groupId share one
frame clock, so two folders (truth and recovered-law rollouts) play in lockstep for a
side-by-side comparison.

This module turns a SplatScene.state() dict into an INRIA-layout Gaussian-splat PLY per
frame, optionally rotates the spherical harmonics into the deformed body frame, and (when
node is available) converts the PLY frames to .sog with the PlayCanvas splat-transform CLI.

SH rotation is exact: the per-band rotation operator is the unique linear map M on band l
that satisfies eval_sh(M @ coeffs, d) == eval_sh(coeffs, R^T d) for every direction d. It
is built by solving that identity at a fixed set of sample directions in the same real-SH
basis that eval_sh uses, so it cannot drift from the evaluator's convention. See
tests/test_splat_export.py for the exactness check on degrees 1 through 3.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .appearance import C1, C2, C3
from .io import GaussianCloud, _rotation_to_quat, save_gaussians_ply

_IDX6 = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _to_numpy(t) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def cov6_to_scale_quat(cov6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factor covariances into scales and rotations by symmetric eigendecomposition.

    cov6 is (N, 6) upper-triangular (xx, xy, xz, yy, yz, zz). Returns scales (N, 3) as the
    square roots of the clamped eigenvalues and quats (N, 4) in (w, x, y, z) order. The
    eigenvector matrix is turned into a proper rotation by flipping one column when its
    determinant is negative, which leaves R diag(scale^2) R^T unchanged.
    """
    cov6 = np.asarray(cov6, dtype=np.float64)
    n = cov6.shape[0]
    sigma = np.zeros((n, 3, 3))
    for c, (i, j) in enumerate(_IDX6):
        sigma[:, i, j] = cov6[:, c]
        sigma[:, j, i] = cov6[:, c]

    evals, evecs = np.linalg.eigh(sigma)
    scales = np.sqrt(np.clip(evals, 0.0, None)).astype(np.float32)

    R = evecs.copy()
    flip = np.linalg.det(R) < 0.0
    R[flip, :, 0] *= -1.0
    quats = _rotation_to_quat(R).astype(np.float32)
    return scales, quats


def _fibonacci_dirs(n: int) -> np.ndarray:
    """n roughly uniform unit directions on the sphere (a Fibonacci spiral)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


# fixed, well spread sample directions used to build every band's rotation operator
_SAMPLE_DIRS = _fibonacci_dirs(32)


def _band_basis(band: int, dirs: np.ndarray) -> np.ndarray:
    """Real-SH basis functions of a band evaluated at dirs (..., 3) -> (..., 2*band+1).

    These are the exact per-coefficient factors that eval_sh multiplies each band's
    coefficients by, so a rotation operator built from them matches eval_sh by construction.
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    if band == 1:
        return np.stack([-C1 * y, C1 * z, -C1 * x], axis=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    if band == 2:
        return np.stack([
            C2[0] * xy,
            C2[1] * yz,
            C2[2] * (2.0 * zz - xx - yy),
            C2[3] * xz,
            C2[4] * (xx - yy),
        ], axis=-1)
    if band == 3:
        return np.stack([
            C3[0] * y * (3.0 * xx - yy),
            C3[1] * xy * z,
            C3[2] * y * (4.0 * zz - xx - yy),
            C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            C3[4] * x * (4.0 * zz - xx - yy),
            C3[5] * z * (xx - yy),
            C3[6] * x * (xx - 3.0 * yy),
        ], axis=-1)
    raise ValueError(f"band rotation supports band = 1, 2, 3, got {band}")


_BAND_PINV: dict[int, np.ndarray] = {}


def _band_pinv(band: int) -> np.ndarray:
    """Pseudo-inverse of the band's sample-basis matrix, cached per band."""
    if band not in _BAND_PINV:
        _BAND_PINV[band] = np.linalg.pinv(_band_basis(band, _SAMPLE_DIRS))
    return _BAND_PINV[band]


def _band_rotation(band: int, R: np.ndarray) -> np.ndarray:
    """Per-splat rotation matrices M (N, 2*band+1, 2*band+1) for one band.

    M is the unique operator with sum_m M[m, m'] Y_m(d) = Y_m'(R^T d) for the band's basis
    Y, so eval_sh(M @ coeffs, d) == eval_sh(coeffs, R^T d). It is recovered exactly (not
    fit) by evaluating that right-hand side at the fixed sample directions and solving the
    band-restricted linear system, which is well posed because the sampled basis has full
    column rank.
    """
    rt_dirs = np.einsum("sj,njk->nsk", _SAMPLE_DIRS, R)   # R^T d for each sample d
    y_rot = _band_basis(band, rt_dirs)                    # (N, S, 2*band+1)
    return np.einsum("ds,nse->nde", _band_pinv(band), y_rot)


def rotate_sh(deg: int, sh: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate SH coefficients into the body frame set by per-splat rotations R.

    sh is (N, K, 3) with K = (deg + 1)^2, R is (N, 3, 3). The band-0 (DC) term is rotation
    invariant and passes through. Bands 1 through deg are rotated so that evaluating the
    result at a view direction d equals evaluating the input at R^T d, which bakes the
    PhysGaussian inverse-view-rotation trick into the stored coefficients.
    """
    sh = _to_numpy(sh).astype(np.float64)
    R = _to_numpy(R).astype(np.float64)
    out = sh.copy()
    for band in range(1, deg + 1):
        s, e = band * band, band * band + (2 * band + 1)
        m = _band_rotation(band, R)                         # (N, dim, dim)
        out[:, s:e, :] = np.einsum("nmp,npc->nmc", m, sh[:, s:e, :])
    return out.astype(np.float32)


def export_frame_ply(state: dict, path, sh_mode: str = "dc") -> Path:
    """Write one INRIA-layout Gaussian-splat PLY from a SplatScene.state() dict.

    state carries torch or numpy arrays: pos (N, 3) world, cov6 (N, 6) world, R (N, 3, 3),
    opacity (N, 1) in [0, 1], sh (N, K, 3). Scales and rotation are recovered from the
    covariance and written as log-scale and a normalized quaternion; opacity is written as
    its logit. The PLY loader inverts both activations.

    sh_mode "dc" bakes only the view-independent DC color and zeros the higher bands, which
    is exact for DC-only clouds and is the v1 default. sh_mode "rotate" rotates every band
    into the deformed body frame with rotate_sh so view-dependent color stays correct as the
    material turns.
    """
    pos = _to_numpy(state["pos"]).astype(np.float32)
    cov6 = _to_numpy(state["cov6"]).astype(np.float64)
    opacity = _to_numpy(state["opacity"]).reshape(-1, 1).astype(np.float32)
    sh = _to_numpy(state["sh"]).astype(np.float32)
    deg = round(sh.shape[1] ** 0.5) - 1

    if sh_mode == "dc":
        sh_out = np.zeros_like(sh)
        sh_out[:, 0, :] = sh[:, 0, :]
    elif sh_mode == "rotate":
        sh_out = rotate_sh(deg, sh, state["R"])
    else:
        raise ValueError(f"sh_mode must be 'dc' or 'rotate', got {sh_mode!r}")

    scales, quats = cov6_to_scale_quat(cov6)
    cloud = GaussianCloud(pos=pos, cov=cov6.astype(np.float32), opacity=opacity,
                          sh=sh_out, sh_degree=deg, scales=scales, quats=quats)
    path = Path(path)
    save_gaussians_ply(cloud, path)
    return path


class FrameRecorder:
    """Capture SplatScene frames to frame_0000.ply, frame_0001.ply, ... for the viewer.

    Call capture(scene) once per control tick. With every > 1 only every nth tick is
    written, and written frames still get consecutive indices so the viewer sees a dense
    sequence. manifest() writes a small JSON alongside the frames.
    """

    def __init__(self, out_dir, every: int = 1, sh_mode: str = "dc", fps: int = 30):
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.every = int(every)
        self.sh_mode = sh_mode
        self.fps = int(fps)
        self.count = 0
        self.paths: list[Path] = []
        self._tick = 0

    def capture(self, scene) -> Path | None:
        """Export the scene's current state when this tick is on the every schedule."""
        write = (self._tick % self.every) == 0
        self._tick += 1
        if not write:
            return None
        path = self.out_dir / f"frame_{self.count:04d}.ply"
        export_frame_ply(scene.state(), path, sh_mode=self.sh_mode)
        self.count += 1
        self.paths.append(path)
        return path

    def manifest(self, path=None) -> Path:
        """Write frame count, fps hint, and sh_mode to manifest.json."""
        path = Path(path) if path is not None else self.out_dir / "manifest.json"
        data = {"frame_count": self.count, "fps": self.fps, "sh_mode": self.sh_mode}
        path.write_text(json.dumps(data, indent=2))
        return path


def convert_to_sog(frames_dir) -> list[Path]:
    """Convert every frame_*.ply in a folder to .sog with the PlayCanvas CLI.

    Shells out to `npx @playcanvas/splat-transform` per frame when node is on PATH, and
    skips with a one-line message otherwise. This is never a hard dependency. PlayCanvas
    also loads .ply splat assets directly, so the viewer's filenamePattern can point at the
    .ply frames; .sog is only a size and load-time optimization.
    """
    frames_dir = Path(frames_dir)
    plys = sorted(frames_dir.glob("frame_*.ply"))
    if shutil.which("npx") is None:
        print("convert_to_sog: node/npx not found on PATH; leaving .ply frames as is "
              "(the viewer can load .ply directly).")
        return []
    if not plys:
        print(f"convert_to_sog: no frame_*.ply files in {frames_dir}")
        return []

    out = []
    for ply in plys:
        sog = ply.with_suffix(".sog")
        cmd = ["npx", "--yes", "@playcanvas/splat-transform", str(ply), str(sog)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"convert_to_sog: splat-transform failed on {ply.name}: {exc}; "
                  "leaving .ply frames as is.")
            return out
        out.append(sog)
    return out
