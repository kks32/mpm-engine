"""Bake a simulated splat trajectory into temporal B-spline models (4D splats).

A recorded clip (per-frame positions, covariances, rotations) becomes one static set of
per-splat temporal models. Positions and symmetric matrix-log covariances use cubic
B-splines on a shared clamped knot grid; exponentiating the interpolated matrix keeps
every covariance positive definite. Rotations use adaptively selected quaternion SLERP
keys, so interpolation stays on SO(3) and carries a measured angular error tolerance.
Opacity and SH coefficients are constants in v1. Any time inside the clip then evaluates
without the simulator and without per-frame files, so a baked clip can be scrubbed,
interpolated past the recorded rate, and re-exported at an arbitrary fps.

Idea sources (see AUTHORS.md): spacetime Gaussians (Li et al., CVPR 2024), 4D Gaussian
splatting (Wu et al., CVPR 2024; Yang et al., ICLR 2024), and the per-splat temporal
extent of the 4splat format (idea only). scipy is imported inside the functions that
need it so ``import warpmpm.splats`` works without the splats extra installed.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import _quat_to_rotation, _rotation_to_quat

_COV6_IDX = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _as_numpy(a) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def _stack_frames(frames: list[dict]) -> dict:
    """Stack a list of state dicts (torch or numpy) into time-major arrays."""
    pos = np.stack([_as_numpy(f["pos"]) for f in frames]).astype(np.float64)
    cov = np.stack([_as_numpy(f["cov6"]) for f in frames]).astype(np.float64)
    quat = None
    if all("R" in f and f["R"] is not None for f in frames):
        quat = np.stack([_rotation_to_quat(_as_numpy(f["R"])) for f in frames])
        # hemisphere alignment: q and -q are the same rotation, so keep each frame on
        # the hemisphere of its predecessor or the component splines cross zero
        for t in range(1, quat.shape[0]):
            flip = (quat[t] * quat[t - 1]).sum(axis=1) < 0.0
            quat[t, flip] *= -1.0
    first = frames[0]
    return {
        "pos": pos, "cov": cov, "quat": quat,
        "opacity": _as_numpy(first["opacity"]).astype(np.float32),
        "sh": _as_numpy(first["sh"]).astype(np.float32),
    }


def _frames_from_dir(frames_dir) -> tuple[list[dict], np.ndarray]:
    """Load a FrameRecorder directory back into state dicts plus times from the manifest.
    PLY frames carry no material rotation, so directory bakes have no quaternion track."""
    from .io import load_gaussians_ply

    frames_dir = Path(frames_dir)
    paths = sorted(frames_dir.glob("frame_*.ply"))
    if not paths:
        raise ValueError(f"no frame_*.ply files in {frames_dir}")
    fps = 30.0
    manifest = frames_dir / "manifest.json"
    if manifest.exists():
        fps = float(json.loads(manifest.read_text()).get("fps", fps))
    frames = []
    for p in paths:
        c = load_gaussians_ply(p)
        frames.append({"pos": c.pos, "cov6": c.cov, "R": None,
                       "opacity": c.opacity, "sh": c.sh})
    times = np.arange(len(paths), dtype=np.float64) / fps
    return frames, times


def _clamped_knots(t0: float, t1: float, n_coef: int, k: int) -> np.ndarray:
    n_interior = n_coef - k - 1
    interior = np.linspace(t0, t1, n_interior + 2)[1:-1]
    return np.concatenate([np.full(k + 1, t0), interior, np.full(k + 1, t1)])


def _fit_channels(times: np.ndarray, y: np.ndarray, n_coef: int, k: int):
    """Least-squares B-spline through y (T, ...) on a shared clamped uniform knot grid.
    Returns a scipy BSpline with vector-valued coefficients (n_coef, ...)."""
    from scipy.interpolate import BSpline, make_lsq_spline

    knots = _clamped_knots(float(times[0]), float(times[-1]), n_coef, k)
    flat = y.reshape(y.shape[0], -1)
    spl = make_lsq_spline(times, flat, knots, k=k)
    return BSpline(knots, np.asarray(spl.c).reshape((n_coef, *y.shape[1:])), k)


def _cov6_to_matrix(cov6: np.ndarray) -> np.ndarray:
    cov6 = np.asarray(cov6, dtype=np.float64)
    if cov6.shape[-1] != 6:
        raise ValueError(f"covariance must end in 6 packed entries, got {cov6.shape}")
    matrix = np.zeros((*cov6.shape[:-1], 3, 3), dtype=np.float64)
    for c, (i, j) in enumerate(_COV6_IDX):
        matrix[..., i, j] = cov6[..., c]
        matrix[..., j, i] = cov6[..., c]
    return matrix


def _matrix_to_cov6(matrix: np.ndarray) -> np.ndarray:
    return np.stack([matrix[..., i, j] for i, j in _COV6_IDX], axis=-1)


def _covariance_log(cov6: np.ndarray) -> tuple[np.ndarray, int]:
    """Map packed covariances to packed symmetric matrix logarithms.

    Non-positive input eigenvalues are repaired relative to the largest eigenvalue and
    reported to the caller. Interpolation after this map is unconstrained; the matching
    matrix exponential maps every finite symmetric result back into the SPD cone.
    """
    matrix = _cov6_to_matrix(cov6)
    if not np.isfinite(matrix).all():
        raise ValueError("covariances contain non-finite entries")
    evals, evecs = np.linalg.eigh(matrix)
    scale = np.maximum(evals[..., -1], 1e-30)
    floor = np.maximum(1e-12 * scale, 1e-30)
    repaired = int(np.count_nonzero(evals <= floor[..., None]))
    evals = np.maximum(evals, floor[..., None])
    log_matrix = np.einsum("...ij,...j,...kj->...ik", evecs, np.log(evals), evecs)
    return _matrix_to_cov6(log_matrix), repaired


def _covariance_exp(log_cov6: np.ndarray) -> np.ndarray:
    matrix = _cov6_to_matrix(log_cov6)
    evals, evecs = np.linalg.eigh(matrix)
    # The bounds are far outside practical splat scales but prevent an invalid file from
    # creating inf/zero covariances during legacy or external-data evaluation.
    evals = np.exp(np.clip(evals, -80.0, 80.0))
    cov = np.einsum("...ij,...j,...kj->...ik", evecs, evals, evecs)
    return _matrix_to_cov6(cov)


def _project_spd(cov6: np.ndarray) -> np.ndarray:
    """Make legacy direct-covariance bakes safe at evaluation time."""
    matrix = _cov6_to_matrix(cov6)
    evals, evecs = np.linalg.eigh(matrix)
    scale = np.maximum(np.max(np.abs(evals), axis=-1), 1e-30)
    evals = np.maximum(evals, 1e-12 * scale[..., None])
    cov = np.einsum("...ij,...j,...kj->...ik", evecs, evals, evecs)
    return _matrix_to_cov6(cov)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Vectorized shortest-path quaternion interpolation with broadcastable u."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    close = sin_theta < 1e-8
    denom = np.where(close, 1.0, sin_theta)
    w0 = np.where(close, 1.0 - u, np.sin((1.0 - u) * theta) / denom)
    w1 = np.where(close, u, np.sin(u * theta) / denom)
    q = w0 * q0 + w1 * q1
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-15)


def _evaluate_rotation_track(query_times: np.ndarray, key_times: np.ndarray,
                             key_quats: np.ndarray) -> np.ndarray:
    query_times = np.atleast_1d(np.asarray(query_times, dtype=np.float64))
    right = np.searchsorted(key_times, query_times, side="right")
    right = np.clip(right, 1, len(key_times) - 1)
    left = right - 1
    span = key_times[right] - key_times[left]
    u = ((query_times - key_times[left]) / span)[:, None, None]
    return _quat_slerp(key_quats[left], key_quats[right], u)


def _fit_rotation_keys(times: np.ndarray, quats: np.ndarray,
                       tol_rot: float) -> tuple[np.ndarray, np.ndarray, float]:
    """SO(3) Douglas-Peucker selection with shared keys and O(N splats) memory."""
    keys = {0, len(times) - 1}
    segments = [(0, len(times) - 1)]
    accepted_max_error = 0.0
    while segments:
        left, right = segments.pop()
        worst_error = 0.0
        worst_time = None
        span = times[right] - times[left]
        for ti in range(left + 1, right):
            u = (times[ti] - times[left]) / span
            estimate = _quat_slerp(quats[left], quats[right], np.asarray(u))
            dot = np.abs(np.sum(estimate * quats[ti], axis=-1))
            error = float(np.max(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))
            if error > worst_error:
                worst_error = error
                worst_time = ti
        if worst_time is not None and worst_error > tol_rot:
            keys.add(worst_time)
            segments.append((left, worst_time))
            segments.append((worst_time, right))
        else:
            accepted_max_error = max(accepted_max_error, worst_error)
    key_idx = np.array(sorted(keys), dtype=int)
    return times[key_idx].copy(), quats[key_idx].copy(), accepted_max_error


@dataclass
class Baked4DSplats:
    """A baked clip: evaluate any t in [t0, t1] to a render-state dict."""

    knots: np.ndarray
    k: int
    coef_pos: np.ndarray                   # (n_coef, N, 3)
    coef_cov: np.ndarray                   # (n_coef, N, 6), matrix-log for new bakes
    coef_quat: np.ndarray | None           # (n_rot_keys, N, 4) or legacy spline coeffs
    opacity: np.ndarray                    # (N, 1)
    sh: np.ndarray                         # (N, K, 3)
    windows: np.ndarray                    # (N, 2) [t_birth, t_death]
    rotation_times: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    _splines: dict = field(default_factory=dict, repr=False)

    @property
    def t0(self) -> float:
        return float(self.knots[0])

    @property
    def t1(self) -> float:
        return float(self.knots[-1])

    @property
    def n(self) -> int:
        return self.coef_pos.shape[1]

    def _spline(self, name: str):
        from scipy.interpolate import BSpline

        if name not in self._splines:
            coef = {"pos": self.coef_pos, "cov": self.coef_cov, "quat": self.coef_quat}[name]
            self._splines[name] = BSpline(self.knots, coef, self.k)
        return self._splines[name]

    def at(self, t: float) -> dict:
        """Render state at time t (clamped to the clip): pos, cov6, R (when the bake has a
        rotation track), opacity, sh, all numpy float32. Same keys as SplatScene.state()."""
        t = float(np.clip(t, self.t0, self.t1))
        raw_cov = self._spline("cov")(t)
        if self.meta.get("covariance_parameterization") == "log_euclidean":
            cov = _covariance_exp(raw_cov)
        else:
            cov = _project_spd(raw_cov)
        state = {
            "pos": self._spline("pos")(t).astype(np.float32),
            "cov6": cov.astype(np.float32),
            "opacity": self.opacity,
            "sh": self.sh,
        }
        if self.coef_quat is not None:
            if (self.rotation_times is not None
                    and self.meta.get("rotation_parameterization") == "slerp"):
                q = _evaluate_rotation_track(
                    np.array([t]), self.rotation_times, self.coef_quat)[0]
            else:
                q = self._spline("quat")(t)
                q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            state["R"] = _quat_to_rotation(q).astype(np.float32)
        else:
            state["R"] = np.broadcast_to(np.eye(3, dtype=np.float32),
                                         (self.n, 3, 3)).copy()
        return state

    def save(self, path) -> Path:
        path = Path(path)
        arrays = dict(knots=self.knots, k=np.int64(self.k), coef_pos=self.coef_pos,
                      coef_cov=self.coef_cov, opacity=self.opacity, sh=self.sh,
                      windows=self.windows, meta=json.dumps(self.meta))
        if self.coef_quat is not None:
            arrays["coef_quat"] = self.coef_quat
        if self.rotation_times is not None:
            arrays["rotation_times"] = self.rotation_times
        with open(path, "wb") as f:                 # exact path, no .npz auto-append
            np.savez_compressed(f, **arrays)
        return path

    @classmethod
    def load(cls, path) -> Baked4DSplats:
        d = np.load(path, allow_pickle=False)
        return cls(knots=d["knots"], k=int(d["k"]), coef_pos=d["coef_pos"],
                   coef_cov=d["coef_cov"],
                   coef_quat=d.get("coef_quat"),
                   opacity=d["opacity"], sh=d["sh"], windows=d["windows"],
                   rotation_times=d.get("rotation_times"),
                   meta=json.loads(str(d["meta"])))

    def report(self) -> dict:
        """Fit errors from the bake plus the compression against per-frame PLY storage.
        The PLY size is the exact per-vertex field count of the export layout (17 floats
        plus 3 per non-DC SH coefficient) so the ratio does not need the files on disk."""
        n_frames = int(self.meta.get("n_frames", 0))
        floats = 17 + 3 * (self.sh.shape[1] - 1)
        ply_bytes = n_frames * self.n * floats * 4
        baked = (self.coef_pos.nbytes + self.coef_cov.nbytes + self.opacity.nbytes
                 + self.sh.nbytes + self.windows.nbytes + self.knots.nbytes
                 + (self.coef_quat.nbytes if self.coef_quat is not None else 0)
                 + (self.rotation_times.nbytes if self.rotation_times is not None else 0))
        return {
            "n_splats": self.n, "n_frames": n_frames, "n_coef": self.coef_pos.shape[0],
            "max_pos_err": self.meta.get("max_pos_err"),
            "max_cov_rel_err": self.meta.get("max_cov_rel_err"),
            "max_rot_err_rad": self.meta.get("max_rot_err_rad"),
            "tol_x": self.meta.get("tol_x"), "tol_cov": self.meta.get("tol_cov"),
            "tol_rot_rad": self.meta.get("tol_rot_rad"),
            "baked_bytes": int(baked), "per_frame_ply_bytes": int(ply_bytes),
            "compression_ratio": float(ply_bytes / baked) if baked else None,
        }

    def write_frames(self, out_dir, n_frames: int | None = None, fps: float = 30.0,
                     t0: float | None = None, t1: float | None = None,
                     sh_mode: str = "dc") -> list[Path]:
        """Export viewer PLY frames by evaluating the bake at n_frames uniform times over
        the clip, so a clip recorded at a low rate plays back smoothly at a higher one.
        Simulated clips are typically milliseconds long, so the output is sized by frame
        count; fps only goes into the manifest as the viewer's playback-rate hint. With
        n_frames None, twice the recorded frame count is written."""
        from .export import export_frame_ply

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = self.t0 if t0 is None else t0
        t1 = self.t1 if t1 is None else t1
        if n_frames is None:
            n_frames = max(2, 2 * int(self.meta.get("n_frames", 2)))
        times = np.linspace(t0, t1, int(n_frames))
        paths = []
        for i, t in enumerate(times):
            p = out_dir / f"frame_{i:04d}.ply"
            export_frame_ply(self.at(float(t)), p, sh_mode=sh_mode)
            paths.append(p)
        (out_dir / "manifest.json").write_text(json.dumps(
            {"frame_count": len(paths), "fps": fps, "sh_mode": sh_mode,
             "baked": True}, indent=2))
        return paths


def bake(frames, times=None, tol_x: float | None = None, tol_cov: float = 1e-2,
         with_rotation: bool = True, tol_rot: float = 1e-3) -> Baked4DSplats:
    """Fit shared-knot cubic B-splines to a recorded splat trajectory.

    frames: a list of SplatScene.state() dicts (torch or numpy), or a FrameRecorder
    directory (whose PLY frames carry no rotation track). times: per-frame times in
    seconds; required for a list, taken from the manifest fps for a directory.

    tol_x: max allowed position reconstruction error at the recorded times; None picks
    0.001 of the first frame's bounding-box diagonal. tol_cov: max allowed relative
    covariance Frobenius error. tol_rot is the maximum angular reconstruction error in
    radians at recorded frames. The knot/key counts grow until their tolerances hold.
    """
    if isinstance(frames, (str, Path)):
        frames, dir_times = _frames_from_dir(frames)
        times = dir_times if times is None else np.asarray(times, dtype=np.float64)
    if times is None:
        raise ValueError("times is required when frames is a list")
    times = np.asarray(times, dtype=np.float64)
    n_frames = len(frames)
    if n_frames < 2:
        raise ValueError(f"need at least 2 frames to bake, got {n_frames}")
    if len(times) != n_frames:
        raise ValueError(f"{n_frames} frames but {len(times)} times")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be finite and strictly increasing")
    if tol_cov <= 0.0 or tol_rot <= 0.0:
        raise ValueError("tol_cov and tol_rot must be positive")

    data = _stack_frames(frames)
    pos, cov, quat = data["pos"], data["cov"], data["quat"]
    if not with_rotation:
        quat = None

    if tol_x is None:
        lo, hi = pos[0].min(0), pos[0].max(0)
        tol_x = 1e-3 * float(np.linalg.norm(hi - lo))

    k = min(3, n_frames - 1)
    n_coef = min(n_frames, max(k + 1, n_frames // 8))
    log_cov, repaired = _covariance_log(cov)
    if repaired:
        warnings.warn(
            f"repaired {repaired} non-positive covariance eigenvalues before baking",
            RuntimeWarning, stacklevel=2)
    while True:
        spl_pos = _fit_channels(times, pos, n_coef, k)
        spl_cov = _fit_channels(times, log_cov, n_coef, k)
        pos_err = float(np.abs(spl_pos(times) - pos).max())
        cov_fit = _covariance_exp(spl_cov(times))
        diff = np.linalg.norm(cov_fit - cov, axis=2)
        cov_err = float((diff / (np.linalg.norm(cov, axis=2) + 1e-30)).max())
        if (pos_err <= tol_x and cov_err <= tol_cov) or n_coef >= n_frames:
            break
        n_coef = min(n_frames, max(n_coef + 1, int(n_coef * 1.8)))
    if pos_err > tol_x or cov_err > tol_cov:
        warnings.warn(
            f"bake at one coefficient per frame still misses tolerance "
            f"(pos {pos_err:.2e} vs {tol_x:.2e}, cov {cov_err:.2e} vs {tol_cov:.2e}); "
            f"the recording is under-sampled for its motion", stacklevel=2)

    rotation_times = None
    rotation_quats = None
    rot_err = None
    if quat is not None:
        rotation_times, rotation_quats, rot_err = _fit_rotation_keys(times, quat, tol_rot)
        if rot_err > tol_rot:
            warnings.warn(
                f"rotation fit misses tolerance ({rot_err:.2e} rad vs {tol_rot:.2e})",
                RuntimeWarning, stacklevel=2)
    knots = _clamped_knots(float(times[0]), float(times[-1]), n_coef, k)
    windows = np.tile(np.array([times[0], times[-1]], dtype=np.float64),
                      (pos.shape[1], 1))
    meta = {
        "n_frames": n_frames,
        "tol_x": tol_x,
        "tol_cov": tol_cov,
        "tol_rot_rad": tol_rot,
        "max_pos_err": pos_err,
        "max_cov_rel_err": cov_err,
        "max_rot_err_rad": rot_err,
        "covariance_parameterization": "log_euclidean",
        "rotation_parameterization": "slerp" if rotation_quats is not None else None,
        "n_rotation_keys": 0 if rotation_times is None else len(rotation_times),
    }
    return Baked4DSplats(
        knots=knots, k=k,
        coef_pos=np.asarray(spl_pos.c, dtype=np.float32),
        coef_cov=np.asarray(spl_cov.c, dtype=np.float32),
        coef_quat=None if rotation_quats is None else rotation_quats.astype(np.float32),
        opacity=data["opacity"], sh=data["sh"], windows=windows,
        rotation_times=rotation_times, meta=meta)
