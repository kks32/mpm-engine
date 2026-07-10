"""Tests for per-frame splat export (warpmpm.splats.export).

The module skips cleanly when plyfile or scipy is missing. Every test runs on CPU with
small clouds and grids so the suite stays fast. The centerpiece is the SH rotation
exactness check: eval_sh(rotate_sh(coeffs, R), d) must equal eval_sh(coeffs, R^T d) to
1e-5 for degrees 1 through 3.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("plyfile")
pytest.importorskip("scipy")

from warpmpm.splats import (
    FrameRecorder,
    convert_to_sog,
    cov6_to_scale_quat,
    eval_sh,
    export_frame_ply,
    load_gaussians_ply,
    make_synthetic_cloud,
    rotate_sh,
)
from warpmpm.splats.appearance import C1
from warpmpm.splats.io import _cov6_from_scale_quat, _quat_to_rotation, _rotation_to_quat

_IDX6 = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _random_rotations(n: int, rng) -> np.ndarray:
    """n proper rotation matrices (n, 3, 3) from QR of random normals."""
    q, r = np.linalg.qr(rng.standard_normal((n, 3, 3)))
    sign = np.sign(np.diagonal(r, axis1=1, axis2=2))
    sign[sign == 0] = 1.0
    q = q * sign[:, None, :]
    det = np.linalg.det(q)
    q[det < 0, :, 0] *= -1.0
    return q


def _cov6_from_matrices(sigma: np.ndarray) -> np.ndarray:
    """Pack (n, 3, 3) symmetric matrices into (n, 6) upper-triangular covariances."""
    return np.stack([sigma[:, i, j] for (i, j) in _IDX6], axis=1)


def _spd_from_scale_rot(scales: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj,nkj->nik", R, scales ** 2, R)


# --- 1. cov6_to_scale_quat round-trip ----------------------------------------------------
def test_cov6_to_scale_quat_round_trip():
    rng = np.random.default_rng(0)

    R = _random_rotations(200, rng)
    scales = rng.uniform(0.05, 1.5, size=(200, 3))
    sigma = _spd_from_scale_rot(scales, R)

    # a block of near-degenerate covariances (two equal eigenvalues)
    R2 = _random_rotations(40, rng)
    s2 = rng.uniform(0.1, 1.0, size=(40, 1))
    scales2 = np.concatenate([s2, s2, rng.uniform(0.1, 1.0, size=(40, 1))], axis=1)
    sigma = np.concatenate([sigma, _spd_from_scale_rot(scales2, R2)], axis=0)

    # a block built from reflection bases (det = -1) so the determinant fix must engage
    Vref = _random_rotations(40, rng).copy()
    Vref[:, :, 0] *= -1.0
    scales3 = rng.uniform(0.1, 1.0, size=(40, 3))
    sigma = np.concatenate([sigma, _spd_from_scale_rot(scales3, Vref)], axis=0)

    cov6 = _cov6_from_matrices(sigma)
    out_scales, out_quats = cov6_to_scale_quat(cov6)

    # every returned quaternion is a proper rotation (det = +1, the fix worked)
    Rout = _quat_to_rotation(out_quats)
    assert np.allclose(np.linalg.det(Rout), 1.0, atol=1e-6)

    rebuilt = _cov6_from_scale_quat(out_scales, out_quats)
    assert np.allclose(rebuilt, cov6, atol=1e-5)


def test_rotation_to_quat_is_stable_at_pi():
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                     [1.0, -2.0, 3.0]])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    # Rodrigues at pi: R = 2 aa^T - I. Include nearby angles through quaternions.
    exact = 2.0 * np.einsum("ni,nj->nij", axes, axes) - np.eye(3)
    angles = np.array([np.pi - 1e-10, np.pi + 1e-10])
    near_q = np.concatenate([
        np.cos(angles[:, None] / 2.0),
        np.sin(angles[:, None] / 2.0) * axes[2][None, :],
    ], axis=1)
    matrices = np.concatenate([exact, _quat_to_rotation(near_q)], axis=0)
    rebuilt = _quat_to_rotation(_rotation_to_quat(matrices))
    assert np.allclose(rebuilt, matrices, atol=1e-9)


# --- 2. export_frame_ply loads back through the PLY loader --------------------------------
def _state_from_cloud(cloud) -> dict:
    n = cloud.n
    return {
        "pos": cloud.pos.copy(),
        "cov6": cloud.cov.copy(),
        "R": np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        "opacity": cloud.opacity.copy(),
        "sh": cloud.sh.copy(),
    }


def test_export_frame_ply_round_trip_dc(tmp_path):
    cloud = make_synthetic_cloud(shape="box", n=1200, sh_degree=0, seed=1)
    state = _state_from_cloud(cloud)
    path = tmp_path / "frame_0000.ply"
    export_frame_ply(state, path, sh_mode="dc")

    back = load_gaussians_ply(path, sh_degree=0)
    assert np.allclose(back.pos, cloud.pos, atol=1e-5)
    assert np.allclose(back.cov, cloud.cov, atol=1e-5)
    assert np.allclose(back.opacity, cloud.opacity, atol=1e-5)
    assert np.allclose(back.sh, cloud.sh, atol=1e-5)


def test_load_infers_degree_from_file(tmp_path):
    # a degree-0 export carries no f_rest fields; the default loader must infer that
    # instead of assuming degree 3 (regression: the recorded-frame smoke crashed here)
    cloud = make_synthetic_cloud(shape="box", n=600, sh_degree=0, seed=3)
    path = tmp_path / "frame_0000.ply"
    export_frame_ply(_state_from_cloud(cloud), path, sh_mode="dc")

    back = load_gaussians_ply(path)
    assert back.sh_degree == 0
    assert np.allclose(back.sh, cloud.sh, atol=1e-5)
    with pytest.raises(ValueError):
        load_gaussians_ply(path, sh_degree=2)


def test_export_frame_ply_dc_zeros_higher_bands(tmp_path):
    cloud = make_synthetic_cloud(shape="box", n=800, sh_degree=2, seed=2)
    # give the higher bands nonzero content that dc mode must drop
    cloud.sh[:, 1:, :] = 0.3
    state = _state_from_cloud(cloud)
    path = tmp_path / "frame_0000.ply"
    export_frame_ply(state, path, sh_mode="dc")

    back = load_gaussians_ply(path, sh_degree=2)
    assert np.allclose(back.sh[:, 0, :], cloud.sh[:, 0, :], atol=1e-5)
    assert np.allclose(back.sh[:, 1:, :], 0.0, atol=1e-6)


# --- 3. SH rotation exactness ------------------------------------------------------------
@pytest.mark.parametrize("deg", [1, 2, 3])
def test_sh_rotation_exactness(deg):
    rng = np.random.default_rng(100 + deg)
    n = 64
    kdim = (deg + 1) ** 2
    # small coefficients keep eval_sh inside [0, 1] so its clamp stays inactive
    sh = rng.uniform(-0.05, 0.05, size=(n, kdim, 3)).astype(np.float32)
    R = _random_rotations(n, rng).astype(np.float32)

    sh_rot = rotate_sh(deg, sh, R)

    d = rng.standard_normal((n, 3)).astype(np.float32)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    rt_d = np.einsum("njk,nj->nk", R, d)     # R^T d per splat

    lhs = eval_sh(deg, sh_rot, d)
    rhs = eval_sh(deg, sh, rt_d)
    assert np.allclose(lhs, rhs, atol=1e-5)


def test_sh_rotation_band1_matches_direction():
    """Band 1 encodes a linear color a . d; rotating its coefficients must rotate a by R."""
    rng = np.random.default_rng(7)
    n = 50
    a = rng.standard_normal((n, 3))

    sh = np.zeros((n, 4, 3), dtype=np.float32)
    sh[:, 1, 0] = -a[:, 1] / C1
    sh[:, 2, 0] = a[:, 2] / C1
    sh[:, 3, 0] = -a[:, 0] / C1

    R = _random_rotations(n, rng)
    sh_rot = rotate_sh(1, sh, R)

    a_rot = np.stack([-C1 * sh_rot[:, 3, 0], -C1 * sh_rot[:, 1, 0], C1 * sh_rot[:, 2, 0]],
                     axis=1)
    expected = np.einsum("nij,nj->ni", R, a)   # R a
    assert np.allclose(a_rot, expected, atol=1e-5)


# --- 4. FrameRecorder file names and manifest --------------------------------------------
class _FakeScene:
    def __init__(self, n=20, deg=0):
        self._state = {
            "pos": np.random.default_rng(0).standard_normal((n, 3)).astype(np.float32),
            "cov6": np.tile(np.array([1e-4, 0, 0, 1e-4, 0, 1e-4], np.float32), (n, 1)),
            "R": np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
            "opacity": np.full((n, 1), 0.9, np.float32),
            "sh": np.zeros((n, (deg + 1) ** 2, 3), np.float32),
        }

    def state(self):
        return self._state


def test_frame_recorder_names_and_manifest(tmp_path):
    scene = _FakeScene()
    rec = FrameRecorder(tmp_path, every=1, sh_mode="dc", fps=24)
    for _ in range(4):
        rec.capture(scene)
    manifest_path = rec.manifest()

    names = sorted(p.name for p in tmp_path.glob("frame_*.ply"))
    assert names == ["frame_0000.ply", "frame_0001.ply", "frame_0002.ply", "frame_0003.ply"]

    data = json.loads(manifest_path.read_text())
    assert data == {"frame_count": 4, "fps": 24, "sh_mode": "dc"}


def test_frame_recorder_every_skips_but_names_stay_dense(tmp_path):
    scene = _FakeScene()
    rec = FrameRecorder(tmp_path, every=2, sh_mode="dc")
    for _ in range(6):
        rec.capture(scene)
    names = sorted(p.name for p in tmp_path.glob("frame_*.ply"))
    assert names == ["frame_0000.ply", "frame_0001.ply", "frame_0002.ply"]
    assert rec.count == 3


def test_convert_to_sog_no_frames_is_safe(tmp_path):
    # empty folder: returns [] whether or not node is present, never raises
    assert convert_to_sog(tmp_path) == []


# --- 5. end-to-end smoke -----------------------------------------------------------------
def test_end_to_end_records_loadable_frames(tmp_path):
    from warpmpm.core.solver import GridConfig
    from warpmpm.materials import elastic
    from warpmpm.splats.scene import SplatScene

    cloud = make_synthetic_cloud(shape="box", n=1200, sh_degree=1, seed=0)
    grid = GridConfig(n_grid=32, grid_lim=0.4)
    scene = SplatScene(cloud, grid=grid, material=elastic(E=3e5, nu=0.3, density=1000),
                       device="cpu", fill=True, cov_mode="step")

    rec = FrameRecorder(tmp_path, sh_mode="rotate", fps=30)
    for _ in range(5):
        scene.step(dt=1e-4, substeps=10)
        rec.capture(scene)
    rec.manifest()

    plys = sorted(tmp_path.glob("frame_*.ply"))
    assert len(plys) == 5
    for p in plys:
        back = load_gaussians_ply(p, sh_degree=1)
        assert back.pos.shape == (scene.n_visible, 3)
        assert back.cov.shape == (scene.n_visible, 6)
        assert not np.isnan(back.pos).any()

    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["frame_count"] == 5
