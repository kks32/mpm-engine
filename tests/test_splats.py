"""Tests for the Gaussian-splat coupling package (warpmpm.splats).

The whole module skips cleanly when plyfile or scipy is missing. Every test runs on CPU
with small clouds and grids so the suite stays fast.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plyfile")
pytest.importorskip("scipy")

from warpmpm.core.solver import GridConfig, Solver
from warpmpm.materials import elastic
from warpmpm.splats import (
    GaussianCloud,
    SimTransform,
    assign_filler_appearance,
    eval_sh,
    fill_interior,
    fit_to_grid,
    load_gaussians_ply,
    make_synthetic_cloud,
    particle_volumes,
    save_gaussians_ply,
)
from warpmpm.splats.io import _cov6_from_scale_quat
from warpmpm.splats.scene import SplatScene

_IDX = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _unpack6(cov6: np.ndarray) -> np.ndarray:
    cov6 = np.asarray(cov6)
    n = cov6.shape[0]
    sig = np.zeros((n, 3, 3))
    for c, (i, j) in enumerate(_IDX):
        sig[:, i, j] = cov6[:, c]
        sig[:, j, i] = cov6[:, c]
    return sig


def _elastic_blob(n_grid=40, n=1200, seed=0, cov6=None, cov_mode="step", fused=True,
                  center=(0.2, 0.2, 0.2), half=0.04, E=2.0e6):
    """A small elastic blob loaded with per-particle covariance, for the transport tests."""
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    rng = np.random.default_rng(seed)
    ctr = np.asarray(center, dtype=np.float64)
    pos = (rng.uniform(-half, half, size=(n, 3)) + ctr).astype(np.float32)
    vol = np.full(n, (grid.dx ** 3) / 2.0, np.float32)
    if cov6 is None:
        cov6 = np.zeros((n, 6), np.float32)
        cov6[:, 0] = cov6[:, 3] = cov6[:, 5] = (0.012) ** 2
    s = Solver(grid=grid, device="cpu", fused=fused).load_particles(
        pos, vol, cov=cov6, cov_mode=cov_mode)
    s.set_material(elastic(E=E, nu=0.3, density=1000.0))
    return s, ctr


# --- 1. PLY round-trip -------------------------------------------------------------------
def test_ply_round_trip(tmp_path):
    cloud = make_synthetic_cloud(shape="box", n=1500, sh_degree=0, seed=3)
    path = tmp_path / "cloud.ply"
    save_gaussians_ply(cloud, path)
    back = load_gaussians_ply(path, sh_degree=0)
    assert back.n == cloud.n
    np.testing.assert_allclose(back.pos, cloud.pos, atol=1e-6)
    np.testing.assert_allclose(back.cov, cloud.cov, atol=1e-6)
    np.testing.assert_allclose(back.opacity, cloud.opacity, atol=1e-6)
    np.testing.assert_allclose(back.sh, cloud.sh, atol=1e-6)


# --- 2. Cov construction from scale + quaternion -----------------------------------------
def test_cov_construction_from_scale_quat(tmp_path):
    # a single splat, scale (a, b, c), rotated 90 degrees about z: x -> y, y -> -x, so the
    # world covariance is diag(b^2, a^2, c^2) with zero off-diagonals.
    a, b, c = 0.03, 0.01, 0.02
    scales = np.array([[a, b, c]], dtype=np.float32)
    quats = np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]], dtype=np.float32)
    cov6 = _cov6_from_scale_quat(scales, quats)
    expected = np.array([[b * b, 0.0, 0.0, a * a, 0.0, c * c]], dtype=np.float32)
    np.testing.assert_allclose(cov6, expected, atol=1e-9)

    # the same through the public loader (exp on the stored log-scale, quat normalize)
    cloud = GaussianCloud(pos=np.zeros((1, 3), np.float32), cov=cov6,
                          opacity=np.full((1, 1), 0.9, np.float32),
                          sh=np.zeros((1, 1, 3), np.float32), sh_degree=0,
                          scales=scales, quats=quats)
    path = tmp_path / "one.ply"
    save_gaussians_ply(cloud, path)
    loaded = load_gaussians_ply(path, sh_degree=0)
    np.testing.assert_allclose(loaded.cov, expected, atol=1e-7)


# --- 3. Transform round-trip -------------------------------------------------------------
def test_transform_round_trip():
    cloud = make_synthetic_cloud(shape="box", n=800, seed=1)
    grid = GridConfig(n_grid=48, grid_lim=0.4)
    tf = fit_to_grid(cloud, grid)
    assert isinstance(tf, SimTransform)
    back = tf.from_sim(tf.to_sim(cloud.pos))
    np.testing.assert_allclose(back, cloud.pos, atol=1e-6)
    cov_sim = tf.to_sim_cov(cloud.cov)
    np.testing.assert_allclose(cov_sim, cloud.cov * tf.s * tf.s, rtol=1e-6)
    np.testing.assert_allclose(tf.from_sim_cov(cov_sim), cloud.cov, rtol=1e-6)


# --- 4. Fill: hollow sphere shell and open cup -------------------------------------------
def _sphere_shell(n, radius, center, jitter=0.02, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = radius * (1.0 + rng.uniform(-jitter, jitter, size=(n, 1)))
    return (v * r + np.asarray(center)).astype(np.float32)


def test_fill_hollow_sphere_inside_shell():
    grid_n, grid_lim = 32, 0.4
    dx = grid_lim / grid_n
    center = np.array([0.2, 0.2, 0.2])
    radius = 0.10
    pos = _sphere_shell(20000, radius, center, seed=1)
    op = np.full((pos.shape[0], 1), 0.95, np.float32)
    cov6 = np.zeros((pos.shape[0], 6), np.float32)
    cov6[:, 0] = cov6[:, 3] = cov6[:, 5] = (0.5 * dx) ** 2
    fillers = fill_interior(pos, op, cov6, grid_n, dx, density_thres=1.0, search_thres=0.5)
    assert fillers.shape[0] > 0
    rr = np.linalg.norm(fillers - center, axis=1)
    assert rr.max() < radius              # strictly inside the shell radius


def test_fill_open_cup_excludes_cavity_above():
    # an open-top cup: solid walls and a floor around a hollow, nothing above the rim.
    grid_n, grid_lim = 40, 0.4
    dx = grid_lim / grid_n
    cx = cy = 0.2
    z0 = 0.10           # floor height
    rim = 0.22          # top of the walls
    r_out, r_in = 0.07, 0.055
    rng = np.random.default_rng(0)
    pts = []
    # walls
    for _ in range(30000):
        th = rng.uniform(0, 2 * np.pi)
        rr = rng.uniform(r_in, r_out)
        z = rng.uniform(z0, rim)
        pts.append([cx + rr * np.cos(th), cy + rr * np.sin(th), z])
    # floor disc
    for _ in range(8000):
        th = rng.uniform(0, 2 * np.pi)
        rr = rng.uniform(0, r_out)
        z = rng.uniform(z0, z0 + 2 * dx)
        pts.append([cx + rr * np.cos(th), cy + rr * np.sin(th), z])
    pos = np.asarray(pts, np.float32)
    op = np.full((pos.shape[0], 1), 0.95, np.float32)
    cov6 = np.zeros((pos.shape[0], 6), np.float32)
    cov6[:, 0] = cov6[:, 3] = cov6[:, 5] = (0.5 * dx) ** 2
    fillers = fill_interior(pos, op, cov6, grid_n, dx, density_thres=1.0, search_thres=0.5,
                            exclude_dirs=("+z",))
    assert fillers.shape[0] > 0
    # no filler sits above the rim (the open cavity above the mouth stays empty)
    assert fillers[:, 2].max() <= rim + dx


# --- 5. Volumes --------------------------------------------------------------------------
def test_particle_volumes_sum_to_occupied_volume():
    grid_n, grid_lim = 32, 0.4
    dx = grid_lim / grid_n
    rng = np.random.default_rng(0)
    pos = (rng.uniform(0.1, 0.3, size=(5000, 3))).astype(np.float32)
    vol = particle_volumes(pos, grid_n, dx)
    # every occupied cell contributes exactly one cell volume once summed over its particles
    ci = np.floor(pos / dx).astype(np.int64)
    n_occ = np.unique(ci, axis=0).shape[0]
    assert abs(vol.sum() - n_occ * dx ** 3) / (n_occ * dx ** 3) < 0.01


# --- 6. Cov transport under rigid rotation, fused vs split -------------------------------
def _rotation_run(fused):
    n = 1200
    cov6 = np.zeros((n, 6), np.float32)
    cov6[:, 0] = (0.020) ** 2
    cov6[:, 3] = (0.008) ** 2
    cov6[:, 5] = (0.012) ** 2
    s, ctr = _elastic_blob(n=n, cov6=cov6, cov_mode="step", fused=fused, half=0.04, E=2.0e6)
    omega = 12.0
    r = s.x() - ctr
    v = np.zeros_like(r)
    v[:, 0] = -omega * r[:, 1]
    v[:, 1] = omega * r[:, 0]
    s.set_v(v.astype(np.float32))
    for _ in range(4):
        s.step(dt=1e-4, substeps=10)
    sig0 = _unpack6(cov6)
    return s.cov(), s.R(), sig0, omega * (4 * 10 * 1e-4)


def test_cov_transport_rigid_rotation():
    cov_f, R_f, sig0, angle = _rotation_run(fused=True)
    cov_s, _, _, _ = _rotation_run(fused=False)

    # the shared g2p_particle wp.func carries the cov advection, so the fused pipeline and
    # the split pipeline advect it identically (bitwise on CPU).
    assert np.array_equal(cov_f, cov_s)

    # R is a proper rotation and tracks the imposed spin about z
    orth = np.abs(np.einsum("nij,nkj->nik", R_f, R_f) - np.eye(3)).max()
    assert orth < 1e-4
    assert abs(np.linalg.det(R_f).mean() - 1.0) < 1e-4
    ang = np.arctan2(R_f[:, 1, 0], R_f[:, 0, 0])
    assert abs(np.median(ang) - angle) < 0.3 * angle

    # advected covariance matches R Sigma0 R^T
    RS = np.einsum("nij,njk,nlk->nil", R_f, sig0, R_f)
    rel = np.linalg.norm(_unpack6(cov_f) - RS, axis=(1, 2)) / np.linalg.norm(sig0, axis=(1, 2))
    assert np.median(rel) < 5e-3


# --- 7. Cov transport under uniaxial squeeze ---------------------------------------------
def test_cov_transport_uniaxial_squeeze():
    s, ctr = _elastic_blob(n=1400, cov_mode="step", half=0.04, E=2.0e5)
    cov0 = s.cov().copy()
    k = 8.0
    for _ in range(8):
        r = s.x() - ctr
        v = np.zeros_like(r)
        v[:, 0] = 0.5 * k * r[:, 0]
        v[:, 1] = 0.5 * k * r[:, 1]
        v[:, 2] = -k * r[:, 2]
        s.set_v(v.astype(np.float32))
        s.step(dt=1e-4, substeps=10)
    cov = s.cov()
    assert np.median(cov[:, 5] / cov0[:, 5]) < 0.98      # zz eigenvalue shrinks
    assert np.median(cov[:, 0] / cov0[:, 0]) > 1.02      # xx grows


# --- 8. cov_mode step vs from_F ----------------------------------------------------------
def _small_deform(cov_mode):
    s, ctr = _elastic_blob(n=1400, seed=1, cov_mode=cov_mode, half=0.04, E=2.0e5)
    k = 3.0
    for _ in range(5):
        r = s.x() - ctr
        v = np.zeros_like(r)
        v[:, 0] = 0.5 * k * r[:, 0]
        v[:, 1] = 0.5 * k * r[:, 1]
        v[:, 2] = -k * r[:, 2]
        s.set_v(v.astype(np.float32))
        s.step(dt=1e-4, substeps=10)
    return s.cov()


def test_cov_mode_step_matches_from_F():
    cs = _small_deform("step")
    cf = _small_deform("from_F")
    diag = [0, 3, 5]
    rel = np.abs(cs[:, diag] - cf[:, diag]) / np.abs(cf[:, diag])
    assert rel.max() < 1e-4
    # the run actually deformed, so this is not a vacuous match
    assert np.median(cs[:, 5]) < 0.99 * (0.012) ** 2


# --- 9. init_cov unchanged after stepping (aliasing regression) --------------------------
def test_init_cov_unchanged_after_stepping():
    n = 1000
    cov6 = np.zeros((n, 6), np.float32)
    cov6[:, 0] = (0.02) ** 2
    cov6[:, 3] = (0.008) ** 2
    cov6[:, 5] = (0.012) ** 2
    s, ctr = _elastic_blob(n=n, cov6=cov6, cov_mode="step", half=0.04, E=2.0e6)
    omega = 12.0
    r = s.x() - ctr
    v = np.zeros_like(r)
    v[:, 0] = -omega * r[:, 1]
    v[:, 1] = omega * r[:, 0]
    s.set_v(v.astype(np.float32))
    init0 = s._sim.mpm_state.particle_init_cov.numpy().copy()
    for _ in range(5):
        s.step(dt=1e-4, substeps=10)
    init1 = s._sim.mpm_state.particle_init_cov.numpy()
    assert np.array_equal(init0, init1)
    # advection did run, so the rest-frame array is untouched while the live cov moved
    assert np.abs(s.cov()[:, 0] - cov6[:, 0]).max() > 0.0


# --- 10. Appearance modes ----------------------------------------------------------------
def _two_color_cloud():
    rng = np.random.default_rng(0)
    left = rng.uniform([-0.1, -0.05, -0.05], [-0.02, 0.05, 0.05], size=(400, 3))
    right = rng.uniform([0.02, -0.05, -0.05], [0.1, 0.05, 0.05], size=(400, 3))
    pos = np.concatenate([left, right]).astype(np.float32)
    sh = np.zeros((800, 1, 3), np.float32)
    from warpmpm.splats.appearance import rgb_to_sh_dc
    sh[:400, 0, :] = rgb_to_sh_dc(np.array([1.0, 0.0, 0.0]))    # red left
    sh[400:, 0, :] = rgb_to_sh_dc(np.array([0.0, 0.0, 1.0]))    # blue right
    cov = np.zeros((800, 6), np.float32)
    cov[:, 0] = cov[:, 3] = cov[:, 5] = 0.01 ** 2
    op = np.full((800, 1), 0.9, np.float32)
    return GaussianCloud(pos=pos, cov=cov, opacity=op, sh=sh, sh_degree=0)


def test_appearance_inherit_carries_neighbor_color():
    cloud = _two_color_cloud()
    left_fillers = np.array([[-0.06, 0.0, 0.0]], np.float32)
    right_fillers = np.array([[0.06, 0.0, 0.0]], np.float32)
    _, _, sh_left = assign_filler_appearance(cloud, left_fillers, mode="inherit", k=8)
    _, _, sh_right = assign_filler_appearance(cloud, right_fillers, mode="inherit", k=8)
    col_left = eval_sh(0, sh_left, np.array([[0.0, 0.0, 1.0]], np.float32))[0]
    col_right = eval_sh(0, sh_right, np.array([[0.0, 0.0, 1.0]], np.float32))[0]
    assert col_left[0] > 0.8 and col_left[2] < 0.2     # left fillers red
    assert col_right[2] > 0.8 and col_right[0] < 0.2   # right fillers blue


def test_appearance_invisible_and_flat():
    cloud = make_synthetic_cloud(shape="box", n=600, sh_degree=0, seed=0)
    grid = GridConfig(n_grid=32, grid_lim=0.4)
    # invisible: n_visible collapses to the original count
    scene = SplatScene(cloud, grid=grid, material=elastic(density=1000), device="cpu",
                       fill=False, filler_appearance="invisible")
    assert scene.n_visible == scene.n_gaussians

    # flat: fillers carry the requested DC color
    fillers = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], np.float32)
    _, op, sh = assign_filler_appearance(cloud, fillers, mode="flat", color=(0.2, 0.7, 0.4),
                                         opacity=0.8)
    col = eval_sh(0, sh, np.tile([0.0, 0.0, 1.0], (2, 1)).astype(np.float32))
    np.testing.assert_allclose(col, np.tile([0.2, 0.7, 0.4], (2, 1)), atol=1e-5)
    np.testing.assert_allclose(op.reshape(-1), 0.8, atol=1e-6)


# --- 11. eval_sh degree 1 analytic -------------------------------------------------------
def test_eval_sh_degree1_analytic():
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    sh = np.zeros((1, 4, 3), np.float32)
    sh[0, 0, :] = 0.3            # DC
    sh[0, 1, :] = 0.10           # y band
    sh[0, 2, :] = 0.20           # z band
    sh[0, 3, :] = -0.15          # x band
    d = np.array([[0.3, -0.4, 0.86602540378]], np.float32)  # unit-ish direction
    d = d / np.linalg.norm(d)
    x, y, z = d[0]
    expected = C0 * 0.3 - C1 * y * 0.10 + C1 * z * 0.20 - C1 * x * (-0.15) + 0.5
    out = eval_sh(1, sh, d)
    np.testing.assert_allclose(out[0], np.full(3, expected), atol=1e-6)


# --- 12. SplatScene smoke ----------------------------------------------------------------
def test_splatscene_smoke():
    cloud = make_synthetic_cloud(shape="box", n=1500, sh_degree=0, seed=0)
    grid = GridConfig(n_grid=32, grid_lim=0.4)
    scene = SplatScene(cloud, grid=grid, material=elastic(E=3e5, nu=0.3, density=1000),
                       device="cpu", fill=True, cov_mode="step")
    for _ in range(10):
        scene.step(dt=1e-4, substeps=10)
    st = scene.state()
    nv = scene.n_visible
    assert tuple(st["pos"].shape) == (nv, 3)
    assert tuple(st["cov6"].shape) == (nv, 6)
    assert tuple(st["R"].shape) == (nv, 3, 3)
    assert str(st["pos"].device) == "cpu"
    assert not bool(np.isnan(st["pos"].cpu().numpy()).any())
    cols = scene.colors(camera_pos=(0.2, -0.3, 0.4))
    assert cols.shape == (nv, 3)
    assert cols.min() >= 0.0 and cols.max() <= 1.0
