"""4D splat baking: reconstruction, interpolation, rotation continuity, io, compression.

Most tests use analytic trajectories (rigid translation, rotation, smooth stretch) so
truth is exact; one end-to-end test bakes a short real SplatScene run.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("plyfile")

from warpmpm.splats import Baked4DSplats, bake, load_gaussians_ply, make_synthetic_cloud


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _analytic_frames(n=200, n_frames=60, dt=0.02, total_angle=0.5 * np.pi, seed=0):
    """Splats translating on a parabola, rotating about z, and breathing in scale."""
    rng = np.random.default_rng(seed)
    cloud = make_synthetic_cloud(shape="box", n=n, sh_degree=0, seed=seed)
    pos0, cov0 = cloud.pos.astype(np.float64), cloud.cov.astype(np.float64)
    sigma0 = np.zeros((n, 3, 3))
    idx = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    for c, (i, j) in enumerate(idx):
        sigma0[:, i, j] = cov0[:, c]
        sigma0[:, j, i] = cov0[:, c]
    drift = rng.uniform(-0.02, 0.02, size=3)

    frames, times = [], []
    for f in range(n_frames):
        t = f * dt
        a = total_angle * t / ((n_frames - 1) * dt)
        R = _rot_z(a)
        stretch = 1.0 + 0.15 * np.sin(2.0 * np.pi * t)          # smooth isotropic breathing
        pos = (pos0 - pos0.mean(0)) @ R.T + pos0.mean(0) + drift * t + [0, 0, -0.4 * t * t]
        sig = (stretch ** 2) * np.einsum("ij,njk,lk->nil", R, sigma0, R)
        cov6 = np.stack([sig[:, i, j] for (i, j) in idx], axis=1)
        frames.append({"pos": pos.astype(np.float32), "cov6": cov6.astype(np.float32),
                       "R": np.broadcast_to(R, (n, 3, 3)).astype(np.float32).copy(),
                       "opacity": cloud.opacity, "sh": cloud.sh})
        times.append(t)
    return frames, np.array(times)


def test_reconstruction_within_tolerance():
    frames, times = _analytic_frames()
    tol_x = 1e-4
    baked = bake(frames, times, tol_x=tol_x, tol_cov=1e-2)
    for f, t in zip(frames[:: len(frames) // 6], times[:: len(frames) // 6], strict=True):
        st = baked.at(float(t))
        assert np.abs(st["pos"] - f["pos"]).max() <= 2 * tol_x
        rel = (np.linalg.norm(st["cov6"] - f["cov6"], axis=1)
               / (np.linalg.norm(f["cov6"], axis=1) + 1e-30))
        assert rel.max() <= 2e-2
    rep = baked.report()
    assert rep["max_pos_err"] <= tol_x
    assert rep["max_cov_rel_err"] <= 1e-2


def test_interpolation_between_recorded_frames():
    # bake the 1x recording, evaluate at the midpoints, compare against a 2x recording
    frames2, times2 = _analytic_frames(n_frames=121, dt=0.01)
    frames1, times1 = frames2[::2], times2[::2]
    tol_x = 1e-4
    baked = bake(list(frames1), np.array(times1), tol_x=tol_x, tol_cov=1e-2)
    mid_idx = range(1, len(frames2) - 1, 2)
    worst = max(np.abs(baked.at(float(times2[i]))["pos"] - frames2[i]["pos"]).max()
                for i in mid_idx)
    assert worst <= 20 * tol_x        # midpoints are unseen data, allow a looser bound


def test_quaternion_continuity_full_turn():
    frames, times = _analytic_frames(n=50, n_frames=90, total_angle=2.0 * np.pi)
    baked = bake(frames, times)
    for t in np.linspace(times[0], times[-1], 25):
        R = baked.at(float(t))["R"]
        err = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max()
        assert err < 1e-4             # orthonormal everywhere: no sign-flip glitches
        expected = _rot_z(2.0 * np.pi * t / times[-1])
        assert np.abs(R - expected).max() < 2e-3
    # end state is a full turn: R(t1) is the identity again
    assert np.abs(baked.at(float(times[-1]))["R"] - np.eye(3)).max() < 5e-2


def test_interpolated_covariance_remains_positive_definite():
    frames, times = _analytic_frames(n=40, n_frames=14, total_angle=np.pi)
    # Strong anisotropic breathing makes entry-wise cubic interpolation a particularly
    # poor parameterization; the matrix-log path must remain SPD at unseen times.
    for i, frame in enumerate(frames):
        frame["cov6"][:, 0] *= np.exp(3.0 * np.sin(2.0 * np.pi * i / (len(frames) - 1)))
    baked = bake(frames, times, tol_cov=2e-2)
    idx = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    for t in np.linspace(times[0], times[-1], 101):
        cov6 = baked.at(float(t))["cov6"]
        cov = np.zeros((len(cov6), 3, 3))
        for c, (a, b) in enumerate(idx):
            cov[:, a, b] = cov6[:, c]
            cov[:, b, a] = cov6[:, c]
        assert np.linalg.eigvalsh(cov).min() > 0.0


def test_save_load_round_trip(tmp_path):
    frames, times = _analytic_frames(n=80, n_frames=30)
    baked = bake(frames, times)
    path = tmp_path / "clip.npz"
    baked.save(path)
    back = Baked4DSplats.load(path)
    assert np.array_equal(back.coef_pos, baked.coef_pos)
    assert np.array_equal(back.coef_cov, baked.coef_cov)
    assert np.array_equal(back.coef_quat, baked.coef_quat)
    assert np.array_equal(back.rotation_times, baked.rotation_times)
    assert np.array_equal(back.knots, baked.knots)
    assert back.meta["n_frames"] == baked.meta["n_frames"]
    st_a, st_b = baked.at(0.5), back.at(0.5)
    assert np.array_equal(st_a["pos"], st_b["pos"])


def test_compression_beats_per_frame_storage(tmp_path):
    frames, times = _analytic_frames(n=300, n_frames=60)
    baked = bake(frames, times)
    rep = baked.report()
    assert rep["compression_ratio"] is not None and rep["compression_ratio"] > 2.0
    path = tmp_path / "clip.npz"
    baked.save(path)
    assert path.stat().st_size < rep["per_frame_ply_bytes"]


def test_write_frames_upsamples_and_loads(tmp_path):
    frames, times = _analytic_frames(n=60, n_frames=20, dt=0.05)
    baked = bake(frames, times)
    out = tmp_path / "upsampled"
    paths = baked.write_frames(out)                        # default: 2x recorded count
    assert len(paths) == 2 * len(frames)
    c = load_gaussians_ply(paths[0])                       # default loader, DC file
    assert c.n == 60
    assert (out / "manifest.json").exists()


def test_directory_bake_has_no_rotation_track(tmp_path):
    # a FrameRecorder directory carries PLYs without R; the bake must still work
    from warpmpm.splats.export import export_frame_ply

    frames, times = _analytic_frames(n=40, n_frames=12)
    rec = tmp_path / "rec"
    rec.mkdir()
    for i, f in enumerate(frames):
        export_frame_ply(f, rec / f"frame_{i:04d}.ply", sh_mode="dc")
    (rec / "manifest.json").write_text('{"frame_count": 12, "fps": 50, "sh_mode": "dc"}')
    baked = bake(rec)
    assert baked.coef_quat is None
    st = baked.at(float(times[3]))
    assert np.allclose(st["R"], np.eye(3), atol=1e-6)      # identity fallback
    assert np.abs(st["pos"] - frames[3]["pos"]).max() < 5e-4


def test_scene_end_to_end_bake():
    # a real short sim: record states in memory, bake, evaluate mid-clip sanely
    from warpmpm.core.solver import GridConfig
    from warpmpm.materials import newtonian
    from warpmpm.splats.scene import SplatScene

    cloud = make_synthetic_cloud(shape="box", n=700, sh_degree=0, seed=4)
    scene = SplatScene(cloud, grid=GridConfig(n_grid=32, grid_lim=0.4),
                       material=newtonian(eta=60.0, density=1200.0).with_yield(600.0),
                       device="cpu", fill=False, floor="sticky")
    states, times = [], []
    dt, sub = 2.0e-5, 20
    for f in range(10):
        scene.step(dt=dt, substeps=sub)
        states.append({k: v.detach().cpu().numpy().copy()
                       for k, v in scene.state().items()})
        times.append((f + 1) * dt * sub)
    baked = bake(states, np.array(times))
    st = baked.at(0.5 * (times[0] + times[-1]))
    assert st["pos"].shape == (scene.n_visible, 3)
    assert np.isfinite(st["pos"]).all() and np.isfinite(st["cov6"]).all()
    assert baked.report()["compression_ratio"] > 1.0
