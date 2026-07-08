"""Minimal mesh->SDF pour: two watertight cup meshes voxelized to signed-distance
colliders, one tilted by a scripted angular velocity. This is the API demo for the
general mesh->SDF collider path (warpmpm.geometry.build_sdf_cached + add_sdf_collider),
which handles arbitrary watertight meshes. For the full Franka pour with metrics,
leak audit, and analytic revolved glasses, see examples/pour_franka.py.

Run:  python -m examples.pour_glass            # simulate + render an mp4
      python -m examples.pour_glass --no-render # simulate only (prints transfer stats)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from warpmpm import GridConfig, Solver
from warpmpm.geometry import build_sdf_cached, make_cup_mesh
from warpmpm.materials import newtonian

OUT = ROOT / "out" / "pour"
CACHE = ROOT / "out" / "sdf_cache"

# cup geometry (metres): a small open-top cup
CUP = dict(inner_radius=0.035, wall_thickness=0.007, height=0.085, base_thickness=0.009)


def _fill_cavity(center, dx, radius, z_lo, z_hi, seed=0):
    """Jittered fluid particles filling a cylinder inside the cup cavity (body frame -> world)."""
    h = dx / 2.0
    xs = np.arange(-radius, radius, h)
    zs = np.arange(z_lo, z_hi, h)
    g = np.stack(np.meshgrid(xs, xs, zs, indexing="ij"), -1).reshape(-1, 3)
    g = g[g[:, 0] ** 2 + g[:, 1] ** 2 < radius**2]
    rng = np.random.default_rng(seed)
    g = g + rng.uniform(-0.25 * h, 0.25 * h, size=g.shape)
    pts = (g + np.asarray(center)).astype(np.float32)
    vol = np.full(len(pts), h**3, dtype=np.float32)
    return pts, vol


def run_pour(n_grid=64, grid_lim=0.4, eta=4.0, density=1000.0, bulk=2.0e5,
             tilt_rate=3.0, settle_s=0.10, pour_s=0.65, drain_s=0.30,
             dt=1.5e-4, substeps=6, sdf_res=56, log=print):
    """Simulate the pour. Returns (frames, poses, floor_z) where frames[i] is (positions[N,3],
    speed[N]) and poses[i] is the pouring cup (center, quat). One frame per control tick."""
    grid = GridConfig(n_grid=n_grid, grid_lim=grid_lim)
    dx = grid.dx
    floor_z = 4.0 * dx

    cv, cf = make_cup_mesh(n_theta=56, **CUP)
    probe = np.array([CUP["inner_radius"] + 0.5 * CUP["wall_thickness"], 0.0, 0.04])
    sdf = build_sdf_cached(cv, cf, res=sdf_res, margin_cells=4, interior_probe=probe,
                           cache_dir=CACHE)
    log(f"[pour] cup SDF res={sdf.res} cell={sdf.cell * 1000:.2f}mm")

    # two cups: the pouring cup (left, raised, will tilt) and the receiving cup (right, on the
    # floor, static). The pourer is raised so its lip clears the receiver rim, and the receiver
    # sits under where the lip swings to at ~90 deg of tilt (x ~ pour_x + cup height).
    pour_c = np.array([0.36 * grid_lim, 0.5 * grid_lim, floor_z + 0.14])
    recv_c = np.array([pour_c[0] + CUP["height"] + 0.5 * CUP["inner_radius"],
                       0.5 * grid_lim, floor_z + 0.001])

    pts, vol = _fill_cavity(pour_c, dx, radius=CUP["inner_radius"] - 1.5 * dx,
                            z_lo=CUP["base_thickness"] + dx, z_hi=0.060)
    log(f"[pour] fluid particles: {len(pts)}  grid={n_grid}^3 dx={dx * 1000:.2f}mm")

    s = Solver(grid=grid, device="cpu").load_particles(pts, vol)
    s.set_material(newtonian(eta=eta, density=density, bulk_modulus=bulk))
    s.add_plane((0, 0, floor_z), (0, 0, 1), "slip", friction=0.2)
    pour_h = s.add_sdf_collider(sdf, center=pour_c, surface="separable", friction=0.35)
    recv_h = s.add_sdf_collider(sdf, center=recv_c, surface="separable", friction=0.35)

    n_settle = int(settle_s / (dt * substeps))
    n_pour = int(pour_s / (dt * substeps))
    n_drain = int(drain_s / (dt * substeps))
    frames, poses = [], []

    def snap():
        x = s.x(); v = s.v()
        frames.append((x.copy(), np.linalg.norm(v, axis=1)))
        p = s._sim.collider_params[pour_h]
        c = np.array([p.center[0], p.center[1], p.center[2]])
        q = np.array([p.quat[0], p.quat[1], p.quat[2], p.quat[3]])
        poses.append((c, q))

    def _stable(i):
        if i % 40 == 0 and np.isnan(s.x()).any():
            log(f"[pour] WARNING: NaN at tick {i}; stopping early (reduce dt / soften bulk)")
            return False
        return True

    tick = 0
    # phase 1: settle upright
    for _ in range(n_settle):
        s.step(dt, substeps); snap(); tick += 1
        if not _stable(tick):
            return frames, poses, floor_z, (pour_c, recv_c)
    # phase 2: tilt the pouring cup forward (+y axis rotation tips the +x rim down) and pour
    s.set_sdf_pose(pour_h, omega=(0.0, tilt_rate, 0.0))
    for _ in range(n_pour):
        s.step(dt, substeps); snap(); tick += 1
        if not _stable(tick):
            return frames, poses, floor_z, (pour_c, recv_c)
    # phase 3: stop tilting (cup roughly inverted), let it drain/settle
    s.set_sdf_pose(pour_h, omega=(0.0, 0.0, 0.0))
    for _ in range(n_drain):
        s.step(dt, substeps); snap(); tick += 1
        if not _stable(tick):
            return frames, poses, floor_z, (pour_c, recv_c)

    x0 = frames[0][0]
    xf = frames[-1][0]
    pour_foot = pour_c[0] + CUP["inner_radius"] + dx          # past the pourer footprint
    recv_lo, recv_hi = recv_c[0] - 1.5 * CUP["inner_radius"], recv_c[0] + 1.5 * CUP["inner_radius"]
    poured = int((xf[:, 0] > pour_foot).sum())
    in_recv = int(((xf[:, 0] > recv_lo) & (xf[:, 0] < recv_hi)
                   & (xf[:, 2] > floor_z + 0.003)).sum())
    n = len(xf)
    log(f"[pour] frames={len(frames)} n={n} inverted={s.inverted_count()} "
        f"nan={int(np.isnan(xf).any())}")
    log(f"[pour] x: start [{x0[:, 0].min():.3f},{x0[:, 0].max():.3f}] -> "
        f"end [{xf[:, 0].min():.3f},{xf[:, 0].max():.3f}]  (pourer x={pour_c[0]:.3f}, "
        f"receiver x={recv_c[0]:.3f})")
    log(f"[pour] poured out of the pourer: {poured}/{n} ({100 * poured / n:.0f}%); "
        f"landed in the receiver: {in_recv}/{n} ({100 * in_recv / n:.0f}%)")
    return frames, poses, floor_z, (pour_c, recv_c)


def render(frames, poses, floor_z, centers, path, fps=24, stride=6):
    """Matplotlib 3D scatter of the fluid (coloured by speed) with the two cup rims drawn,
    written to an mp4. Self-contained (no pyvista needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    pour_c, recv_c = centers
    th = np.linspace(0, 2 * np.pi, 40)
    rim_local = np.stack([CUP["inner_radius"] * np.cos(th), CUP["inner_radius"] * np.sin(th),
                          np.full_like(th, CUP["height"])], -1)
    base_local = rim_local.copy(); base_local[:, 2] = 0.0

    def rot(q, p):
        x, y, z, w = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        return p @ R.T

    OUT.mkdir(parents=True, exist_ok=True)
    vmax = max(1e-6, np.percentile(np.concatenate([f[1] for f in frames]), 98))
    fr_dir = OUT / "frames"
    fr_dir.mkdir(exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    lim = 0.4
    for i in range(0, len(frames), stride):
        x, spd = frames[i]
        cpos, cquat = poses[i]
        fig = plt.figure(figsize=(6, 5), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=spd, cmap="viridis", vmin=0, vmax=vmax,
                   s=4, alpha=0.7, linewidths=0)
        for loc, q, col in [(cpos, cquat, "0.25"), (recv_c, np.array([0, 0, 0, 1.0]), "0.5")]:
            for lp in (rim_local, base_local):
                w = rot(q, lp) + loc
                ax.plot(w[:, 0], w[:, 1], w[:, 2], color=col, lw=1.3)
        ax.set_xlim(0.05, 0.05 + lim * 0.7); ax.set_ylim(0.1, 0.1 + lim * 0.7)
        ax.set_zlim(floor_z, floor_z + 0.28)
        ax.set_box_aspect((0.7, 0.7, 0.6)); ax.view_init(elev=14, azim=-72)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"mesh-SDF pour: cup tilts, fluid transfers  ({i}/{len(frames)})", fontsize=9)
        fig.tight_layout()
        fr = fr_dir / f"f{i:04d}.png"
        fig.savefig(fr); plt.close(fig)
        writer.append_data(imageio.imread(fr))
    writer.close()
    return path


if __name__ == "__main__":
    do_render = "--no-render" not in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    frames, poses, floor_z, centers = run_pour()
    if do_render:
        p = render(frames, poses, floor_z, centers, OUT / "pour_glass.mp4")
        print(f"[pour] wrote {p}")
