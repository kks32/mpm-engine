"""Gaussian-splat scene simulated with the warpmpm engine, PhysGaussian style.

Loads or synthesizes a 3D Gaussian-splat cloud, fits it into the grid, fills the interior
with solid particles so the thin splat shell behaves as a body, then simulates: positions
advect with the material, each splat's covariance deforms, and its spherical harmonics
rotate by the polar rotation of the deformation gradient (applied at render time by
inverse-rotating the view direction). The default scene drops a splat box of dough, lets it
settle, then presses it with a scripted box collider driven through the public solver.

Run:  python examples/splat_sim.py
      python examples/splat_sim.py --material elastic --frames 30
      python examples/splat_sim.py --no-fill --device cpu
      python examples/splat_sim.py --ply path/to/point_cloud.ply --material sand

Outputs: out/splat_sim.mp4 (preview frames of the splat centers, colored by their SH color
for a fixed camera and sized by covariance).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import device_cli, write_mp4
from warpmpm.materials import elastic, granular, newtonian
from warpmpm.splats import load_gaussians_ply, make_synthetic_cloud
from warpmpm.splats.render import preview_frame
from warpmpm.splats.scene import SplatScene

OUT = Path(__file__).resolve().parents[1] / "out"


def _material(name: str):
    if name == "dough":
        return newtonian(eta=80.0, density=1200.0).with_yield(900.0)
    if name == "elastic":
        return elastic(E=3.0e5, nu=0.3, density=1000.0)
    if name == "sand":
        return granular(mu_s=0.4, delta_mu=0.26, I0=0.3, density=1590.0)
    raise ValueError(f"unknown material {name!r}")


def run(material="dough", ply=None, fill=True, filler_appearance="inherit", filler_k=8,
        frames=40, grid_n=48, dt=2.0e-5, substeps=20, rasterize=False, device="auto",
        out_name="splat_sim.mp4", render=True):
    from warpmpm.core.solver import GridConfig

    grid = GridConfig(n_grid=grid_n, grid_lim=0.4)
    if ply is not None:
        cloud = load_gaussians_ply(ply)
        print(f"loaded {cloud.n} gaussians from {ply}")
    else:
        cloud = make_synthetic_cloud(shape="box", n=6000, sh_degree=0, seed=0)

    filler_kwargs = {"k": filler_k} if filler_appearance == "inherit" else {}
    scene = SplatScene(cloud, grid=grid, material=_material(material), device=device,
                       fill=fill, filler_appearance=filler_appearance,
                       filler_kwargs=filler_kwargs, cov_mode="step", floor="sticky")
    n_filler = scene.solver.n_particles - scene.n_gaussians
    print(f"gaussians={scene.n_gaussians} fillers={n_filler} visible={scene.n_visible} "
          f"floor_z={scene.floor_z:.3f}")

    # a scripted box collider that descends onto the settled cloud after a short settle
    x0 = scene.solver.x()
    lo, hi = x0.min(0), x0.max(0)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    box_half = (0.5 * (hi[0] - lo[0]) + 0.01, 0.5 * (hi[1] - lo[1]) + 0.01, 0.6 * grid.dx)
    clearance = 0.01
    z_top = hi[2] + box_half[2] + clearance
    z_stop = scene.floor_z + box_half[2] + 0.6 * (hi[2] - lo[2])
    settle = max(6, frames // 4)
    dt_ctrl = dt * substeps
    v_desc = 0.5           # fixed, stable descent speed (m/s); stop clamps the depth
    tool = scene.solver.add_box((cx, cy, z_top), box_half, velocity=(0.0, 0.0, 0.0))

    frame_dir = Path(tempfile.mkdtemp(prefix="splat_"))
    camera = None
    z_box = z_top
    for f in range(frames):
        if f >= settle and z_box > z_stop:
            v_now = -min(v_desc, (z_box - z_stop) / dt_ctrl)   # never overshoot the stop
            scene.solver.set_box(tool, center=(cx, cy, z_box), velocity=(0.0, 0.0, v_now))
            z_box = z_box + v_now * dt_ctrl
        scene.step(dt=dt, substeps=substeps)
        v = scene.solver.v()
        vmax = float(np.sqrt((v * v).sum(1)).max())
        print(f"frame {f:3d}  max|v|={vmax:6.3f} m/s  fillers={n_filler}")
        if render:
            st = scene.state()
            if camera is None:
                camera = _fixed_camera(st)
            preview_frame(st, camera=camera, path=frame_dir / f"f_{f:04d}.png",
                          sh_degree=scene.sh_degree)

    if rasterize:
        _try_rasterize(scene)

    mp4 = None
    if render:
        mp4 = write_mp4(frame_dir, OUT / out_name, fps=12)
    return {"n_gaussians": scene.n_gaussians, "n_filler": n_filler,
            "n_visible": scene.n_visible, "mp4": None if mp4 is None else str(mp4)}


def _fixed_camera(state, pad=1.4):
    """One camera fixed from the first frame so the video does not zoom frame to frame."""
    x = state["pos"].detach().cpu().numpy()
    lo, hi = x.min(0), x.max(0)
    center = 0.5 * (lo + hi)
    span = float(np.max(hi - lo)) * pad + 1e-6
    pos = center + np.array([1.6, -1.8, 1.2]) * span
    return {"pos": pos.astype(np.float32), "elev": 18.0, "azim": -60.0,
            "center": center.astype(np.float32), "span": span}


def _try_rasterize(scene):
    from warpmpm.splats.render import rasterize_inria
    try:
        rasterize_inria(scene.state(), camera={})
    except ImportError as exc:
        print(f"rasterize: {exc}")
    except (KeyError, TypeError):
        print("rasterize: diff_gaussian_rasterization is importable but needs a full camera "
              "(world_view_transform, full_proj_transform, tanfovx, tanfovy) from a real "
              "checkpoint; pass one to rasterize_inria to use this path.")


def main():
    parser = device_cli(description="Gaussian-splat scene on warpmpm", no_render=True)
    parser.add_argument("--ply", default=None, help="INRIA-layout point_cloud.ply to load")
    parser.add_argument("--material", default="dough", choices=("dough", "elastic", "sand"))
    parser.add_argument("--fill", dest="fill", action="store_true", default=True)
    parser.add_argument("--no-fill", dest="fill", action="store_false")
    parser.add_argument("--filler-appearance", default="inherit",
                        choices=("inherit", "invisible", "flat"))
    parser.add_argument("--filler-k", type=int, default=8)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--rasterize", action="store_true",
                        help="use rasterize_inria when diff_gaussian_rasterization is importable")
    args = parser.parse_args()
    run(material=args.material, ply=args.ply, fill=args.fill,
        filler_appearance=args.filler_appearance, filler_k=args.filler_k,
        frames=args.frames, grid_n=args.grid, rasterize=args.rasterize,
        device=args.device, render=not args.no_render)


if __name__ == "__main__":
    main()
