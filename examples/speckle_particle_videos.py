"""Speckle PARTICLE renders (the view CoTracker actually tracks) of the 1x and 1.5x squeeze.

These are the gray-speckle particle videos (front orthographic, per-particle brightness =
material-locked texture), as opposed to the marching-cubes surface render. Re-runs the same
quasi-2D plane-strain squeeze as the real-data datasets and renders via the validated
perception speckle renderer. Run:  ../.venv/bin/python examples/speckle_particle_videos.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, newtonian
from warpmpm.coupling.backend import WarpMPMBackend

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parents[1] / "out" / "speckle_particles"


def dump_slab(scale, n_grid=64, v_plate=0.08, press_strain=0.5, dt=1.0e-4,
              substeps=20, frame_stride=3, device="auto"):
    """Quasi-2D plane-strain squeeze; record particle positions x[F,N,3] + times."""
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4); dx = grid.dx; s_lin = float(scale) ** 0.5
    col_w = 0.13 * s_lin; col_h = 0.06 * s_lin; slab = 6 * dx; cx = cy = 0.2; floor = 3 * dx
    h = dx / 2
    xs = np.arange(cx - 0.5 * col_w + 0.5 * h, cx + 0.5 * col_w, h)
    ys = np.arange(cy - 0.5 * slab + 0.5 * h, cy + 0.5 * slab, h)
    zs = np.arange(floor + 0.5 * h, floor + col_h, h)
    pos = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    pos += np.random.default_rng(0).uniform(-0.2 * h, 0.2 * h, pos.shape).astype(np.float32)
    s = Solver(grid=grid, device=device).load_particles(pos, np.full(len(pos), h ** 3, np.float32))
    s.set_material(newtonian(eta=40.0, density=1000.0, bulk_modulus=9.0e5).with_yield(200.0))
    pad = 3 * dx
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    s.add_plane((pad, 0, 0), (1, 0, 0), "slip")
    s.add_plane((grid.grid_lim - pad, 0, 0), (-1, 0, 0), "slip")
    s.add_plane((0, cy - 0.5 * slab, 0), (0, 1, 0), "slip")
    s.add_plane((0, cy + 0.5 * slab, 0), (0, -1, 0), "slip")
    bh = (0.5 * col_w + 0.012, 0.5 * slab + 0.012, 0.6 * dx)
    be = WarpMPMBackend(solver=s); z = floor + col_h + bh[2]; tool = be.attach_tool((cx, cy, z), bh)
    fdt = dt * substeps; nf = round(press_strain * col_h / v_plate / fdt); prev = z
    X, T = [], []
    for f in range(nf + 1):
        zn = z - v_plate * fdt if f > 0 else z; vz = (zn - prev) / fdt
        if f > 0:
            be.set_tool_kinematics(tool, center=(cx, cy, prev), velocity=(0, 0, vz))
            be.step(dt, substeps)
        z = zn; prev = zn
        if f % frame_stride == 0:
            X.append(s.x().copy()); T.append(f * fdt)
    return np.array(X), np.array(T)


def run(device="auto"):
    from perception.render_collapse3d_speckle import render
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for scale, tag in ((1.0, "squeeze_1x"), (1.5, "squeeze_1p5x")):
        X, times = dump_slab(scale, device=device)
        npz = OUT / f"{tag}.npz"
        np.savez(npz, x=X, times=times)
        render(str(npz), out_dir=str(OUT), frame_stride=1, dot_px=3.2)
        mp4 = OUT / f"{tag}_speckle.mp4"
        print(f"[{tag}] {X.shape[0]} frames, {X.shape[1]} particles -> {mp4}")
        out[tag] = str(mp4)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="Warp MPM device, e.g. cuda:0 or cuda:1")
    args = parser.parse_args()
    run(device=args.device)
