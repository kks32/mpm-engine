"""Matched-work dam-break throughput benchmark, the warpmpm side of a claymore
head-to-head.

Claymore (Wang, Qiu et al., ACM TOG 2020; ClaymoreUW fork) is the specialized
CUDA MLS-MPM this engine's fused pass was ported from. Its remaining edge,
block-shared-memory P2G arenas, was deliberately not ported (needs wp.tile or
native CUDA; docs/performance.md "what was ported"). This benchmark measures
OUR particle-updates/s on a dam break whose work profile (particle count, grid
resolution, substep count) is set to MATCH a claymore run, so the two binaries
can be compared on the same GPU doing the same amount of work. Constitutive
details differ (their JFluid EOS vs our weakly compressible fluid); the
contract is matched work, not matched trajectories.

Protocol on a GPU node (e.g. Vista GH200):

1. Build ClaymoreUW with the GPU's arch (GH200: add `-arch=sm_90` to
   TARGET_CUDA_ARCH in claymore/setup_cuda.cmake), then run its OSU_LWF project
   with a scene JSON. Note from its log: particle count, grid dx and domain
   (grid cells = domain/dx per axis), dt, and the wall time per step or frame.
2. Run this script with those numbers:
       python benchmarks/bench_vs_claymore.py --n-particles 5000000 \
           --n-grid 256 --substeps 200
3. Compare million-particle-updates/s (MPUP/s). Report both numbers with the
   GPU model; claymore's shared-memory P2G is expected to win the particle
   pass, and the measured ratio is the honest price of this engine's
   portability.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from warpmpm import GridConfig, Solver
from warpmpm.materials import newtonian


def dam_break(n_particles: int, n_grid: int, device: str) -> Solver:
    """Water column filling ~1/4 of the domain floor area, dam-break style: the
    particle count is hit exactly by sampling a box at the density that yields
    it (claymore uses MODEL_PPC ~ 8; we land within one particle per row)."""
    grid = GridConfig(n_grid=n_grid, grid_lim=1.0)
    lim = grid.grid_lim
    pad = 3 * grid.dx
    # column footprint: half the width, a quarter of the length, most of the height
    ext = np.array([0.5 * lim - 2 * pad, 0.25 * lim - pad, 0.8 * lim - 2 * pad])
    h = float((ext.prod() / n_particles) ** (1.0 / 3.0))
    ax = [np.arange(pad + h / 2, pad + e, h) for e in ext]
    pos = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    s = Solver(grid=grid, device=device)
    s.load_particles(pos, np.full(len(pos), h ** 3, np.float32))
    s.set_material(newtonian(eta=1.0e-3, density=1000.0, bulk_modulus=2.0e5),
                   g=[0.0, 0.0, -9.81])
    s.add_plane((0, 0, pad), (0, 0, 1), "separate", friction=0.0)
    s.add_domain_walls()
    return s


def bench(n_particles: int, n_grid: int, substeps: int, dt: float,
          device: str) -> dict:
    s = dam_break(n_particles, n_grid, device)
    n_warm = min(50, substeps // 4)
    s.step(dt, n_warm)                       # JIT + graph capture outside the timing
    if device.startswith("cuda"):
        wp.synchronize_device(device)
    t0 = time.perf_counter()
    s.step(dt, substeps)
    if device.startswith("cuda"):
        wp.synchronize_device(device)
    wall = time.perf_counter() - t0
    per_sub = wall / substeps
    return {
        "n_particles": s.n_particles, "n_grid": n_grid, "substeps": substeps,
        "dt": dt, "ms_per_substep": 1e3 * per_sub,
        "mpups": s.n_particles / per_sub / 1e6,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-particles", type=int, default=1_000_000,
                    help="match claymore's logged particle count")
    ap.add_argument("--n-grid", type=int, default=256,
                    help="match claymore's grid cells per axis (domain/dx)")
    ap.add_argument("--substeps", type=int, default=200,
                    help="timed substeps; match the claymore segment you timed")
    ap.add_argument("--dt", type=float, default=2.0e-5,
                    help="substep dt (throughput is dt-insensitive; keep it stable)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    r = bench(args.n_particles, args.n_grid, args.substeps, args.dt, args.device)
    print(f"n_particles={r['n_particles']}  grid={r['n_grid']}^3  "
          f"substeps={r['substeps']}  dt={r['dt']:.1e}")
    print(f"{r['ms_per_substep']:.3f} ms/substep  ->  {r['mpups']:.1f} "
          f"million particle-updates/s")
