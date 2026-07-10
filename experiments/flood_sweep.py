"""Vehicle response over a grid of flood depths and surge velocities.

Uses warpmpm.vehicle.FloodScene as an importable study harness: each (depth, velocity)
pair runs the same splat-captured truck, and the figure overlays downstream
displacement and yaw over time. Final displacement saturating near 0.83 m means the
truck was washed into the downstream wall (the domain ran out, not the surge).

The truck is model scale (1.45 m); read results at full size via Froude scaling
(see the examples/flood_vehicle.py header): depths and displacements x lam,
velocities x sqrt(lam), masses x lam^3, with lam = L_real / 1.45.

Run:  python experiments/flood_sweep.py [--frames 60] [--vehicle PATH] [--up z]

Outputs (out/flood_sweep/): one CSV per case and sweep.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from warpmpm.vehicle import FloodScene, load_vehicle

OUT = Path(__file__).resolve().parents[1] / "out" / "flood_sweep"
DEFAULT_PLY = Path(__file__).resolve().parents[2] / "truck_trimmed.ply"

CASES = [  # (depth m, velocity m/s)
    (0.08, 1.0),
    (0.15, 1.0),
    (0.08, 2.0),
    (0.15, 2.0),
    (0.15, 3.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", default=str(DEFAULT_PLY))
    ap.add_argument("--up", default="z", choices=("x", "y", "z", "-x", "-y", "-z"))
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for depth, vel in CASES:
        v = load_vehicle(args.vehicle, up=args.up)
        scene = FloodScene(v, depth=depth, velocity=vel, n_grid=args.grid,
                           device=args.device)
        hist = scene.run(args.frames)
        tag = f"d{int(depth * 100):02d}_v{vel:.1f}"
        hist.to_csv(OUT / f"{tag}.csv")
        a = hist.arrays()
        d = a["displacement"]
        results.append((depth, vel, a))
        print(f"depth={depth:.2f}m v={vel:.1f}m/s: |d|={np.linalg.norm(d[-1]):.3f}m "
              f"peak_yaw={np.abs(a['yaw_deg']).max():.1f}deg "
              f"leaked={scene.leaked}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
    for depth, vel, a in results:
        lbl = f"{depth * 100:.0f} cm, {vel:.1f} m/s"
        d = np.linalg.norm(a["displacement"], axis=1)
        a0.plot(a["t"], d * 100, label=lbl)
        a1.plot(a["t"], a["yaw_deg"], label=lbl)
    a0.set_xlabel("time (s)"); a0.set_ylabel("|displacement| (cm)")
    a0.grid(alpha=0.3); a0.legend(fontsize=8, title="depth, surge")
    a1.set_xlabel("time (s)"); a1.set_ylabel("yaw (deg)")
    a1.grid(alpha=0.3)
    fig.suptitle("truck response vs flood depth and surge velocity", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "sweep.png", dpi=130)
    print("wrote", OUT / "sweep.png")


if __name__ == "__main__":
    main()
