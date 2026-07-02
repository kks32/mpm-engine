"""Three-prong (3-jaw) gripper on dough, driven by the identified von-Mises law.

The engine exposes an axis-aligned box collider only, so we model each prong as a small
near-cube fingertip (its footprint is roughly isotropic, so the approach angle does not
need a rotated box). Three fingertips sit at 0, 120, and 240 degrees around the dough
center and close radially inward to a core radius, which forms the slab into a three-lobed
cross-section. The forward model is our warp von-Mises engine with the law identified
from one press probe (#75). A cleaner version would use an oriented capsule or sphere SDF
fingertip; that is a small engine addition noted for later.

Run:  ../.venv/bin/python -m examples.three_prong
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver
from warpmpm.materials import vonmises
from warpmpm.scenes import block

OUT = Path(__file__).resolve().parents[1] / "out" / "three_prong"
ID_LAW = dict(E=7.70e5, nu=0.30, yield_stress=3045.3)     # identified from the press probe (#75)
ANGLES = np.deg2rad([90.0, 210.0, 330.0])                 # three prongs, 120 deg apart


class ThreeProngScene:
    def __init__(self, n_grid=32, grid_lim=0.30, size=(0.09, 0.09, 0.05),
                 center=(0.15, 0.15, 0.045), ppc=2, seed=0, device="auto"):
        self.g = GridConfig(n_grid=n_grid, grid_lim=grid_lim)
        self.device = device
        self.pos0, self.vol0, self.floor = block(self.g, size=size, center=center, ppc=ppc, seed=seed)
        self.N = len(self.pos0)
        self.cx, self.cy, self.cz = center
        self.dt, self.sub = 2e-4, 4
        self.dt_ctrl = self.dt * self.sub
        self.fhalf = (0.013, 0.013, 0.035)                # near-cube footprint, tall enough to span dough
        self.park = [(0.02, 0.02, 0.26), (0.28, 0.02, 0.26), (0.02, 0.28, 0.26)]

    def _prong_centers(self, radius):
        return [(self.cx + radius * np.cos(a), self.cy + radius * np.sin(a), self.cz) for a in ANGLES]

    def run(self, params=None, R0=0.075, Rf=0.040, vclose=0.10, settle=40, every=8):
        p = dict(params or ID_LAW)
        s = Solver(self.g, device=self.device).load_particles(self.pos0.copy(), self.vol0.copy())
        s.set_material(vonmises(E=p["E"], nu=p["nu"], yield_stress=p["yield_stress"]))
        s.add_plane(point=(0, 0, self.floor), normal=(0, 0, 1), surface="sticky")
        c0 = self._prong_centers(R0)
        H = [s.add_box(center=self.park[k], half_size=self.fhalf, velocity=(0, 0, 0)) for k in range(3)]
        for k in range(3):
            s.set_box(H[k], center=c0[k], velocity=(0, 0, 0))

        nclose = max(1, int((R0 - Rf) / vclose / self.dt_ctrl))
        vel = [(-vclose * np.cos(a), -vclose * np.sin(a), 0.0) for a in ANGLES]   # radial inward
        pos = [list(c) for c in c0]
        frames = [dict(x=s.x().copy(), boxes=[(tuple(pos[k]), self.fhalf) for k in range(3)],
                       label="prongs open")]
        for i in range(nclose):
            # pass the START-of-tick centre + velocity; the fork advances the collider
            # by dt_ctrl*velocity over the step. Advance pos AFTER the step, never
            # before, or the box double-moves (2x the commanded close speed).
            for k in range(3):
                s.set_box(H[k], center=tuple(pos[k]), velocity=vel[k])
            s.step(self.dt, substeps=self.sub)
            for k in range(3):
                pos[k][0] += vel[k][0] * self.dt_ctrl
                pos[k][1] += vel[k][1] * self.dt_ctrl
            if i % every == 0:
                frames.append(dict(x=s.x().copy(), boxes=[(tuple(pos[k]), self.fhalf) for k in range(3)],
                                   label=f"closing {i+1}/{nclose}"))
        for k in range(3):
            s.set_box(H[k], center=self.park[k], velocity=(0, 0, 0))
        for j in range(settle):
            s.step(self.dt, substeps=self.sub)
            if j % every == 0:
                frames.append(dict(x=s.x().copy(), boxes=[], label="release / settle"))
        frames.append(dict(x=s.x().copy(), boxes=[], label="final"))
        return s.x(), frames


def lobedness(x0, xf, cx, cy):
    """Crude 3-fold metric: variance of the angular radius profile, before vs after.

    Project to the x-y plane, bin the boundary radius by angle, and report the strength
    of the 3-cycle (a triangular/3-lobed section has a strong period-120-deg component)."""
    def profile(x):
        dx, dy = x[:, 0] - cx, x[:, 1] - cy
        ang = np.arctan2(dy, dx); rad = np.sqrt(dx * dx + dy * dy)
        bins = np.linspace(-np.pi, np.pi, 49)
        idx = np.clip(np.digitize(ang, bins) - 1, 0, len(bins) - 2)
        rmax = np.array([rad[idx == b].max() if np.any(idx == b) else np.nan for b in range(len(bins) - 1)])
        return bins[:-1] + np.diff(bins) / 2, rmax
    th, r0 = profile(x0); _, rf = profile(xf)
    # amplitude of the 3-per-revolution Fourier mode, normalized by the mean radius
    def mode3(th, r):
        ok = np.isfinite(r); r = r[ok] - np.nanmean(r); t = th[ok]
        return float(np.abs(np.sum(r * np.exp(-3j * t))) / max(np.sum(ok), 1) / (np.nanmean(profile(x0)[1]) + 1e-9))
    return mode3(th, r0), mode3(th, rf)


def main(device="auto"):
    OUT.mkdir(parents=True, exist_ok=True)
    sc = ThreeProngScene(device=device)
    print(f"=== 3-prong gripper on dough (N={sc.N} particles, law={ID_LAW}) ===", flush=True)
    xf, frames = sc.run()
    x0 = frames[0]["x"]
    m0, mf = lobedness(x0, xf, sc.cx, sc.cy)
    print(f"3-fold (period-120deg) boundary amplitude: before={m0:.4f}  after={mf:.4f}  "
          f"({'formed a 3-lobed section' if mf > 1.5 * m0 + 1e-3 else 'little 3-fold change'})", flush=True)
    _figure(x0, xf, frames, sc, OUT / "three_prong.png")
    print(f"wrote {OUT/'three_prong.png'} ({len(frames)} frames recorded)", flush=True)


def _figure(x0, xf, frames, sc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 5.2))
    cx, cy = sc.cx, sc.cy
    lo, hi = 0.085, 0.215                                  # zoom on the dough region
    # (a) top view before, with the three prong start positions
    axa = fig.add_subplot(1, 3, 1)
    axa.scatter(x0[:, 0], x0[:, 1], s=10, c="#d9a05b", alpha=0.7, edgecolors="none")
    for c, _ in frames[0]["boxes"]:
        axa.add_patch(plt.Rectangle((c[0] - sc.fhalf[0], c[1] - sc.fhalf[1]),
                                    2 * sc.fhalf[0], 2 * sc.fhalf[1], color="#3a3a44"))
        axa.annotate("", xy=(cx + 0.045 * (c[0]-cx)/0.075, cy + 0.045 * (c[1]-cy)/0.075),
                     xytext=(c[0], c[1]), arrowprops=dict(arrowstyle="->", color="#3a3a44", lw=1.5))
    axa.set_title("(a) before: square slab + 3 prongs (120$^\\circ$)", fontsize=12)
    axa.set_aspect("equal"); axa.set_xlim(lo, hi); axa.set_ylim(lo, hi)
    # (b) top view after: the three-lobed section
    axb = fig.add_subplot(1, 3, 2)
    axb.scatter(xf[:, 0], xf[:, 1], s=10, c="#c0703a", alpha=0.7, edgecolors="none")
    for a in ANGLES:
        axb.annotate("", xy=(cx + 0.030 * np.cos(a), cy + 0.030 * np.sin(a)),
                     xytext=(cx + 0.075 * np.cos(a), cy + 0.075 * np.sin(a)),
                     arrowprops=dict(arrowstyle="->", color="#3a3a44", lw=1.5))
    axb.set_title("(b) after radial close: three-lobed section", fontsize=12)
    axb.set_aspect("equal"); axb.set_xlim(lo, hi); axb.set_ylim(lo, hi)
    # (c) iso of the final shape
    axc = fig.add_subplot(1, 3, 3, projection="3d")
    axc.scatter(xf[:, 0], xf[:, 1], xf[:, 2], s=8, c=xf[:, 2], cmap="copper", alpha=0.7)
    axc.set_title("(c) final shape (3D)", fontsize=12); axc.set_box_aspect((1, 1, 0.6)); axc.view_init(elev=24, azim=-60)
    fig.suptitle("Three-prong gripper on dough (identified von-Mises law)", fontsize=13, y=0.99)
    fig.tight_layout(); fig.savefig(path, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    args = parser.parse_args()
    main(device=args.device)
