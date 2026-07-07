"""Three-prong (3-jaw) gripper on the MuJoCo Franka arm, forming dough.

The Franka carries a three-fingered tool: three box fingertips spaced 120 degrees apart,
mounted at the gripper and closing radially inward. The arm descends onto a dough slab,
the three prongs bite in and close to a core radius, and the slab is formed into a
three-lobed cross-section. The dough is our warp von-Mises engine with the law identified
from one press probe (#75). The Panda's own two-finger gripper is hidden, so the three
prong boxes read as the end-effector.

Same scaffolding as dough_franka_press.py: the MPM tools are driven by WarpMPMBackend, the
arm is posed by inverting its EE kinematics so the gripper tracks the prong cluster (one
shared metric frame), and the composite view is rendered with render_with_particles
(arm + the three prong boxes + dough particles).

Run:  ../.venv/bin/python examples/dough_franka_threeprong.py
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver, block
from warpmpm.coupling.backend import WarpMPMBackend
from warpmpm.materials import vonmises

OUT = Path(__file__).resolve().parents[1] / "out" / "three_prong"
ID_LAW = dict(E=7.70e5, nu=0.30, yield_stress=3045.3)     # identified from the press probe (#75)
ANGLES = np.deg2rad([90.0, 210.0, 330.0])                 # three prongs, 120 deg apart


def lobedness(x0, xf, cx, cy):
    """Strength of the period-120-degree (3-fold) boundary mode, before vs after."""
    def profile(x):
        dx, dy = x[:, 0] - cx, x[:, 1] - cy
        ang = np.arctan2(dy, dx)
        rad = np.sqrt(dx * dx + dy * dy)
        bins = np.linspace(-np.pi, np.pi, 49)
        idx = np.clip(np.digitize(ang, bins) - 1, 0, len(bins) - 2)
        rmax = np.array([rad[idx == b].max() if np.any(idx == b) else np.nan
                         for b in range(len(bins) - 1)])
        return bins[:-1] + np.diff(bins) / 2, rmax
    rbar = np.nanmean(profile(x0)[1]) + 1e-9

    def mode3(x):
        th, r = profile(x)
        ok = np.isfinite(r)
        r = r[ok] - np.nanmean(r)
        t = th[ok]
        return float(np.abs(np.sum(r * np.exp(-3j * t))) / max(ok.sum(), 1) / rbar)
    return mode3(x0), mode3(xf)


class FrankaThreeProng:
    """Three box fingertips mounted on the Franka, closing radially on dough."""

    def __init__(self, n_grid=32, grid_lim=0.30, size=(0.09, 0.09, 0.05),
                 center=(0.15, 0.15, 0.045), ppc=2, seed=0, device="auto"):
        self.grid = GridConfig(n_grid=n_grid, grid_lim=grid_lim)
        self.device = device
        self.pos0, self.vol0, self.floor = block(self.grid, size=size, center=center,
                                                  ppc=ppc, seed=seed)
        self.N = len(self.pos0)
        self.cx, self.cy, self.cz = center
        self.dough_top = self.floor + size[2]
        self.fhalf = (0.013, 0.013, 0.035)                # near-cube fingertip, spans the dough
        self.dt, self.sub = 2.0e-4, 4
        self.dt_ctrl = self.dt * self.sub

    def _prong_centers(self, radius, z):
        return [(self.cx + radius * np.cos(a), self.cy + radius * np.sin(a), z)
                for a in ANGLES]

    def run(self, params=None, R0=0.075, Rf=0.040, vclose=0.10, vdesc=0.18,
            lift=0.10, settle=50, every=3, render=True,
            out_name="dough_franka_threeprong.mp4"):
        p = dict(params or ID_LAW)
        s = Solver(self.grid, device=self.device).load_particles(self.pos0.copy(), self.vol0.copy())
        s.set_material(vonmises(E=p["E"], nu=p["nu"], yield_stress=p["yield_stress"]))
        s.add_plane(point=(0, 0, self.floor), normal=(0, 0, 1), surface="sticky")

        z_work = self.cz                                  # prong cluster centre when forming
        z_high = self.cz + lift                           # cluster starts this far above
        backend = WarpMPMBackend(solver=s)
        c0 = self._prong_centers(R0, z_high)
        H = [backend.attach_tool(c0[k], self.fhalf, velocity=(0, 0, 0)) for k in range(3)]

        n_desc = max(1, int(lift / vdesc / self.dt_ctrl))
        n_close = max(1, int((R0 - Rf) / vclose / self.dt_ctrl))
        radial = [(-vclose * np.cos(a), -vclose * np.sin(a), 0.0) for a in ANGLES]

        # --- arm pose by EE-kinematics inversion (the cluster top maps to a descent frac) --
        arm = a_grid = ee_z = None
        ex0 = ey0 = z_off = 0.0
        if render:
            from warpmpm.adapters.mujoco_adapter import FrankaArm
            arm = FrankaArm(height=620, width=820, hide_gripper=True)
            a_grid = np.linspace(0.0, 1.0, 80)
            ee = np.array([arm.set_descent(float(a), self.dt_ctrl)["pos"] for a in a_grid])
            arm._prev_ee = None
            ee_z = ee[:, 2]
            ex0, ey0 = float(ee[len(ee) // 2, 0]), float(ee[len(ee) // 2, 1])
            # place the cluster (when high) under the EE at descent frac ~0.28
            z_off = float(np.interp(0.28, a_grid, ee_z)) - (z_high + self.fhalf[2])

        def a_of(cluster_top_mpm: float) -> float:
            return float(np.interp(cluster_top_mpm + z_off, ee_z[::-1], a_grid[::-1]))

        def to_world(q):
            out = np.empty_like(q)
            out[:, 0] = ex0 - self.cx + q[:, 0]
            out[:, 1] = ey0 - self.cy + q[:, 1]
            out[:, 2] = q[:, 2] + z_off
            return out

        tmp = None
        if render:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import colormaps
            table_z = self.floor + z_off
            arm.cam.lookat[:] = [ex0, ey0, table_z + 0.03]
            arm.cam.distance = 0.66
            arm.cam.azimuth = 118
            arm.cam.elevation = -38
            tmp = Path(tempfile.mkdtemp())
            PRONG_RGBA = (0.80, 0.82, 0.87, 1.0)

        pos = [list(c) for c in c0]
        x0 = s.x().copy()
        fcount = 0

        def draw(phase: str, grip_mm: float):
            nonlocal fcount
            if not render:
                return
            arm.set_descent(a_of(pos[0][2] + self.fhalf[2]), self.dt_ctrl,
                            track_camera=False)
            x = s.x()
            spd = np.linalg.norm(s.v(), axis=1)
            col = colormaps["YlOrBr_r"](np.clip(spd / 0.35, 0, 1))
            col[:, 3] = 1.0
            boxes = [(to_world(np.array([pos[k]]))[0], self.fhalf, PRONG_RGBA)
                     for k in range(3)]
            rgb = arm.render_with_particles(to_world(x), col, radius=0.0035,
                                            table=(ex0, ey0, table_z, 0.26), boxes=boxes)
            # left: the Franka composite; right: a live top-down view so all three prongs
            # and the forming three-lobed section read clearly
            fig = plt.figure(figsize=(12, 5.6), facecolor="black")
            axl = fig.add_axes([0.0, 0.0, 0.62, 1.0])
            axl.imshow(rgb)
            axl.axis("off")
            axl.text(0.02, 0.97, "three-prong gripper  (von-Mises law, identified)",
                     color="w", fontsize=12, transform=axl.transAxes, va="top")
            axl.text(0.02, 0.91, f"[{phase}]   prong radius = {grip_mm:4.1f} mm",
                     color="#ffd27f", fontsize=12, transform=axl.transAxes, va="top")
            axr = fig.add_axes([0.64, 0.06, 0.34, 0.88])
            axr.set_facecolor("black")
            axr.scatter(x[:, 0], x[:, 1], s=7, c=col, edgecolors="none")
            for k in range(3):
                axr.add_patch(plt.Rectangle((pos[k][0] - self.fhalf[0],
                                             pos[k][1] - self.fhalf[1]),
                                            2 * self.fhalf[0], 2 * self.fhalf[1],
                                            color="#c8cad0"))
            lo, hi = self.cx - 0.085, self.cx + 0.085
            axr.set_xlim(lo, hi)
            axr.set_ylim(lo, hi)
            axr.set_aspect("equal")
            axr.set_xticks([])
            axr.set_yticks([])
            for sp in axr.spines.values():
                sp.set_color("#444")
            axr.set_title("top-down: dough + 3 prongs", color="w", fontsize=11)
            fig.savefig(tmp / f"f_{fcount:04d}.png", dpi=105, facecolor="black")
            plt.close(fig)
            fcount += 1

        # phase A: descend the open cluster onto the dough. Pass the START-of-tick
        # centre + velocity; the fork advances the collider by dt_ctrl*velocity over
        # the step, so pos is advanced AFTER the step (pre-advancing double-moves it).
        for i in range(n_desc):
            for k in range(3):
                backend.set_tool_kinematics(H[k], center=tuple(pos[k]),
                                            velocity=(0, 0, -vdesc))
            backend.step(self.dt, self.sub)
            for k in range(3):
                pos[k][2] -= vdesc * self.dt_ctrl
            if i % every == 0:
                draw("descend", R0 * 1e3)
            print(f"descend {i+1}/{n_desc}  z={pos[0][2]:.4f}", flush=True)

        # phase B: close the three prongs radially inward (start-of-tick centre,
        # advance pos after the step to avoid the double-move)
        for i in range(n_close):
            for k in range(3):
                backend.set_tool_kinematics(H[k], center=tuple(pos[k]), velocity=radial[k])
            backend.step(self.dt, self.sub)
            for k in range(3):
                pos[k][0] += radial[k][0] * self.dt_ctrl
                pos[k][1] += radial[k][1] * self.dt_ctrl
            r_now = np.hypot(pos[0][0] - self.cx, pos[0][1] - self.cy)
            if i % every == 0:
                draw("close", r_now * 1e3)
            print(f"close {i+1}/{n_close}  r={r_now:.4f}", flush=True)

        # phase C: retract the prongs and let the formed dough settle
        for k in range(3):
            pos[k][2] = z_high
        for j in range(settle):
            for k in range(3):
                backend.set_tool_kinematics(H[k], center=tuple(pos[k]), velocity=(0, 0, 0))
            backend.step(self.dt, self.sub)
            if j % every == 0:
                draw("settle", Rf * 1e3)
            print(f"settle {j+1}/{settle}", flush=True)

        xf = s.x().copy()
        m0, mf = lobedness(x0, xf, self.cx, self.cy)
        verdict = "formed a 3-lobed section" if mf > 1.5 * m0 + 1e-3 else "little 3-fold change"
        print(f"\n3-fold boundary amplitude: before={m0:.4f}  after={mf:.4f}  ({verdict})",
              flush=True)

        mp4 = None
        if render:
            OUT.mkdir(parents=True, exist_ok=True)
            mp4 = OUT / out_name
            subprocess.run(["ffmpeg", "-y", "-framerate", "16", "-i", str(tmp / "f_%04d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)],
                           check=True, capture_output=True)
            print(f"wrote {mp4} ({fcount} frames)", flush=True)
            _topview(x0, xf, self.cx, self.cy, self.fhalf, OUT / "three_prong_franka.png")
            print(f"wrote {OUT / 'three_prong_franka.png'}", flush=True)
        return dict(N=self.N, mode3_before=m0, mode3_after=mf, verdict=verdict,
                    mp4=str(mp4) if mp4 else None)


def _topview(x0, xf, cx, cy, fhalf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 5.4))
    lo, hi = cx - 0.075, cx + 0.075
    axa.scatter(x0[:, 0], x0[:, 1], s=11, c="#d9a05b", alpha=0.75, edgecolors="none")
    for a in ANGLES:
        c = (cx + 0.075 * np.cos(a), cy + 0.075 * np.sin(a))
        axa.add_patch(plt.Rectangle((c[0] - fhalf[0], c[1] - fhalf[1]),
                                    2 * fhalf[0], 2 * fhalf[1], color="#3a3a44"))
        axa.annotate("", xy=(cx + 0.045 * np.cos(a), cy + 0.045 * np.sin(a)),
                     xytext=c, arrowprops=dict(arrowstyle="->", color="#3a3a44", lw=1.6))
    axa.set_title("before: square slab + 3 prongs (120$^\\circ$)", fontsize=12)
    axb.scatter(xf[:, 0], xf[:, 1], s=11, c="#c0703a", alpha=0.75, edgecolors="none")
    axb.set_title("after radial close: three-lobed section", fontsize=12)
    for ax in (axa, axb):
        ax.set_aspect("equal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    fig.suptitle("Three-prong Franka gripper forming dough (identified von-Mises law)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(device="auto"):
    sc = FrankaThreeProng(device=device)
    print(f"=== 3-prong Franka gripper on dough (N={sc.N}, law={ID_LAW}) ===", flush=True)
    m = sc.run()
    print("\nmetrics:", {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in m.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    args = parser.parse_args()
    main(device=args.device)
