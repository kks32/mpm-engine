"""Arm-driven pour: the Franka holds a cup (a mesh-SDF rigidly attached to the gripper) and
tilts it to pour fluid into a receiving cup, rendered as a single MuJoCo composite (arm +
fluid particles + translucent glasses + table) in the reference aesthetic.

The cup is the same signed-distance-field collider as examples/pour_glass.py, but its pose is
driven by the Franka end-effector each control tick (set_sdf_pose fed the gripper pose), so the
ARM genuinely drives the pour. The wrist (joint q5) sweeps the gripper approach from straight
down to forward, tipping the held cup past the rim. The MPM domain is aligned to the arm by a
fixed world<->mpm shift, so the rendered fluid and arm share one metric frame.

Run:  python -m examples.pour_franka --probe     # fast static alignment frames (no fluid sim)
      python -m examples.pour_franka              # full pour sim + composite mp4
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from warpmpm import GridConfig, Solver                        # noqa: E402
from warpmpm.adapters.mujoco_adapter import FrankaArm          # noqa: E402
from warpmpm.geometry import build_sdf_cached, make_cup_mesh   # noqa: E402
from warpmpm.materials import newtonian                        # noqa: E402

OUT = ROOT / "out" / "pour_franka"
CACHE = ROOT / "out" / "sdf_cache"

CUP = dict(inner_radius=0.044, wall_thickness=0.007, height=0.105, base_thickness=0.010)
# the receiver is a wider, shallower bowl: the wrist-driven tilt sweeps the pour stream across
# several cm, so a wide catch basin reliably collects it (vs a narrow cup that the stream walks off)
RECV = dict(inner_radius=0.075, wall_thickness=0.007, height=0.075, base_thickness=0.010)
# world -> mpm shift: places the arm's pour column near the centre of the grid
SHIFT = np.array([-0.15, 0.35, 0.03])
GRID_LIM = 0.7
# The cup is held by a HANDLE that sticks out its side, mirroring the Genesis pouring glass (the
# Panda grasps an offset handle box, not the glass body, so the fingers never enter the glass).
# The gripper grasp point (TCP) sits at the handle TIP; the cup body is offset to the far side.
TCP_OFFSET = np.array([0.0, 0.0, 0.11])        # grasp point along the gripper approach (fingertips)
HANDLE_LOCAL = np.array([0.10, 0.0, 0.060])    # cup centre -> handle tip, in the cup frame
HANDLE_HALF = np.array([0.030, 0.015, 0.016])  # rendered handle rod half-extents (along cup +x)
HANDLE_MID = np.array([0.072, 0.0, 0.060])     # handle box centre in the cup frame


def w2m(p):
    return np.asarray(p) + SHIFT


def m2w(p):
    return np.asarray(p) - SHIFT


def _mat2quat(R):
    """Rotation matrix -> (x, y, z, w) quaternion."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


class PourArm:
    """Franka posed to hold a cup and tilt it. q5 (wrist flex) sweeps the gripper from
    approach-down (hold) to approach-forward (pour); the cup is rigidly attached so it tips."""

    def __init__(self, hold_frac=0.30, q5_max=1.35, **kw):
        self.arm = FrankaArm(**kw)
        self.q_hold = (1 - hold_frac) * FrankaArm.Q_UP + hold_frac * FrankaArm.Q_DOWN
        self.q5_max = q5_max
        self.R0 = self._pose(0.0)[1]              # hand orientation at the hold (upright cup)

    def _q(self, frac):
        q = self.q_hold.copy()
        q[5] += self.q5_max * frac
        return q

    def _pose(self, frac):
        a = self.arm
        a.data.qpos[:7] = self._q(frac)
        a.mj.mj_forward(a.model, a.data)
        return a.data.xpos[a.ee].copy(), a.data.xmat[a.ee].reshape(3, 3).copy()

    def cup_world(self, frac):
        """World (center, R, quat_xyzw) of the held cup at tilt fraction frac. The gripper grips
        the handle TIP (TCP), and the cup body hangs off the handle: cup_center = grasp_tip -
        cup_R @ HANDLE_LOCAL. The cup starts upright and tips with the hand's relative rotation."""
        pos, R = self._pose(frac)
        grasp = pos + R @ TCP_OFFSET               # gripper TCP at the handle tip
        cup_R = R @ self.R0.T
        cup_center = grasp - cup_R @ HANDLE_LOCAL
        return cup_center, cup_R, _mat2quat(cup_R)

    def handle_world(self, frac):
        """World (center, R) of the rendered handle box for the held cup at fraction frac."""
        cup_center, cup_R, _ = self.cup_world(frac)
        return cup_center + cup_R @ HANDLE_MID, cup_R


def _fill_cavity(center_w, dx, radius, z_lo, z_hi, seed=0):
    h = dx / 2.0
    xs = np.arange(-radius, radius, h); zs = np.arange(z_lo, z_hi, h)
    g = np.stack(np.meshgrid(xs, xs, zs, indexing="ij"), -1).reshape(-1, 3)
    g = g[g[:, 0] ** 2 + g[:, 1] ** 2 < radius**2]
    rng = np.random.default_rng(seed)
    g = g + rng.uniform(-0.25 * h, 0.25 * h, size=g.shape)
    pts = w2m(g + np.asarray(center_w)).astype(np.float32)     # fill in world, store in mpm
    return pts, np.full(len(pts), h**3, dtype=np.float32)


def _axis_to_mat(d):
    """Orthonormal rotation matrix whose local z-axis points along world direction d (for
    orienting a cylinder geom, whose axis is its local z, along an arbitrary direction)."""
    z = np.asarray(d, float); z = z / (np.linalg.norm(z) + 1e-12)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = a - z * (a @ z); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _handle_cylinder(handle_center_w, cup_R, col):
    """The handle rendered as a rod (cylinder) along the cup's +x axis, between the glass body
    and the gripper grasp point."""
    axis = cup_R @ np.array([1.0, 0.0, 0.0])
    return (handle_center_w, HANDLE_HALF[1], HANDLE_HALF[0], _axis_to_mat(axis), col)


def _cup_cylinder(cup_center_w, cup_R, col, dims=CUP):
    """A translucent cylinder geom approximating a glass: centred at half-height up the cup
    axis (cup +z), radius = outer radius, half-height = height/2."""
    axis = cup_R @ np.array([0.0, 0.0, 1.0])
    center = cup_center_w + axis * (0.5 * dims["height"])
    return (center, dims["inner_radius"] + dims["wall_thickness"], 0.5 * dims["height"], cup_R, col)


def _setup_camera(arm, look_w, az=130, elev=-12, dist=1.5):
    arm.cam.lookat[:] = look_w
    arm.cam.azimuth = az; arm.cam.elevation = elev; arm.cam.distance = dist


def run(n_grid=72, eta=3.0, density=1000.0, bulk=1.5e5, hold_frac=0.30, q5_max=-2.0,
        settle_s=0.15, pour_s=1.45, drain_s=0.55, dt=1.2e-4, substeps=6, sdf_res=56,
        render_every=4, target_frac=0.6, probe=False, log=print):
    grid = GridConfig(n_grid=n_grid, grid_lim=GRID_LIM)
    dx = grid.dx
    floor_z_w = 0.0                                   # table top at world z = 0
    floor_z_m = w2m(np.array([0, 0, floor_z_w]))[2]

    cv, cf = make_cup_mesh(n_theta=56, **CUP)
    probe_pt = np.array([CUP["inner_radius"] + 0.5 * CUP["wall_thickness"], 0.0, 0.04])
    # margin_cells=6: the SDF box margin must exceed the contact band (~grid dx, 9.7 mm at
    # n_grid=72); at 4 cells the cup's margin is ~8.9 mm and the collider guard rejects it
    sdf = build_sdf_cached(cv, cf, res=sdf_res, margin_cells=6, interior_probe=probe_pt,
                           cache_dir=CACHE)
    rcv, rcf = make_cup_mesh(n_theta=56, **RECV)
    recv_probe = np.array([RECV["inner_radius"] + 0.5 * RECV["wall_thickness"], 0.0, 0.03])
    recv_sdf = build_sdf_cached(rcv, rcf, res=sdf_res, margin_cells=4, interior_probe=recv_probe,
                                cache_dir=CACHE)

    # the gripper is VISIBLE and grips the cup's handle (offset to the side), so the fingers grip
    # the handle rod, not the glass body -- the Genesis grasp design, no penetration
    pa = PourArm(hold_frac=hold_frac, q5_max=q5_max, height=600, width=800, hide_gripper=False)
    cup0_w, cupR0, cupq0 = pa.cup_world(0.0)
    fluid_pts, fluid_vol = _fill_cavity(cup0_w, dx, radius=CUP["inner_radius"] - 1.3 * dx,
                                        z_lo=CUP["base_thickness"] + dx, z_hi=0.080)
    n_fluid = len(fluid_pts)
    glass = np.array([0.80, 0.90, 1.0, 0.28], np.float32)         # clearer glass
    fluid_rgba = np.array([0.95, 0.55, 0.15, 1.0], np.float32)    # "orange" held-out colour
    handle_rgba = np.array([0.55, 0.40, 0.30, 1.0], np.float32)
    n_settle = int(settle_s / (dt * substeps))
    n_pour = int(pour_s / (dt * substeps))
    n_drain = int(drain_s / (dt * substeps))

    import imageio.v2 as imageio

    def simulate(recv_w, do_render, tmp=None):
        """One full pour with the receiver at recv_w; returns final world positions. Renders
        frames into tmp when do_render (the receiver bowl is drawn at recv_w)."""
        s = Solver(grid=grid, device="cpu").load_particles(fluid_pts, fluid_vol)
        s.set_material(newtonian(eta=eta, density=density, bulk_modulus=bulk))
        s.add_plane((0, 0, floor_z_m), (0, 0, 1), "slip", friction=0.3)
        pour_h = s.add_sdf_collider(sdf, center=w2m(cup0_w), quat=cupq0, surface="separable",
                                    friction=0.4)
        s.add_sdf_collider(recv_sdf, center=w2m(recv_w), surface="separable", friction=0.4)
        frames = []

        def draw(frac):
            cup_w, cupR, _ = pa.cup_world(frac)
            hc, _ = pa.handle_world(frac)
            cyl = [_cup_cylinder(cup_w, cupR, glass),
                   _cup_cylinder(recv_w, np.eye(3), glass, dims=RECV),
                   _handle_cylinder(hc, cupR, handle_rgba)]
            col = np.tile(fluid_rgba, (n_fluid, 1))
            img = pa.arm.render_with_particles(m2w(s.x()), col, radius=0.0068,
                                               table=(cup0_w[0] + 0.05, cup0_w[1], floor_z_w, 0.42),
                                               cylinders=cyl)
            imageio.imwrite(tmp / f"f_{len(frames):04d}.png", img); frames.append(1)

        tick = 0
        for phase, nph in (("settle", n_settle), ("pour", n_pour), ("drain", n_drain)):
            for k in range(nph):
                frac = 0.0 if phase == "settle" else (1.0 if phase == "drain"
                                                      else (k + 1) / n_pour)
                cup_w, _, cupq = pa.cup_world(frac)
                s.set_sdf_pose(pour_h, center=w2m(cup_w), quat=cupq)
                s.step(dt, substeps); tick += 1
                if np.isnan(s.x()).any():
                    log(f"[pour_franka] NaN at tick {tick}; stopping"); break
                if do_render and tick % render_every == 0:
                    draw(frac)
        return m2w(s.x()), frames, s.inverted_count()

    if probe:
        tmp = Path(tempfile.mkdtemp()); OUT.mkdir(parents=True, exist_ok=True)
        for i, frac in enumerate((0.0, 0.5, 1.0)):
            cup_w, cupR, _ = pa.cup_world(frac); hc, _ = pa.handle_world(frac)
            cyl = [_cup_cylinder(cup_w, cupR, glass), _handle_cylinder(hc, cupR, handle_rgba)]
            img = pa.arm.render_with_particles(m2w(fluid_pts), np.tile(fluid_rgba, (n_fluid, 1)),
                                               radius=0.0068,
                                               table=(cup0_w[0] + 0.05, cup0_w[1], floor_z_w, 0.42),
                                               cylinders=cyl)
            imageio.imwrite(OUT / f"probe_{i}.png", img)
        log(f"[pour_franka] wrote 3 probe frames to {OUT}")
        return

    # PASS 1 (no render): park the receiver far away so the fluid falls to the floor, then use
    # the median landing point of the poured fluid as the true catch location (deterministic,
    # vs guessing the sweeping lip). PASS 2 places the bowl there and renders.
    recv_far = np.array([cup0_w[0], cup0_w[1] + 0.4, floor_z_w + 0.001])
    xf1, _, _ = simulate(recv_far, do_render=False)
    landed = xf1[xf1[:, 2] < floor_z_w + 0.06]
    if len(landed) < 30:
        landed = xf1
    recv_w = np.array([float(np.median(landed[:, 0])), float(np.median(landed[:, 1])),
                       floor_z_w + 0.001])
    log(f"[pour_franka] located landing -> receiver at "
        f"({recv_w[0]:.3f},{recv_w[1]:.3f}); {len(landed)}/{n_fluid} on floor")

    cup_top = pa.cup_world(1.0)[0][2] + CUP["height"]
    look = np.array([0.5 * (cup0_w[0] + recv_w[0]), 0.5 * (cup0_w[1] + recv_w[1]), 0.45 * cup_top])
    _setup_camera(pa.arm, look, az=128, elev=-11, dist=1.5)

    tmp = Path(tempfile.mkdtemp()); OUT.mkdir(parents=True, exist_ok=True)
    xf_w, frames, inv = simulate(recv_w, do_render=True, tmp=tmp)
    in_bowl = int(((np.hypot(xf_w[:, 0] - recv_w[0], xf_w[:, 1] - recv_w[1]) < RECV["inner_radius"])
                   & (xf_w[:, 2] < floor_z_w + RECV["height"])).sum())
    log(f"[pour_franka] frames={len(frames)} fluid={n_fluid} inverted={inv} "
        f"caught in receiver bowl: {in_bowl}/{n_fluid} ({100 * in_bowl / n_fluid:.0f}%)")
    mp4 = OUT / "pour_franka.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", "20", "-i", str(tmp / "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)],
                   check=True, capture_output=True)
    log(f"[pour_franka] wrote {mp4}")
    return mp4


if __name__ == "__main__":
    run(probe="--probe" in sys.argv)
