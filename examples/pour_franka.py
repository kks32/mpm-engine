"""Franka pours MPM honey from one glass into another, the pouring counterpart of the
viscoplastic dough-press experiments on the same engine.

The robot action and the scene geometry are ported from a companion Genesis (SPH)
pouring study: the same Panda joint trajectory (the two Panda models agree in forward
kinematics), the same glass profile, poses, and 80% fill, so the same pour runs in
both simulators and the results compare directly. The liquid is honey on warpmpm's
weakly compressible generalized-Newtonian fluid (eta = 10 Pa.s, rho = 1420 kg/m^3).
That viscosity sits far above the grid's numerical-viscosity floor, so eta is a
meaningful physical parameter, and the stream never splashes, which avoids the
impact-aeration artifact water-like pours show on this solver (splash-deposited beds
that the J-blind EOS freezes about 35% loose). The remaining volume excess is a loose
crown about one cell thick, measured first order in dx (1.060 at 128^3, 1.036 at
192^3), hence the 192^3 default. Both glasses are kinematic revolved-SDF colliders
with Newton-exact wrench accumulators: the held glass reads the wrist-load analog,
and the receiving glass acts as a scale, its Fz growth giving the transferred weight,
cross-checked against the particle count each frame.

Leak control (grid BC contact band plus sticky core) is audited every frame: any
particle embedded in a glass wall is counted, projected back out, and reported; the
run fails loudly if the audit ever grows past a fraction of the fill.

Run:
  python examples/pour_franka.py                    # device auto-resolves (cuda:0 if present)
  python examples/pour_franka.py --fast             # coarse smoke run (96^3)
  python examples/pour_franka.py --skip-video       # metrics only
  python examples/pour_franka.py --fast --record    # sim now, render after in a fresh
                                                    # GL-only subprocess. Use this on
                                                    # GH200/Vista nodes, where GL and heavy
                                                    # CUDA sharing one process fault the
                                                    # driver (dmesg Xid 31/109); each alone
                                                    # is fine.
  python examples/pour_franka.py --render-only --n-grid 96   # re-render a recording

Outputs (out/pour_franka/):
  pour_franka.mp4    composite MuJoCo render: arm + glasses + honey (60 fps)
  metrics.csv        per-frame region counts/fractions, tilt, wrenches, leak audit
  pour_metrics.png   receiver/spill fractions + glass wrenches vs time
  final_*.npz        end-state particles for volume/bed-density analysis
  settled_*.npz      cached settled particle state (delete or --rebake to refresh)
  _rec_n*/           --record frame dumps (positions, speeds, glass pose per frame)
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from warpmpm import GlassProfile, GridConfig, Solver, cup_fill, newtonian
from warpmpm.adapters.mujoco_adapter import PandaPour
from warpmpm.colliders.glass import (
    angular_velocity_between,
    cavity_mask,
    glass_sdf_local,
    project_out_of_solid,
    quat_to_mat,
    world_to_local,
    write_glass_obj,
)

OUT = Path(__file__).resolve().parents[1] / "out" / "pour_franka"

# ---- scene constants, world frame (= the Genesis study's world; the Panda base at (-0.15, 0, 0)) --
PROFILE = GlassProfile()                                   # the Genesis study's glass
# Receiver rests on the floor (z = height/2). the Genesis study used (0.29, -0.20); ours is shifted
# to the MEASURED stream centroid at rim height (the MPM stream lands 5 cm further +x
# than the SPH one; at the old spot it hit the far rim and the splash crown ejected over
# it). Swept-source-glass clearance at this spot is 50 mm (was 19 mm).
RECEIVER_POS = np.array([0.34, -0.19, 0.12])
Q_ID = np.array([1.0, 0.0, 0.0, 0.0])
FILL_FRACTION = 0.80
# Real honey viscosity/density; bulk_modulus stays the artificial 9e5 (true K ~ 2 GPa
# would cost ~50x the acoustic substeps; at pour speeds Ma < 0.1 the compressibility
# error is < 1%). eta = 10 Pa.s sits far above the grid's numerical-viscosity floor
# and suppresses the impact splash that aerates water-like pours on this solver.
HONEY = dict(eta=10.0, density=1420.0, bulk_modulus=9.0e5)
GLASS_FRICTION = 0.05                                      # the Genesis study's coup_friction
FPS = 60
HOLD_SECONDS = 1.5    # keep filming after the return so the beds settle on camera
LIQUID_RGBA = np.array([0.93, 0.66, 0.12, 1.0])
GLASS_RGBA = (0.76, 0.92, 1.0, 0.30)                       # the Genesis study's glass_surface
HANDLE_RGBA = np.array([0.06, 0.07, 0.08, 1.0])
BACKDROP_RGBA = np.array([0.24, 0.26, 0.30, 1.0])
# matches the Genesis study's camera side: camera at ~(0.95, -1.35, 0.62) looking at the scene
# (MuJoCo azimuth/elevation give the LOOK direction: from -y+x toward the glasses)
CAMERA = dict(lookat=(0.08, 0.0, 0.30), distance=1.5, azimuth=122.8, elevation=-13.0)

# world -> MPM domain offset: fits the swept glass (x -0.005..0.32, z <= 0.60), the
# receiver, and the floor spill area inside [0, 0.7]^3 with >= 3-cell padding
GRID_LIM = 0.7
WORLD_TO_MPM = np.array([0.16, 0.53, 0.02])                # floor z=0 -> 3.7 cells at 128


def sound_speed(liq: dict) -> float:
    return float(np.sqrt(1.1 * liq["bulk_modulus"] / liq["density"]))


def substeps_per_tick(liq: dict, dx: float, dt_tick: float) -> int:
    """Acoustic CFL (0.28) plus the explicit-viscosity stability bound: the viscous
    stress 2*eta*dev(D) is integrated explicitly, which needs dt <= rho*dx^2/(6*eta)
    (3-D diffusion limit). Water never feels the second term; it overtakes the
    acoustic bound around eta ~ 100 Pa.s at 128^3."""
    acoustic = sound_speed(liq) / (0.28 * dx)
    viscous = 6.0 * liq["eta"] / (liq["density"] * dx * dx)
    return int(np.ceil(dt_tick * max(acoustic, viscous)))


RENDER_MAX = 120_000  # particle cap per rendered frame; the honey is opaque so interior
                      # particles are invisible, and MuJoCo's classic renderer pays a
                      # per-geom draw cost (~6 us/geom) that dominates past ~1e5 spheres.
                      # The subsample is a fixed permutation (same particles every frame,
                      # no flicker) and the radius is bumped by (n/m)^(1/3) to keep the
                      # rendered liquid volume. --render-max 0 restores every particle.


def render_subsample(n: int, cap: int = RENDER_MAX):
    """(indices, radius_scale) for render-time particle thinning; identity if under cap."""
    if not cap or n <= cap:
        return slice(None), 1.0
    idx = np.sort(np.random.default_rng(0).permutation(n)[:cap])
    return idx, float((n / cap) ** (1.0 / 3.0))


def make_arm(mesh_path: Path) -> PandaPour:
    arm = PandaPour(height=720, width=1280, max_geom=360000,  # every particle at 192^3
                    glass_mesh=mesh_path, glass_rgba=GLASS_RGBA,
                    sphere_detail=(8, 6))  # particles are a few px; 28x16 default is 2x slower
    arm.set_glass_pose("glass_rcv", RECEIVER_POS, Q_ID)  # static receiver, set once
    arm.cam.lookat[:] = CAMERA["lookat"]
    arm.cam.distance = CAMERA["distance"]
    arm.cam.azimuth = CAMERA["azimuth"]
    arm.cam.elevation = CAMERA["elevation"]
    return arm


def render_frame(arm: PandaPour, x_world, spd, p_now, q_now, t_now: float, h: float,
                 radius_scale: float = 1.0):
    """One composite frame: arm at t, source glass at (p, q), particles colored by speed."""
    arm.set_glass_pose("glass_src", p_now, q_now)
    arm.set_time(t_now)
    col = np.tile(LIQUID_RGBA, (len(x_world), 1)).astype(np.float32)
    col[:, :3] = np.clip(
        col[:, :3] + 0.35 * np.clip(spd / 2.0, 0, 1)[:, None], 0, 1
    )  # brighten fast liquid
    handle_c = p_now + quat_to_mat(q_now) @ arm.GRASP_LOCAL
    return arm.render_with_particles(
        x_world, col, radius=0.85 * h * radius_scale,
        table=(0.19, -0.18, 0.0, 0.85),
        boxes=[
            (handle_c, (0.060, 0.025, 0.050), HANDLE_RGBA, quat_to_mat(q_now)),
            ((0.10, 0.50, 0.70), (1.60, 0.02, 0.75), BACKDROP_RGBA),  # backdrop
        ],
    )


def write_mp4(frames_dir: Path, fps: int) -> Path:
    import imageio.v2 as imageio

    mp4 = OUT / "pour_franka.mp4"
    with imageio.get_writer(mp4, fps=fps, codec="libx264",
                            quality=8, macro_block_size=2,
                            # moov atom up front: streamable/previewable in IDEs
                            output_params=["-movflags", "+faststart"]) as wtr:
        for p in sorted(frames_dir.glob("f_*.png")):
            wtr.append_data(imageio.imread(p))
    return mp4


def _render_chunk(task) -> int:
    """Worker: render recorded frames [lo, hi) to PNGs. Top-level so it spawns cleanly;
    each worker owns its own PandaPour (own GL context)."""
    n_grid, lo, hi, render_max = task
    import imageio.v2 as imageio

    rec_dir = OUT / f"_rec_n{n_grid}"
    files = sorted(rec_dir.glob("f_*.npz"))[lo:hi]
    h = float(np.load(rec_dir / "meta.npz")["h"])
    arm = make_arm(OUT / "glass_render.obj")  # written by the parent before spawning
    frames_dir = OUT / "_frames"
    sub, rscale = None, 1.0
    t0 = time.time()
    for j, fp in enumerate(files):
        d = np.load(fp)
        x, spd = d["x"], d["spd"].astype(np.float32)
        if sub is None:
            sub, rscale = render_subsample(len(x), render_max)
        img = render_frame(arm, x[sub], spd[sub], d["p"], d["q"], float(d["t"]), h,
                           radius_scale=rscale)
        imageio.imwrite(frames_dir / f"f_{lo + j:04d}.png", img)
        if lo == 0 and (j % 30 == 0 or j == len(files) - 1):
            print(f"render {j+1:3d}/{len(files)} (worker 0) "
                  f"[{(time.time()-t0)/(j+1)*1000:.0f}ms/frame]")
    arm.close()
    return len(files)


def render_recording(n_grid: int, workers: int = 0, render_max: int = RENDER_MAX) -> Path:
    """Render a --record dump to mp4 in a process with no warp compute (safe on nodes
    where GL + heavy CUDA in one process fault the driver). Frames are independent, so
    they are split across `workers` GL-only subprocesses (0 = min(8, cpu count))."""
    rec_dir = OUT / f"_rec_n{n_grid}"
    n = len(list(rec_dir.glob("f_*.npz")))
    if not n:
        raise SystemExit(f"no recorded frames in {rec_dir}; run with --record first")
    meta = np.load(rec_dir / "meta.npz")
    write_glass_obj(PROFILE, OUT / "glass_render.obj")  # once, before workers race
    frames_dir = OUT / "_frames"
    frames_dir.mkdir(exist_ok=True)
    for stale in frames_dir.glob("f_*.png"):
        stale.unlink()

    import os
    workers = min(workers or min(8, os.cpu_count() or 1), n)
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        bounds = np.linspace(0, n, workers + 1).astype(int)
        tasks = [(n_grid, int(a), int(b), render_max)
                 for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        with ProcessPoolExecutor(max_workers=len(tasks),
                                 mp_context=mp.get_context("spawn")) as ex:
            done = sum(ex.map(_render_chunk, tasks))
    else:
        done = _render_chunk((n_grid, 0, n, render_max))
    print(f"rendered {done} frames with {workers} worker(s) "
          f"[{(time.time()-t0)/max(done,1)*1000:.0f}ms/frame effective]")
    mp4 = write_mp4(frames_dir, fps=int(meta["fps"]))
    print("wrote", mp4)
    return mp4


def build_scene(device: str, n_grid: int, arm: PandaPour, sparse: bool = False,
                fused: bool = True, sort_interval: int = 0):
    grid = GridConfig(n_grid=n_grid, grid_lim=GRID_LIM)
    h = grid.dx / 2
    cup_pos0, cup_quat0 = arm.cup_pose_at(0.0)
    pos_local, vol = cup_fill(PROFILE, h, fill_fraction=FILL_FRACTION)
    pos = (cup_pos0 + pos_local @ quat_to_mat(cup_quat0).T + WORLD_TO_MPM).astype(np.float32)

    s = Solver(grid=grid, device=device, sparse=sparse, fused=fused,
               sort_interval=sort_interval).load_particles(pos, vol)
    s.set_material(newtonian(**HONEY))
    s.add_plane((0, 0, WORLD_TO_MPM[2]), (0, 0, 1), "separate", friction=0.3)
    s.add_domain_walls()
    src = s.add_cup(PROFILE, cup_pos0 + WORLD_TO_MPM, tuple(cup_quat0),
                    friction=GLASS_FRICTION)
    rcv = s.add_cup(PROFILE, RECEIVER_POS + WORLD_TO_MPM, tuple(Q_ID),
                    friction=GLASS_FRICTION)
    return s, grid, src, rcv, vol


def settle(s: Solver, dt: float, substeps: int, seconds: float = 0.4) -> None:
    for _ in range(round(seconds * FPS)):
        s.step(dt, substeps)


def level_volume(x_world, pos, quat, h: float) -> float:
    """Apparent liquid volume in a glass from its fill level (robust 97th-pct depth,
    fillet-aware cavity volume). Compared with the count-implied volume, the ratio
    exposes packing voids the J-based EOS cannot see (sum(V0*J) itself conserves to
    <0.1%). At water-like viscosity this exposed impact aeration (splash-deposited
    beds frozen ~1.3x loose), which is why this example pours honey: the viscous
    stream never splashes, and the residual excess is a ~1-cell loose crown that
    converges first order in dx (level asymptote 1.058 at 128^3, 1.008 at 192^3;
    interior particles-per-cell on final_n*.npz is the exact bed-density instrument).
    A DFSPH solver (the Genesis twin of this scene) reads 1.0 by construction.
    """
    from warpmpm.colliders.glass import world_to_local

    m = cavity_mask(x_world + WORLD_TO_MPM, np.asarray(pos) + WORLD_TO_MPM, quat,
                    PROFILE, pad=0.75 * h)
    if int(m.sum()) < 50:
        return 0.0
    zl = world_to_local(x_world[m], pos, quat)[:, 2]
    depth = float(np.quantile(zl, 0.97) - PROFILE.inner_floor_z) + 0.5 * h
    return PROFILE.cavity_volume(depth)


def glass_audit(x_mpm, pos_world, quat, h: float):
    """Fused per-glass audit: ONE world_to_local transform yields the solid (leak)
    count, the cavity mask, and the apparent level volume. Replaces four separate
    transform sweeps (solid_mask + cavity_mask + level_volume's two) that dominated
    the per-frame host time at 340k particles. Semantics identical to the originals."""
    local = world_to_local(x_mpm, np.asarray(pos_world) + WORLD_TO_MPM, quat)
    n_solid = int((glass_sdf_local(local, PROFILE) < 0.0).sum())
    r_xy = np.hypot(local[:, 0], local[:, 1])
    z = local[:, 2]
    cav = ((r_xy < PROFILE.inner_radius_at_z(z) + 0.75 * h)
           & (z >= PROFILE.inner_floor_z - 1e-3)
           & (z <= PROFILE.rim_z))
    if int(cav.sum()) < 50:
        vol = 0.0
    else:
        depth = float(np.quantile(z[cav], 0.97) - PROFILE.inner_floor_z) + 0.5 * h
        vol = PROFILE.cavity_volume(depth)
    return n_solid, cav, vol


def run(device: str = "auto", n_grid: int = 192, video: bool = True,
        record: bool = False, rebake: bool = False, render_stride: int = 1,
        frames: int | None = None, render_workers: int = 0,
        render_max: int = RENDER_MAX, sparse: bool = False,
        fused: bool = True, sort_interval: int = 0,
        profile: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    arm = make_arm(write_glass_obj(PROFILE, OUT / "glass_render.obj"))

    s, grid, src, rcv, vol = build_scene(device, n_grid, arm, sparse=sparse,
                                         fused=fused, sort_interval=sort_interval)
    s.profile = profile
    h = grid.dx / 2
    n0 = s.n_particles
    m_liq = float(HONEY["density"] * vol.sum())
    dt_tick = 1.0 / FPS
    substeps = substeps_per_tick(HONEY, grid.dx, dt_tick)
    dt = dt_tick / substeps
    n_frames = round((arm.duration + HOLD_SECONDS) * FPS)  # pose clamps home after duration
    if frames is not None:
        n_frames = min(n_frames, frames)
    print(f"n_grid={n_grid} dx={grid.dx*1000:.2f}mm N={n0} honey={m_liq:.2f}kg "
          f"dt={dt:.2e} ({substeps} substeps/frame, CFL={sound_speed(HONEY)*dt/grid.dx:.2f}) "
          f"{n_frames} frames")

    # ---- settle (cached) --------------------------------------------------------------
    cache = OUT / f"settled_n{n_grid}_f{int(100*FILL_FRACTION)}.npz"
    cup_pos0, cup_quat0 = arm.cup_pose_at(0.0)
    if cache.exists() and not rebake:
        d = np.load(cache)
        if len(d["x"]) == n0:
            s.set_x(d["x"])
            s.set_v(d["v"])
            print(f"loaded settled state from {cache.name}")
        else:
            rebake = True
    if rebake or not cache.exists():
        t0 = time.time()
        settle(s, dt, substeps)
        x, v, npx = project_out_of_solid(s.x(), s.v(), cup_pos0 + WORLD_TO_MPM, cup_quat0,
                                         PROFILE, clearance=0.35 * h)
        s.set_x(x)
        s.set_v(v)
        np.savez_compressed(cache, x=s.x(), v=s.v())
        print(f"settled {time.time()-t0:.0f}s (projected {npx}); cached -> {cache.name}")

    n_src0 = int(cavity_mask(s.x(), cup_pos0 + WORLD_TO_MPM, cup_quat0, PROFILE,
                             pad=0.75 * h).sum())

    # ---- pour -------------------------------------------------------------------------
    frames_dir = OUT / "_frames"
    if video:
        frames_dir.mkdir(exist_ok=True)
        for stale in frames_dir.glob("f_*.png"):  # a rerun must not keep old tail frames
            stale.unlink()
        import imageio.v2 as imageio
    rec_dir = OUT / f"_rec_n{n_grid}"
    if record:
        rec_dir.mkdir(exist_ok=True)
        for stale in rec_dir.glob("f_*.npz"):
            stale.unlink()
        np.savez(rec_dir / "meta.npz", h=h, fps=FPS // render_stride, n_grid=n_grid)
    rows = []
    max_embedded = 0
    proj_total = 0
    render_sub, render_rscale = render_subsample(n0, render_max)
    t_sim = t_host = t_io = 0.0
    t_start = time.time()
    for frame in range(n_frames):
        t = frame * dt_tick
        t_a = time.time()
        # start-of-tick pose + per-tick velocities: modify_bc sweeps the cup to the
        # commanded end-of-tick pose over the substeps (the set_box contract + rotation)
        p0, q0 = arm.cup_pose_at(t)
        p1, q1 = arm.cup_pose_at(t + dt_tick)
        vel = (p1 - p0) / dt_tick
        omega = angular_velocity_between(q0, q1, dt_tick)
        s.set_cup(src, center=p0 + WORLD_TO_MPM, quat=q0, velocity=vel, omega=omega)
        s.reset_cup_wrench(src)
        s.reset_cup_wrench(rcv)
        s.step(dt, substeps)
        w_src = s.cup_wrench(src, dt_tick)
        w_rcv = s.cup_wrench(rcv, dt_tick)
        t_now = t + dt_tick
        p_now, q_now = p1, q1
        t_b = time.time()

        # ---- leak audit + rescue net (counts reported; grid BC should keep this ~0) ----
        x = s.x()
        v = s.v()
        ns_src, cav_src, vol_src = glass_audit(x, p_now, q_now, h)
        ns_rcv, cav_rcv, vol_rcv = glass_audit(x, RECEIVER_POS, Q_ID, h)
        max_embedded = max(max_embedded, ns_src + ns_rcv)
        if ns_src or ns_rcv:
            x, v, n1 = project_out_of_solid(x, v, p_now + WORLD_TO_MPM, q_now, PROFILE,
                                            clearance=0.35 * h,
                                            solid_velocity=(vel, omega))
            x, v, n2 = project_out_of_solid(x, v, RECEIVER_POS + WORLD_TO_MPM, Q_ID,
                                            PROFILE, clearance=0.35 * h)
            proj_total += n1 + n2
            s.set_x(x)
            s.set_v(v)
            # rare path: re-account on the projected positions (matches the old code,
            # which masked after the rescue moved embedded particles out of the walls)
            _, cav_src, vol_src = glass_audit(x, p_now, q_now, h)
            _, cav_rcv, vol_rcv = glass_audit(x, RECEIVER_POS, Q_ID, h)

        in_src = int(cav_src.sum())
        in_rcv = int(cav_rcv.sum())
        tilt = arm.tilt_degrees(q_now)
        rows.append(dict(
            frame=frame, t=round(t_now, 5), tilt_deg=round(tilt, 2), n_src=in_src,
            n_rcv=in_rcv, n_air_spill=n0 - in_src - in_rcv,
            frac_rcv=round(in_rcv / max(n_src0, 1), 5),
            frac_spill=round((n0 - in_src - in_rcv) / max(n_src0, 1), 5),
            src_fx=round(w_src["force"][0], 4), src_fy=round(w_src["force"][1], 4),
            src_fz=round(w_src["force"][2], 4), src_ty=round(w_src["torque"][1], 5),
            rcv_fz=round(w_rcv["force"][2], 4),
            level_vol_src_L=round(vol_src * 1e3, 5), level_vol_rcv_L=round(vol_rcv * 1e3, 5),
            embedded=ns_src + ns_rcv, projected=proj_total,
        ))

        t_c = time.time()
        if frame % render_stride == 0:
            if video:
                img = render_frame(arm, x[render_sub] - WORLD_TO_MPM,
                                   np.linalg.norm(v[render_sub], axis=1),
                                   p_now, q_now, t_now, h, radius_scale=render_rscale)
                imageio.imwrite(frames_dir / f"f_{frame//render_stride:04d}.png", img)
            elif record:
                # positions f32 (f16 would quantize to ~0.3 mm over the 0.7 m domain),
                # speeds f16 (color only): ~0.6 MB/frame at 40k particles
                np.savez(rec_dir / f"f_{frame//render_stride:04d}.npz",
                         x=(x - WORLD_TO_MPM).astype(np.float32),
                         spd=np.linalg.norm(v, axis=1).astype(np.float16),
                         p=p_now, q=q_now, t=t_now)
        t_sim += t_b - t_a
        t_host += t_c - t_b
        t_io += time.time() - t_c

        if frame % 30 == 0 or frame == n_frames - 1:
            el = time.time() - t_start
            print(f"frame {frame+1:3d}/{n_frames} t={t_now:5.2f}s tilt={tilt:5.1f} "
                  f"src={in_src:6d} rcv={in_rcv:6d} spill={n0-in_src-in_rcv:5d} "
                  f"emb={ns_src+ns_rcv} rcvFz={w_rcv['force'][2]:7.2f}N "
                  f"[{el/(frame+1)*1000:.0f}ms/frame]")

    # ---- outputs ---------------------------------------------------------------------
    csv_path = OUT / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    r = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax = axes[0]
    ax.plot(r["t"], r["frac_rcv"], label="receiver fraction", color="tab:blue")
    ax.plot(r["t"], r["frac_spill"], label="air+spill fraction", color="tab:red")
    ax.plot(r["t"], r["tilt_deg"] / 100, "--", color="grey", label="tilt/100 (deg)")
    v_ref_l = 1e3 * float(vol.sum())
    ax.plot(r["t"], (r["level_vol_src_L"] + r["level_vol_rcv_L"]) / v_ref_l, ":",
            color="tab:purple",
            label="apparent (level) volume / true volume  [aeration diagnostic]")
    ax.set_ylabel("fraction of initial fill")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(r["t"], -r["src_fz"], label="held glass -Fz (load)", color="tab:green")
    ax.plot(r["t"], -r["rcv_fz"], label="receiver -Fz (scale)", color="tab:blue")
    ax.axhline(9.81 * m_liq, ls=":", color="k", label=f"total weight {9.81*m_liq:.1f}N")
    # the scale cross-check: weight implied by the particle count in the receiver
    ax.plot(r["t"], 9.81 * m_liq * r["n_rcv"] / n0, "--", color="tab:cyan",
            label="receiver count * m*g/N (check)")
    ax.set_ylabel("force (N)")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(r["t"], r["embedded"], label="embedded in glass (per-frame audit)",
            color="tab:red")
    ax.plot(r["t"], r["projected"], label="cumulative projected out", color="tab:orange")
    ax.set_xlabel("time (s)"); ax.set_ylabel("particles")
    ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"Franka pour, MPM honey (N={n0}, dx={grid.dx*1000:.1f}mm)")
    fig.tight_layout()
    fig.savefig(OUT / "pour_metrics.png", dpi=130)
    plt.close(fig)

    mp4 = None
    if video:
        mp4 = write_mp4(frames_dir, fps=FPS // render_stride)
        print("wrote", mp4)
    # end-state particles for post-hoc volume analysis: the level metric over-reads on
    # viscous mounds, so bed density must be read from particles-per-cell, not levels
    final_npz = OUT / f"final_n{n_grid}.npz"
    np.savez_compressed(final_npz, x=s.x(), v=s.v(), vol=vol, rho0=HONEY["density"])
    print("wrote", csv_path, "and", OUT / "pour_metrics.png", "and", final_npz.name)
    print(f"final: receiver {r['frac_rcv'][-1]*100:.1f}% spill/air {r['frac_spill'][-1]*100:.1f}% "
          f"| max embedded ever {max_embedded} | total projected {proj_total}")
    print(f"timing: sim+wrench {t_sim:.0f}s | audit+metrics (host numpy) {t_host:.0f}s | "
          f"render/record io {t_io:.0f}s | per frame "
          f"{t_sim/n_frames*1000:.0f}/{t_host/n_frames*1000:.0f}/{t_io/n_frames*1000:.0f} ms")
    if profile:
        print(s.profile_report())
        print(f"  untimed (host python, launches, box/pose updates): "
              f"{t_sim - sum(sum(v) for v in s._sim.time_profile.values()) / 1000.0:.1f}s "
              f"of the sim block")
    arm.close()

    if record:
        n_rec = len(list(rec_dir.glob("f_*.npz")))
        print(f"recorded {n_rec} frames -> {rec_dir}")
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--render-only", "--n-grid", str(n_grid),
               "--render-workers", str(render_workers), "--render-max", str(render_max)]
        print("rendering in fresh GL-only processes (GL + CUDA compute must not share "
              "a process on GH200 nodes):", " ".join(cmd))
        if subprocess.run(cmd).returncode == 0:
            mp4 = OUT / "pour_franka.mp4"
        else:
            print("post-render failed; the recording is kept, rerun:", " ".join(cmd))
    return {"rows": rows, "mp4": mp4, "max_embedded": max_embedded, "n0": n0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto",
                    help="Warp device: auto (cuda if available), cuda:N, or cpu")
    ap.add_argument("--n-grid", type=int, default=192)
    ap.add_argument("--fast", action="store_true", help="coarse smoke run (n_grid 96)")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--record", action="store_true",
                    help="dump render state during the sim, then render it in a fresh "
                         "GL-only subprocess (use on GH200/Vista: inline GL faults)")
    ap.add_argument("--render-only", action="store_true",
                    help="render an existing --record dump to mp4 (no simulation)")
    ap.add_argument("--rebake", action="store_true", help="refresh the settled cache")
    ap.add_argument("--frames", type=int, default=None,
                    help="cap the number of frames (smoke/debug)")
    ap.add_argument("--render-workers", type=int, default=0,
                    help="parallel GL render processes for --record/--render-only "
                         "(0 = min(8, cpus); frames are independent)")
    ap.add_argument("--render-max", type=int, default=RENDER_MAX,
                    help=f"max particles drawn per frame (0 = all; default {RENDER_MAX})")
    ap.add_argument("--sparse", action="store_true",
                    help="active-block sparse grid compute (disables CUDA graph capture; "
                         "A/B it against the default and WARPMPM_NO_CUDA_GRAPH=1)")
    ap.add_argument("--fused", action="store_true",
                    help="(default; kept for compatibility) claymore-fused particle pass")
    ap.add_argument("--no-fused", action="store_true",
                    help="restore the three-pass pipeline (A/B against the fused default)")
    ap.add_argument("--sort-interval", type=int, default=0,
                    help="re-sort particles by grid block every K frames (0 = off; "
                         "GPU locality; changes particle index identity at sort ticks)")
    ap.add_argument("--profile", action="store_true",
                    help="per-phase substep timing table (forces live launches + a device "
                         "sync per phase, so the run is slower; the shares are the signal)")
    args = ap.parse_args()
    if args.render_only:
        render_recording(96 if args.fast else args.n_grid,
                         workers=args.render_workers, render_max=args.render_max)
    else:
        run(device=args.device, n_grid=96 if args.fast else args.n_grid,
            video=not args.skip_video and not args.record, record=args.record,
            rebake=args.rebake, frames=args.frames,
            render_workers=args.render_workers, render_max=args.render_max,
            sparse=args.sparse, fused=not args.no_fused,
            sort_interval=args.sort_interval, profile=args.profile)
