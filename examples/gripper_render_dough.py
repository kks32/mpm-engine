"""Dough-style PyVista render of the gripper-shaping animation: the particle cloud is surfaced
per frame (density field -> marching cubes -> Taubin smoothing) into a smooth shaded dough solid,
the gripper fingers are drawn as solid boxes, on a ground plane with a soft shadow. Two panels:
the target dough (left) and the achieved dough + gripper (right).

Run:
  python examples/gripper_shape.py plan --device cuda:0
  python examples/gripper_render_dough.py plan --device cuda:0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `examples` importable when run as a script
from examples.gripper_shape import GripperShapeScene, ID_LAW, TRUE, OUT  # noqa: E402


def _poly(pts, h, sigma=1.3, iso_frac=0.30):
    """Density-field marching-cubes surface of a particle cloud -> smoothed PyVista mesh."""
    import pyvista as pv
    from scipy.ndimage import gaussian_filter
    from skimage import measure
    lo = pts.min(0) - 4 * h; hi = pts.max(0) + 4 * h
    dims = np.ceil((hi - lo) / h).astype(int) + 1
    idx = np.clip(np.floor((pts - lo) / h).astype(int), 0, dims - 1)
    fld = np.zeros(dims); np.add.at(fld, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    fld = gaussian_filter(fld, sigma)
    v, f, *_ = measure.marching_cubes(fld, level=iso_frac * fld[fld > 0].mean(), spacing=(h, h, h))
    pd = pv.PolyData(v + lo, np.hstack([np.full((len(f), 1), 3), f]).astype(np.int64).ravel())
    return pd.smooth_taubin(n_iter=18, pass_band=0.05)


def _setup(p, focal, view_radius, zfloor, az=-55.0, elev=24.0):
    import pyvista as pv
    gp = pv.Plane(center=(focal[0], focal[1], zfloor), direction=(0, 0, 1),
                  i_size=8 * view_radius, j_size=8 * view_radius)
    p.add_mesh(gp, color="#eef0f3", ambient=0.55, diffuse=0.55, specular=0.0)
    try:
        p.enable_ssao(radius=0.012, bias=0.001); p.enable_anti_aliasing("ssaa"); p.enable_shadows()
    except Exception:
        pass
    p.set_background("white"); p.enable_parallel_projection()
    a, e = np.deg2rad(az), np.deg2rad(elev)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    p.camera.focal_point = tuple(focal)
    p.camera.position = tuple(np.array(focal) + 4.0 * view_radius * d)
    p.camera.up = (0, 0, 1); p.camera.parallel_scale = view_radius


def _box(center, half):
    import pyvista as pv
    c, h = center, half
    return pv.Box(bounds=(c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1], c[2] - h[2], c[2] + h[2]))


def _plates_and_slab(boxes, dough_pts, floor, clear=0.004, th=0.006):
    """From the two finger colliders + the dough, return (slab_bounds, [plate1, plate2]) where the
    slab is the inter-plate region the dough is clipped to (so no dough sits in front of a plate),
    and each plate is a WALL drawn a few mm OUTSIDE the dough, spanning past it in perp + z."""
    c0, _ = boxes[0]; c1, _ = boxes[1]
    ax = int(np.argmax(np.abs(np.array(c0) - np.array(c1))))     # pinch (close) axis: 0=x or 1=y
    pa = 1 - ax                                                   # in-plane perpendicular axis
    dlo, dhi = dough_pts.min(0), dough_pts.max(0); dctr = 0.5 * (dlo + dhi)
    inner_lo, inner_hi = (dlo[ax], dhi[ax])                       # dough's own extent along the pinch axis
    slab = [-1.0, 3.0, -1.0, 3.0, -1.0, 3.0]
    slab[2 * ax] = inner_lo - 1e-4; slab[2 * ax + 1] = inner_hi + 1e-4
    plates = []
    for sgn, face in [(-1, inner_lo - clear), (+1, inner_hi + clear)]:
        center = [0.0, 0.0, 0.0]; half = [0.0, 0.0, 0.0]
        center[ax] = face + sgn * th; half[ax] = th               # wall just outside the dough
        center[pa] = dctr[pa]; half[pa] = 0.5 * (dhi[pa] - dlo[pa]) + 0.02   # span past dough in perp
        center[2] = 0.5 * (floor + dhi[2]); half[2] = 0.5 * (dhi[2] - floor) + 0.012   # floor->above top
        plates.append((tuple(center), tuple(half)))
    return slab, plates


def _dough_and_fingers(pts, dx, boxes, floor):
    """Surface the dough and return (mesh, drawn_fingers). FULL-WIDTH plates: clip the dough to the
    inter-plate slab + draw wall plates. LOCALIZED fingers (small perp, for carving a T): carve each
    finger box out + draw the actual small finger offset outward (a slab clip would delete the bar)."""
    d = _poly(pts, dx, iso_frac=0.45)
    if not boxes:
        return d, []
    c0, h0 = boxes[0]; c1, h1 = boxes[1]
    ax = int(np.argmax(np.abs(np.array(c0) - np.array(c1)))); pa = 1 - ax
    localized = h0[pa] < 0.045
    if not localized:
        slab, plates = _plates_and_slab(boxes, pts, floor)
        try:
            d = d.clip_box(slab, invert=False).extract_surface()
        except Exception:
            pass
        return d, plates
    # localized: carve each finger out; draw the finger offset outward, spanning floor->top in z
    ztop = float(pts[:, 2].max())
    drawn = []; ctr_ax = 0.5 * (c0[ax] + c1[ax])
    for (c, h) in boxes:
        bnds = (c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1], c[2] - h[2], c[2] + h[2])
        try:
            d = d.clip_box(bnds, invert=True).extract_surface()
        except Exception:
            pass
        sgn = 1.0 if c[ax] > ctr_ax else -1.0
        cc = list(c); cc[ax] += sgn * 0.004; cc[2] = 0.5 * (floor + ztop)
        hh = list(h); hh[2] = 0.5 * (ztop - floor) + 0.012
        drawn.append((tuple(cc), tuple(hh)))
    return d, drawn


def render(frames, target, mp4, dx, floor, fps=12, turntable=48):
    import pyvista as pv
    pv.OFF_SCREEN = True
    DOUGH = "#e3b079"; GREY = "#b9b9c2"; STEEL = "#54545e"
    allpts = np.concatenate([f["x"] for f in frames] + [target], 0)
    lo, hi = allpts.min(0), allpts.max(0); ctr = 0.5 * (lo + hi)
    focal = np.array([ctr[0], ctr[1], lo[2] + 0.42 * (hi[2] - lo[2])])
    vr = 0.52 * float(np.linalg.norm(hi - lo)); zfloor = float(lo[2]) - 0.002
    tgt_mesh = _poly(target, dx)
    tmp = Path(tempfile.mkdtemp()); k = 0
    for fr in frames:
        p = pv.Plotter(off_screen=True, window_size=(1280, 640), shape=(1, 2),
                       lighting="light kit", border=False)
        p.subplot(0, 0)
        p.add_mesh(tgt_mesh, color=GREY, smooth_shading=True, specular=0.2, ambient=0.4)
        _setup(p, focal, vr, zfloor)
        p.add_text("target shape", position="upper_edge", font_size=10, color="#222222")
        p.subplot(0, 1)
        dmesh, drawn = _dough_and_fingers(fr["x"], dx, fr["boxes"], floor)
        p.add_mesh(dmesh, color=DOUGH, smooth_shading=True,
                   specular=0.35, specular_power=18, ambient=0.35, diffuse=0.75)
        for (c, h) in drawn:
            p.add_mesh(_box(c, h), color=STEEL, ambient=0.3, diffuse=0.6, specular=0.5, specular_power=30)
        _setup(p, focal, vr, zfloor)
        p.add_text(fr["label"], position="upper_edge", font_size=10, color="#222222")
        p.screenshot(str(tmp / f"f_{k:04d}.png")); p.close(); k += 1
    # turntable of the final shape (single full-frame panel, camera azimuth sweeps 360 deg)
    final = _poly(frames[-1]["x"], dx, iso_frac=0.45)
    fctr = 0.5 * (frames[-1]["x"].min(0) + frames[-1]["x"].max(0))
    ffocal = np.array([fctr[0], fctr[1], floor + 0.42 * (frames[-1]["x"][:, 2].max() - floor)])
    fvr = 0.62 * float(np.linalg.norm(frames[-1]["x"].max(0) - frames[-1]["x"].min(0)))
    for j in range(turntable):
        az = -55.0 + 360.0 * j / turntable
        p = pv.Plotter(off_screen=True, window_size=(1280, 640), lighting="light kit", border=False)
        p.add_mesh(final, color=DOUGH, smooth_shading=True, specular=0.35, specular_power=18,
                   ambient=0.35, diffuse=0.75)
        _setup(p, ffocal, fvr, zfloor, az=az)
        p.add_text("final shaped dough", position="upper_edge", font_size=11, color="#222222")
        p.screenshot(str(tmp / f"f_{k:04d}.png")); p.close(); k += 1
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp / "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)], check=True, capture_output=True)
    print(f"wrote {mp4} ({k} frames incl. {turntable} turntable)", flush=True)
    return mp4


def main(mode="plan", device="auto"):
    import json
    print(f"=== dough-style gripper-shaping render (mode={mode}) ===", flush=True)
    sc = GripperShapeScene(device=device)
    if mode == "t":
        d = json.load(open(OUT / "t_plan.json")); grips = [tuple(g) for g in d["planned"]]
        target = np.load(OUT / "t_target.npy"); out_mp4 = OUT / "t_shaping_dough.mp4"
    else:
        d = json.load(open(OUT / "plan.json")); grips = [tuple(g) for g in d["planned"]]; ref = [tuple(g) for g in d["ref"]]
        target = sc.simulate(ref, params=TRUE); out_mp4 = OUT / "gripper_shaping_dough.mp4"
    print(f"  replaying {len(grips)} grips, recording frames...", flush=True)
    _, frames = sc.simulate_record(grips, params=ID_LAW, every=10)
    OUT.mkdir(parents=True, exist_ok=True)
    render(frames, target, out_mp4, dx=sc.dx, floor=sc.floor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render smooth paper-style gripper shaping videos.")
    parser.add_argument("mode", nargs="?", default="plan", choices=("plan", "t"))
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    args = parser.parse_args()
    main(mode=args.mode, device=args.device)
