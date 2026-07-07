"""Press-to-identify, then GRIPPER-to-shape -- the RoboCraft/RoboCook loop driven by our identified
MPM. A 2-finger gripper (two axis-aligned box colliders that close toward a midpoint) pinches the
dough; a sequence of grips at chosen (x, y, axis, closing-width) sculpts it. The forward model is
our warp von-Mises engine instantiated with the law identified from one press probe (#75).

Engine constraint: the fork's box collider is axis-aligned, so a grip pinches along x or y (not an
arbitrary angle). Coupling is sticky/no-friction, so this is the validated-regime approximation of a
gripper (normal compression is faithful; tangential finger drag is an artifact we note).

Run:  ../.venv/bin/python -m examples.gripper_shape demo      # multi-grip sculpt demo
      ../.venv/bin/python -m examples.gripper_shape plan      # CEM-plan grips to a target
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from warpmpm import GridConfig, Solver
from warpmpm.materials import vonmises
from warpmpm.scenes import block

OUT = Path(__file__).resolve().parents[1] / "out" / "gripper_shape"
ID_LAW = dict(E=7.70e5, nu=0.30, yield_stress=3045.3)        # identified from the press probe (#75)
TRUE = dict(E=5e5, nu=0.30, yield_stress=3000.0)


def chamfer(a, b):
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1) + 1e-18)
    return float(d.min(1).mean() + d.min(0).mean())


class GripperShapeScene:
    def __init__(self, n_grid=32, grid_lim=0.30, size=(0.12, 0.08, 0.06), center=(0.15, 0.15, 0.05),
                 ppc=2, m_match=300, seed=0, device="auto"):
        self.g = GridConfig(n_grid=n_grid, grid_lim=grid_lim)
        self.device = device
        self.pos0, self.vol0, self.floor = block(self.g, size=size, center=center, ppc=ppc, seed=seed)
        self.N = len(self.pos0); self.size = size; self.center = center
        self.dx = self.g.dx; self.dt = 2e-4; self.sub = 4; self.dt_ctrl = self.dt * self.sub
        self.cx, self.cy = center[0], center[1]
        self.bw = 0.01; self.fperp = 0.06; self.zc = self.floor + 0.03; self.hz = 0.04
        self.dough_half = (size[0] / 2, size[1] / 2)
        rng = np.random.default_rng(seed)
        self.match_idx = rng.choice(self.N, size=min(m_match, self.N), replace=False)
        self.z0 = float(self.pos0[:, 2].max())
        self.corner = (0.02, 0.02, 0.25)                      # park spot (in-domain, away from dough)

    def _box_centers(self, axis, cx, cy, gap, perp_half=None):
        """Inner-face gap -> the two finger-box centers for an x- or y-pinch. perp_half sets how much
        of the perpendicular direction the fingers span (small = LOCALIZED pinch, for carving)."""
        ph = self.fperp if perp_half is None else perp_half
        off = gap / 2 + self.bw
        if axis == "x":
            return (cx - off, cy, self.zc), (cx + off, cy, self.zc), (self.bw, ph, self.hz)
        return (cx, cy - off, self.zc), (cx, cy + off, self.zc), (ph, self.bw, self.hz)

    def simulate(self, grips, params=None, vclose=0.10, settle=25, gap_open=0.15):
        """Run a sequence of grips. grips: list of (axis, cx, cy, gap_final). Returns final positions."""
        p = dict(params or ID_LAW)
        s = Solver(self.g, device=self.device).load_particles(self.pos0.copy(), self.vol0.copy())
        s.set_material(vonmises(E=p["E"], nu=p["nu"], yield_stress=p["yield_stress"]))
        s.add_plane(point=(0, 0, self.floor), normal=(0, 0, 1), surface="sticky")
        cL, cR, half = self._box_centers("x", self.cx, self.cy, gap_open)
        hL = s.add_box(center=self.corner, half_size=half, velocity=(0, 0, 0))
        hR = s.add_box(center=self.corner, half_size=half, velocity=(0, 0, 0))
        for grip in grips:
            axis, gx, gy, gap_final = grip[:4]; ph = grip[4] if len(grip) > 4 else None
            cLo, cRo, half = self._box_centers(axis, gx, gy, gap_open, ph)
            cLf, cRf, _ = self._box_centers(axis, gx, gy, gap_final, ph)
            s.set_box(hL, center=cLo, velocity=(0, 0, 0)); s.set_box(hR, center=cRo, velocity=(0, 0, 0))
            ax = 0 if axis == "x" else 1
            dist = abs(cLf[ax] - cLo[ax]); nclose = max(1, int(dist / vclose / self.dt_ctrl))
            vL = [0, 0, 0]; vR = [0, 0, 0]
            vL[ax] = +vclose if cLf[ax] > cLo[ax] else -vclose
            vR[ax] = +vclose if cRf[ax] > cRo[ax] else -vclose
            posL = list(cLo); posR = list(cRo)
            for i in range(nclose):
                # start-of-tick centre + velocity; the fork advances by dt_ctrl*velocity
                # over the step, so advance pos AFTER (pre-advancing double-moves it)
                s.set_box(hL, center=tuple(posL), velocity=tuple(vL))
                s.set_box(hR, center=tuple(posR), velocity=tuple(vR))
                s.step(self.dt, substeps=self.sub)
                posL[ax] += vL[ax] * self.dt_ctrl; posR[ax] += vR[ax] * self.dt_ctrl
            # release: park both fingers away (no drag), settle
            s.set_box(hL, center=self.corner, velocity=(0, 0, 0))
            s.set_box(hR, center=(0.28, 0.02, 0.25), velocity=(0, 0, 0))
            for _ in range(settle):
                s.step(self.dt, substeps=self.sub)
        return s.x()

    def simulate_record(self, grips, params=None, vclose=0.10, settle=25, gap_open=0.15, every=8):
        """Like simulate, but record (positions, visible finger boxes) frames for rendering."""
        p = dict(params or ID_LAW)
        s = Solver(self.g, device=self.device).load_particles(self.pos0.copy(), self.vol0.copy())
        s.set_material(vonmises(E=p["E"], nu=p["nu"], yield_stress=p["yield_stress"]))
        s.add_plane(point=(0, 0, self.floor), normal=(0, 0, 1), surface="sticky")
        _, _, half0 = self._box_centers("x", self.cx, self.cy, gap_open)
        hL = s.add_box(center=self.corner, half_size=half0, velocity=(0, 0, 0))
        hR = s.add_box(center=self.corner, half_size=half0, velocity=(0, 0, 0))
        frames = []
        for gi, grip in enumerate(grips):
            axis, gx, gy, gap_final = grip[:4]; ph = grip[4] if len(grip) > 4 else None
            cLo, cRo, half = self._box_centers(axis, gx, gy, gap_open, ph)
            cLf, cRf, _ = self._box_centers(axis, gx, gy, gap_final, ph)
            s.set_box(hL, center=cLo, velocity=(0, 0, 0)); s.set_box(hR, center=cRo, velocity=(0, 0, 0))
            ax = 0 if axis == "x" else 1
            dist = abs(cLf[ax] - cLo[ax]); nclose = max(1, int(dist / vclose / self.dt_ctrl))
            vL = [0, 0, 0]; vR = [0, 0, 0]
            vL[ax] = +vclose if cLf[ax] > cLo[ax] else -vclose
            vR[ax] = +vclose if cRf[ax] > cRo[ax] else -vclose
            posL = list(cLo); posR = list(cRo)
            for i in range(nclose):
                # start-of-tick centre + velocity; the fork advances by dt_ctrl*velocity
                # over the step, so advance pos AFTER (pre-advancing double-moves it)
                s.set_box(hL, center=tuple(posL), velocity=tuple(vL))
                s.set_box(hR, center=tuple(posR), velocity=tuple(vR))
                s.step(self.dt, substeps=self.sub)
                posL[ax] += vL[ax] * self.dt_ctrl; posR[ax] += vR[ax] * self.dt_ctrl
                if i % every == 0:
                    frames.append(dict(x=s.x().copy(), boxes=[(tuple(posL), half), (tuple(posR), half)],
                                       label=f"grip {gi+1}/{len(grips)} ({axis}-pinch)"))
            s.set_box(hL, center=self.corner, velocity=(0, 0, 0)); s.set_box(hR, center=(0.28, 0.02, 0.25), velocity=(0, 0, 0))
            for j in range(settle):
                s.step(self.dt, substeps=self.sub)
                if j % every == 0:
                    frames.append(dict(x=s.x().copy(), boxes=[], label=f"grip {gi+1} release/settle"))
        frames.append(dict(x=s.x().copy(), boxes=[], label="final"))
        return s.x(), frames

    def render_video(self, frames, target, mp4, fps=10):
        import tempfile, subprocess
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        def edges(c, h):
            cx, cy, cz = c; hx, hy, hz = h
            pts = np.array([[sx, sy, sz] for sx in (cx - hx, cx + hx) for sy in (cy - hy, cy + hy) for sz in (cz - hz, cz + hz)])
            E = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
            return [[pts[a], pts[b]] for a, b in E]
        tmp = Path(tempfile.mkdtemp())
        for k, fr in enumerate(frames):
            x = fr["x"]
            fig = plt.figure(figsize=(9, 4.4), facecolor="white")
            axT = fig.add_subplot(1, 2, 1, projection="3d")     # target (static reference)
            axT.scatter(target[:, 0], target[:, 1], target[:, 2], c="0.6", s=4, edgecolors="none")
            axT.set_title("target shape", fontsize=10)
            axA = fig.add_subplot(1, 2, 2, projection="3d")     # achieved + gripper
            axA.scatter(x[:, 0], x[:, 1], x[:, 2], c=x[:, 2], cmap="viridis", s=4, edgecolors="none")
            for (c, h) in fr["boxes"]:
                axA.add_collection3d(Line3DCollection(edges(c, h), colors="crimson", linewidths=1.5))
            axA.set_title(fr["label"], fontsize=10)
            for ax in (axT, axA):
                ax.set_xlim(0.05, 0.25); ax.set_ylim(0.05, 0.25); ax.set_zlim(self.floor, 0.16)
                ax.set_box_aspect((1, 1, 0.7)); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.view_init(elev=28, azim=-60)
            fig.suptitle("Press-identified MPM: 2-finger gripper shaping", fontsize=12)
            fig.tight_layout(); fig.savefig(tmp / f"f_{k:04d}.png", dpi=120, facecolor="white"); plt.close(fig)
        subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp / "f_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)], check=True, capture_output=True)
        print(f"wrote {mp4} ({len(frames)} frames)", flush=True)
        return mp4

    def loss(self, grips, target, params=None):
        x = self.simulate(grips, params)
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 0.29:   # ejected/unstable -> reject
            return 1e3
        return chamfer(x[self.match_idx], target)


def _topdown_png(clouds_labels, path, lim=0.30):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(clouds_labels); fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1: axs = [axs]
    for ax, (cloud, lab) in zip(axs, clouds_labels):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=3, c=cloud[:, 2], cmap="viridis")
        ax.set_title(lab, fontsize=9); ax.set_aspect("equal")
        ax.set_xlim(0.05, 0.25); ax.set_ylim(0.05, 0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); print(f"figure -> {path}", flush=True)


def demo(device="auto"):
    print("=== gripper shaping demo (identified MPM): x-pinch then y-pinch ===", flush=True)
    sc = GripperShapeScene(device=device)
    grips = [("x", 0.15, 0.15, 0.06), ("y", 0.15, 0.15, 0.06)]
    t = time.time()
    x = sc.simulate(grips)
    print(f"  2-grip sculpt in {time.time()-t:.0f}s; final extent "
          f"x={(x[:,0].max()-x[:,0].min())*1000:.0f} y={(x[:,1].max()-x[:,1].min())*1000:.0f} "
          f"z={(x[:,2].max())*1000:.0f}mm", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    _topdown_png([(sc.pos0, "initial block"), (x, "after x-pinch + y-pinch")], OUT / "demo.png")
    return sc, x


def _grips_from_vec(v, axes=("x", "y")):
    """[cx1,cy1,gap1, cx2,cy2,gap2,...] + fixed axes -> grip list."""
    return [(axes[i], float(v[3 * i]), float(v[3 * i + 1]), float(v[3 * i + 2])) for i in range(len(axes))]


def cem_plan_grips(sc, target, params=None, axes=("x", "y"), pop=12, elite=4, n_iter=3, seed=1, verbose=True):
    """CEM over grip parameters (cx,cy,gap per grip; axes fixed) to minimize Chamfer to target."""
    lo = np.array([0.13, 0.13, 0.06] * len(axes)); hi = np.array([0.17, 0.17, 0.10] * len(axes))
    mean = 0.5 * (lo + hi); std = 0.3 * (hi - lo); rng = np.random.default_rng(seed)
    best_v, best = np.inf, mean.copy()
    for it in range(n_iter):
        S = np.clip(mean[None] + std[None] * rng.standard_normal((pop, len(lo))), lo, hi)
        vals = np.array([sc.loss(_grips_from_vec(s, axes), target, params) for s in S])
        idx = np.argsort(vals); el = S[idx[:elite]]; mean, std = el.mean(0), el.std(0) + 1e-3
        if vals[idx[0]] < best_v: best_v, best = float(vals[idx[0]]), S[idx[0]].copy()
        if verbose: print(f"  [CEM {it}] best={best_v*1000:.2f}mm", flush=True)
    return _grips_from_vec(best, axes), best_v


def plan(device="auto"):
    """End-to-end: a target is made by a reference grip sequence (TRUE law); the gripper planner
    reaches it THROUGH THE IDENTIFIED LAW; executed in the TRUE engine. Closes the press->identify->
    gripper-shape loop."""
    print("=== gripper shaping to a target (identified-MPM planner) ===", flush=True)
    sc = GripperShapeScene(device=device)
    ref = [("x", 0.14, 0.15, 0.06), ("y", 0.15, 0.16, 0.07)]      # the target-generating grips (TRUE law)
    target_full = sc.simulate(ref, params=TRUE)
    target = target_full[sc.match_idx]
    print(f"  target made by ref grips {ref}", flush=True)
    t = time.time()
    grips, v = cem_plan_grips(sc, target, params=ID_LAW, pop=12, elite=4, n_iter=3)
    x = sc.simulate(grips, params=TRUE)                            # execute the plan in the TRUE engine
    cd = chamfer(x[sc.match_idx], target) * 1000
    print(f"  planned grips {[(a,round(cx,3),round(cy,3),round(g,3)) for a,cx,cy,g in grips]}", flush=True)
    print(f"  EXECUTED Chamfer to target = {cd:.2f} mm   [{time.time()-t:.0f}s]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    _topdown_png([(sc.pos0, "initial"), (target_full, "target"), (x, f"gripper-shaped ({cd:.1f}mm)")],
                 OUT / "plan.png")
    import json
    json.dump(dict(ref=ref, planned=grips, executed_chamfer_mm=cd), open(OUT / "plan.json", "w"), indent=2)
    return cd


def _t_target(sc, n=320, seed=0):
    """A T-shaped target point cloud in the dough frame: wide bar (high y) + narrow stem (low y)."""
    rng = np.random.default_rng(seed); f = sc.floor; pts = []
    while len(pts) < n:
        x = rng.uniform(0.10, 0.20); y = rng.uniform(0.09, 0.215); z = rng.uniform(f, f + 0.075)
        bar = (0.10 <= x <= 0.20) and (0.175 <= y <= 0.215)
        stem = (0.135 <= x <= 0.165) and (0.09 <= y <= 0.18)
        if bar or stem:
            pts.append((x, y, z))
    return np.array(pts, np.float32)


def tshape(pop=10, elite=3, n_iter=3, seed=1, device="auto"):
    """Plan localized x-pinches at low y to carve a T (stem) while the top stays wide (bar). CEM over
    (cy, gap, perp_half) for 2 grips, seeded at the working prototype."""
    import json
    print("=== T-shape: localized-pinch planning to a T target ===", flush=True)
    sc = GripperShapeScene(device=device)
    target = _t_target(sc); tgt = target[np.random.default_rng(0).choice(len(target), 300)]
    lo = np.array([0.115, 0.045, 0.025] * 2); hi = np.array([0.150, 0.065, 0.040] * 2)
    mean = np.array([0.128, 0.048, 0.030, 0.128, 0.050, 0.030]); std = 0.3 * (hi - lo)
    rng = np.random.default_rng(seed); best_v, best = np.inf, mean.copy()

    def grips_of(v):
        return [("x", 0.15, float(v[0]), float(v[1]), float(v[2])),
                ("x", 0.15, float(v[3]), float(v[4]), float(v[5]))]

    t0 = time.time()
    for it in range(n_iter):
        S = np.clip(mean[None] + std[None] * rng.standard_normal((pop, 6)), lo, hi)
        vals = np.array([sc.loss(grips_of(s), tgt, ID_LAW) for s in S])
        idx = np.argsort(vals); el = S[idx[:elite]]; mean, std = el.mean(0), el.std(0) + 1e-3
        if vals[idx[0]] < best_v: best_v, best = float(vals[idx[0]]), S[idx[0]].copy()
        print(f"  [CEM {it}] best={best_v*1000:.2f}mm", flush=True)
    grips = grips_of(best)
    x = sc.simulate(grips, params=ID_LAW)
    cd = chamfer(x[sc.match_idx], tgt) * 1000
    print(f"  planned T grips {[(round(c,3) if isinstance(c,float) else c) for g in grips for c in g]}", flush=True)
    print(f"  Chamfer to T target = {cd:.2f} mm   [{time.time()-t0:.0f}s]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "t_target.npy", target)
    json.dump(dict(planned=grips, chamfer_mm=cd), open(OUT / "t_plan.json", "w"), indent=2)
    _topdown_png([(sc.pos0, "initial"), (target, "T target"), (x, f"shaped T ({cd:.1f}mm)")], OUT / "t_plan.png")
    return grips


def video(device="auto"):
    """Render the PLANNED grips (from plan.json) as an MP4: gripper fingers close, dough deforms,
    target shown alongside."""
    import json
    print("=== rendering gripper-shaping video (planned grips) ===", flush=True)
    sc = GripperShapeScene(device=device)
    pj = OUT / "plan.json"
    if pj.exists():
        d = json.load(open(pj)); grips = [tuple(g) for g in d["planned"]]; ref = [tuple(g) for g in d["ref"]]
    else:
        grips = [("x", 0.144, 0.149, 0.064), ("y", 0.15, 0.155, 0.076)]; ref = [("x", 0.14, 0.15, 0.06), ("y", 0.15, 0.16, 0.07)]
    target = sc.simulate(ref, params=TRUE)                      # the target shape (TRUE law)
    print(f"  replaying planned grips {grips}", flush=True)
    _, frames = sc.simulate_record(grips, params=ID_LAW, every=6)
    OUT.mkdir(parents=True, exist_ok=True)
    sc.render_video(frames, target, OUT / "gripper_shaping.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="demo", choices=("demo", "plan", "video", "tshape"))
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    args = parser.parse_args()
    if args.which == "demo":
        demo(device=args.device)
    elif args.which == "plan":
        plan(device=args.device)
    elif args.which == "video":
        video(device=args.device)
    elif args.which == "tshape":
        tshape(device=args.device)
