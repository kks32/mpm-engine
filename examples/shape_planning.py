"""Model-based shape planning over the warp MPM engine -- the RoboCraft/RoboCook paradigm
(sampled MPC to a Chamfer/EMD target shape), but the forward model is our PHYSICAL engine whose
constitutive law is meant to be IDENTIFIED from one contact probe rather than hand-set.

Both RoboCraft (Fig. 12) and RoboCook (Table 1, CEM+MPM baseline) use a parameterized MPM as a
forward model; its only liabilities there are (1) the parameters are GUESSED and (2) planning is
slow. Our engine is fast (~0.9 s / rollout, measured) and our inference system supplies the
parameters by convex weak-form identification. This module is piece (2), the planner.

Material: von-Mises plasticine ("metal" in the fork) -- it HOLDS a shape after release (validated:
97% of the imposed compression retained), the dough/plasticine analog of the prior work.

Scope (honest): the box-plate coupling is sticky with no tangential-slip model, so actions are
VERTICAL multi-press only and targets are compression / free-extrusion shapes (flattened patty),
not directed lateral transport. Planning is SAMPLED (CEM) -- the engine is not differentiable, and
the project invariant forbids differentiating the simulator for IDENTIFICATION anyway (here we only
do forward sampled planning; identification stays the convex weak-form solve).

Run the harness self-consistency validation:  ../.venv/bin/python examples/shape_planning.py t0
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from warpmpm import GridConfig, Solver
from warpmpm.materials import vonmises
from warpmpm.scenes import block


# ----------------------------------------------------------------------------- shape losses
def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer (mean of min L2 distances), the RoboCraft/RoboCook metric. O(Na*Nb);
    subsample before calling for large clouds."""
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1) + 1e-18)
    return float(d.min(1).mean() + d.min(0).mean())


def emd(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Earth-Mover (Hungarian) distance; equal counts (subsample to the smaller)."""
    from scipy.optimize import linear_sum_assignment
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1) + 1e-18)
    ri, ci = linear_sum_assignment(cost)
    return float(cost[ri, ci].mean())


# ----------------------------------------------------------------------------- the plate scene
class PlateShapeScene:
    """A von-Mises block on the floor; a flat plate pressed straight down. An action is a vector of
    per-segment vertical plate SPEEDS (m/s). simulate(action, params) rolls the engine forward with
    the given constitutive params and returns the final particle positions -- so the SAME planner
    can run through an identified law, a guessed law, or the true law."""

    def __init__(self, n_grid=32, grid_lim=0.30, size=(0.12, 0.08, 0.06), center=(0.15, 0.15, 0.05),
                 ppc=2, n_seg=2, n_frames=40, sub=4, v_ref=0.10, m_match=200, seed=0,
                 device="auto"):
        self.g = GridConfig(n_grid=n_grid, grid_lim=grid_lim)
        self.device = device
        self.pos0, self.vol0, self.floor = block(self.g, size=size, center=center, ppc=ppc, seed=seed)
        self.N = len(self.pos0)
        self.size = size; self.center = center
        self.dx = self.g.dx; self.dt = 2e-4; self.sub = sub; self.dt_ctrl = self.dt * sub
        self.n_seg = n_seg; self.n_frames = n_frames; self.v_ref = v_ref
        self.seg_len = int(np.ceil(n_frames / n_seg))
        self.ztop = self.floor + size[2]
        self.half = (size[0] / 2 + 0.01, size[1] / 2 + 0.01, 2 * self.dx)
        self.cx, self.cy = center[0], center[1]
        rng = np.random.default_rng(seed)
        self.match_idx = rng.choice(self.N, size=min(m_match, self.N), replace=False)
        self.z0 = float(self.pos0[:, 2].max())

    def _schedule(self, action):
        v = np.repeat(np.asarray(action, float), self.seg_len)[: self.n_frames]
        return v

    def simulate(self, action, params=None, settle=0, release=False):
        """Run a vertical multi-press plan. params: dict to override material (E, nu, yield_stress).
        Returns final particle positions (N,3). settle>0 adds free-settle frames after a release."""
        p = dict(E=5e5, nu=0.3, yield_stress=3e3)
        if params:
            p.update(params)
        s = Solver(self.g, device=self.device).load_particles(self.pos0.copy(), self.vol0.copy())
        s.set_material(vonmises(E=p["E"], nu=p["nu"], yield_stress=p["yield_stress"]))
        s.add_plane(point=(0, 0, self.floor), normal=(0, 0, 1), surface="sticky")
        v = self._schedule(action)
        zc = self.ztop + self.half[2]
        h = s.add_box(center=(self.cx, self.cy, zc), half_size=self.half, velocity=(0, 0, -float(v[0])))
        for f in range(self.n_frames):
            vf = float(v[f])
            zc -= vf * self.dt_ctrl
            s.set_box(h, center=(self.cx, self.cy, zc + vf * self.dt_ctrl), velocity=(0, 0, -vf))
            s.step(self.dt, substeps=self.sub)
        if release:
            s.set_box(h, center=(self.cx, self.cy, 0.25), velocity=(0, 0, 0))
            for _ in range(settle):
                s.step(self.dt, substeps=self.sub)
        return s.x()

    def loss(self, action, target, params=None, **kw):
        x = self.simulate(action, params, **kw)
        return chamfer(x[self.match_idx], target[self.match_idx])


# ----------------------------------------------------------------------------- CEM planner
def cem_plan(scene, target, lo, hi, params=None, pop=20, elite=5, n_iter=5, seed=0,
             init_mean=None, init_std=None, sim_kw=None, verbose=True):
    """Sampled MPC (RoboCook-style CEM): sample plate-speed actions, evaluate each by a full engine
    rollout, refit a Gaussian to the elites. Serial rollouts (the engine is not vmappable), ~0.9 s
    each; pop*n_iter rollouts total."""
    sim_kw = sim_kw or {}
    dim = scene.n_seg
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    mean = (0.5 * (lo + hi)) if init_mean is None else np.asarray(init_mean, float)
    std = (0.35 * (hi - lo)) if init_std is None else np.asarray(init_std, float)
    rng = np.random.default_rng(seed)
    best_a, best_v, hist = mean.copy(), np.inf, []
    for it in range(n_iter):
        samples = np.clip(mean[None] + std[None] * rng.standard_normal((pop, dim)), lo, hi)
        vals = np.array([scene.loss(a, target, params, **sim_kw) for a in samples])
        order = np.argsort(vals)
        el = samples[order[:elite]]
        mean, std = el.mean(0), el.std(0) + 1e-4
        if vals[order[0]] < best_v:
            best_v, best_a = float(vals[order[0]]), samples[order[0]].copy()
        hist.append(best_v)
        if verbose:
            print(f"  [CEM {it}] best={best_v*1000:.3f} mm  mean={np.array2string(mean, precision=4)}", flush=True)
    return best_a, best_v, hist


# ----------------------------------------------------------------------------- T0 validation
def t0(device="auto"):
    """Self-consistency: generate a target by pressing with a KNOWN plate-speed action, then check
    CEM recovers an action reaching that shape. Validates the warp planning harness end to end."""
    print("=== warp shape-planning harness validation (von-Mises) ===", flush=True)
    scene = PlateShapeScene(n_grid=32, ppc=2, n_seg=2, n_frames=40, sub=4, device=device)
    print(f"scene: N={scene.N} particles, dx={scene.dx:.4f}, {scene.n_frames}x{scene.sub} substeps", flush=True)
    action_true = np.array([0.12, 0.05])
    t = time.time()
    x_target = scene.simulate(action_true)
    print(f"target from action_true={action_true} in {time.time()-t:.1f}s; "
          f"block top {scene.z0:.4f} -> {x_target[:,2].max():.4f} (compressed {100*(scene.z0-x_target[:,2].max())/scene.z0:.1f}%)", flush=True)
    t = time.time()
    a, v, hist = cem_plan(scene, x_target, lo=[0.0, 0.0], hi=[0.20, 0.20], pop=20, elite=5, n_iter=5, seed=1)
    dt = time.time() - t
    xr = scene.simulate(a)
    cd = chamfer(xr[scene.match_idx], x_target[scene.match_idx])
    print(f"\nrecovered action = {a}  (true {action_true})", flush=True)
    print(f"final Chamfer = {cd*1000:.3f} mm   [{dt:.0f}s, {20*5} rollouts, {dt/100*1000:.0f} ms/rollout]", flush=True)
    return scene, x_target, a, cd


def ablation(true_yield=3000.0, action_true=(0.30, 0.18), factors=(0.25, 0.5, 1.0, 2.0, 4.0),
             pop=16, elite=4, n_iter=4, out=None, device="auto"):
    """#73 guessed-vs-identified ablation. A target is generated by pressing the TRUE-yield model
    with action_true. For each model whose yield is mis-set by a factor, CEM plans the press that
    reaches the target IN THAT MODEL; the planned action is then EXECUTED in the true model and the
    achieved Chamfer recorded. Plan quality should be V-shaped in the model error, minimized at the
    true yield -- i.e. an MPM with guessed params shapes poorly, with identified params shapes well
    (RoboCraft Fig.12 / RoboCook CEM+MPM, reproduced and fixed)."""
    import json
    from pathlib import Path
    print(f"=== #73 guessed-vs-identified ablation (von-Mises, true_yield={true_yield}) ===", flush=True)
    sc = PlateShapeScene(n_grid=32, ppc=2, n_seg=2, n_frames=80, sub=4, device=device)
    a_true = np.asarray(action_true)
    target = sc.simulate(a_true, params=dict(yield_stress=true_yield))
    comp = 100 * (sc.z0 - target[:, 2].max()) / sc.z0
    print(f"target: action_true={a_true}, compression={comp:.1f}%, N={sc.N}", flush=True)
    lo, hi = [0.0, 0.0], [0.5, 0.5]
    rows = []
    t0_ = time.time()
    for fac in factors:
        my = true_yield * fac
        a_plan, vplan, _ = cem_plan(sc, target, lo, hi, params=dict(yield_stress=my),
                                    pop=pop, elite=elite, n_iter=n_iter, seed=2, verbose=False)
        x_exec = sc.simulate(a_plan, params=dict(yield_stress=true_yield))   # execute in TRUE engine
        cd_exec = chamfer(x_exec[sc.match_idx], target[sc.match_idx])
        rows.append(dict(yield_factor=float(fac), model_yield=float(my),
                         plan_action=[float(z) for z in a_plan],
                         planning_chamfer_mm=float(vplan * 1000), executed_chamfer_mm=float(cd_exec * 1000)))
        tag = " (TRUE/identified target)" if abs(fac - 1.0) < 1e-9 else ""
        print(f"  factor={fac:4.2f} (yield={my:7.0f}): plan a={np.array2string(a_plan, precision=3)} "
              f"in-model CD={vplan*1000:.3f}mm  EXECUTED CD={cd_exec*1000:.3f}mm{tag}", flush=True)
    print(f"[{time.time()-t0_:.0f}s total]", flush=True)
    # the V: executed Chamfer vs model error
    best = min(rows, key=lambda r: r["executed_chamfer_mm"])
    print(f"\nminimum executed Chamfer at factor={best['yield_factor']} "
          f"({best['executed_chamfer_mm']:.3f}mm); worst guess "
          f"{max(r['executed_chamfer_mm'] for r in rows):.3f}mm", flush=True)
    out = Path(out or (Path(__file__).resolve().parents[1] / "out" / "ablation_guessed_vs_identified"))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(true_yield=float(true_yield), action_true=[float(z) for z in a_true],
                   compression_pct=float(comp), rows=rows),
              open(out / "results.json", "w"), indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f = [r["yield_factor"] for r in rows]; ce = [r["executed_chamfer_mm"] for r in rows]
        cp = [r["planning_chamfer_mm"] for r in rows]
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot(f, ce, "o-", color="C3", label="executed (true engine)")
        ax.plot(f, cp, "s--", color="C0", alpha=0.7, label="planning (in wrong model)")
        ax.axvline(1.0, color="k", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("model yield / true yield"); ax.set_ylabel("Chamfer (mm)")
        ax.set_title("Plan quality vs model error\n(identified=1.0 at the bottom; guessed=off the V)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "v_curve.png", dpi=130)
        print(f"figure -> {out/'v_curve.png'}", flush=True)
    except Exception as e:
        print("plot skipped:", e, flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="t0", choices=("t0", "abl"))
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    args = parser.parse_args()
    if args.which == "t0":
        t0(device=args.device)
    elif args.which == "abl":
        ablation(device=args.device)
