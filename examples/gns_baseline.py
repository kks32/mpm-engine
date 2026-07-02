"""In-house GNS (Graph Network Simulator) baseline -- the RoboCraft/RoboCook/lgbnd learned-dynamics
approach, reproduced in pure torch (their repos need pytorch3d/torch_geometric/taichi, all absent).
Architecture is the standard Sanchez-Gonzalez encode-process-decode: a KNN particle graph, node +
edge MLP encoders, L message-passing steps, a node decoder predicting per-particle displacement.

Trained on K warp von-Mises rollouts; used as the forward model for the SAME CEM shape planner as
our identified MPM. This is the head-to-head for "how do we do vs a GNS in their setting":
  * data efficiency: ours needs ONE force probe; the GNS needs K rollouts to reach a given accuracy.
  * transfer:        ours is a material law (size-independent); the GNS is trained per-instance.

ident/ must never import torch; this lives under mpm_engine/examples (a baseline, not the method).

Run:  ../.venv/bin/python -m examples.gns_baseline gen      # generate warp rollout data
      ../.venv/bin/python -m examples.gns_baseline train    # train + 1-step/rollout accuracy vs K
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from warpmpm import GridConfig, Solver
from warpmpm.materials import vonmises
from warpmpm.scenes import block

OUT = Path(__file__).resolve().parents[1] / "out" / "gns_baseline"
DEV = "cpu"
M = 512            # subsampled particles for the GNS (CPU-tractable)
H = 3              # position-history length (velocity features)
RADIUS = None      # connectivity radius (set from dx)
TRUE = dict(E=5e5, nu=0.30, yield_stress=3000.0)


# --------------------------------------------------------------------------- warp data generation
def _rollout(action, size=(0.12, 0.08, 0.06), n_grid=32, ppc=2, n_frames=60, sub=4,
             params=None, sub_idx=None, seed=0, device="auto"):
    """One warp von-Mises press; record subsampled particle positions per frame + tool center."""
    p = dict(TRUE);
    if params: p.update(params)
    g = GridConfig(n_grid=n_grid, grid_lim=0.30)
    pos, vol0, floor = block(g, size=size, center=(0.15, 0.15, 0.05), ppc=ppc, seed=seed)
    N = len(pos)
    if sub_idx is None:
        sub_idx = np.sort(np.random.default_rng(0).choice(N, size=min(M, N), replace=False))
    s = Solver(g, device=device).load_particles(pos, vol0).set_material(
        vonmises(E=p["E"], nu=p["nu"], yield_stress=p["yield_stress"]))
    s.add_plane(point=(0, 0, floor), normal=(0, 0, 1), surface="sticky")
    dx = g.dx; dt = 2e-4; dt_ctrl = dt * sub
    half = (size[0] / 2 + 0.01, size[1] / 2 + 0.01, 2 * dx); ztop = floor + size[2]; zc = ztop + half[2]
    seg = int(np.ceil(n_frames / len(action)))
    vsched = np.repeat(np.asarray(action), seg)[:n_frames]
    h = s.add_box(center=(0.15, 0.15, zc), half_size=half, velocity=(0, 0, -float(vsched[0])))
    traj = [pos[sub_idx].copy()]; tool = [zc]
    for f in range(n_frames):
        vf = float(vsched[f]); zc -= vf * dt_ctrl
        s.set_box(h, center=(0.15, 0.15, zc + vf * dt_ctrl), velocity=(0, 0, -vf))
        s.step(dt, substeps=sub)
        traj.append(s.x()[sub_idx].copy()); tool.append(zc)
    return np.array(traj, np.float32), np.array(tool, np.float32), sub_idx, dx


def gen(K=40, n_frames=60, out=OUT, device="auto"):
    """Generate K rollouts with random press actions (the GNS training set)."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    sub_idx = None; rolls = []
    t0 = time.time()
    for k in range(K):
        action = rng.uniform(0.05, 0.35, size=2)
        traj, tool, sub_idx, dx = _rollout(action, n_frames=n_frames, sub_idx=sub_idx,
                                           device=device)
        rolls.append(dict(traj=traj, tool=tool, action=action))
        if (k + 1) % 5 == 0:
            print(f"  gen {k+1}/{K}  [{time.time()-t0:.0f}s]", flush=True)
    np.savez(out / "rollouts.npz", sub_idx=sub_idx, dx=dx, n_frames=n_frames,
             trajs=np.array([r["traj"] for r in rolls]),
             tools=np.array([r["tool"] for r in rolls]),
             actions=np.array([r["action"] for r in rolls]))
    print(f"saved {K} rollouts ({M} particles, {n_frames} frames) -> {out/'rollouts.npz'}", flush=True)


# --------------------------------------------------------------------------- GNS model
def _mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class GNS(nn.Module):
    """Encode-process-decode graph network. Node feat: velocity history (H*3) + type(1) + tool-z(1).
    Edge feat: relative pos (3) + dist (1). Predicts per-particle displacement (3)."""
    def __init__(self, n_mp=3, hid=128):
        super().__init__()
        self.node_enc = _mlp(H * 3 + 2, hid, hid)
        self.edge_enc = _mlp(4, hid, hid)
        self.edge_mp = nn.ModuleList([_mlp(3 * hid, hid, hid) for _ in range(n_mp)])
        self.node_mp = nn.ModuleList([_mlp(2 * hid, hid, hid) for _ in range(n_mp)])
        self.dec = _mlp(hid, 3, hid)
        self.n_mp = n_mp

    def forward(self, vel_hist, node_extra, pos, edges, radius):
        # node features: velocity history (H*3) + node_extra (plate-gap, particle-z)
        nf = torch.cat([vel_hist.reshape(vel_hist.shape[0], -1), node_extra], -1)
        n = self.node_enc(nf)
        src, dst = edges
        rel = (pos[src] - pos[dst]) / radius
        ef = self.edge_enc(torch.cat([rel, torch.linalg.norm(rel, dim=-1, keepdim=True)], -1))
        for i in range(self.n_mp):
            e_in = torch.cat([n[src], n[dst], ef], -1)
            ef = ef + self.edge_mp[i](e_in)
            agg = torch.zeros_like(n).index_add_(0, dst, ef)
            n = n + self.node_mp[i](torch.cat([n, agg], -1))
        return self.dec(n)        # per-node predicted displacement


def _knn_edges(pos, radius, max_deg=16):
    """Undirected radius graph (capped degree), pure torch."""
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(1e9)
    within = d < radius
    # cap degree: keep nearest max_deg
    idx = torch.topk(-d, k=min(max_deg, pos.shape[0]), dim=1).indices
    src = torch.arange(pos.shape[0])[:, None].expand_as(idx).reshape(-1)
    dst = idx.reshape(-1)
    keep = within[src, dst]
    return torch.stack([src[keep], dst[keep]])


def _node_extra(pos, tool_z):
    """Per-node features: plate gap (plate_z - particle_z, contact cue) and particle height z."""
    gap = tool_z - pos[:, 2]
    return torch.stack([gap, pos[:, 2]], -1)


def _samples(data, k_use, radius):
    """Build (vel_hist, node_extra, pos, target_disp, edges) tuples; edges precomputed (static)."""
    trajs = data["trajs"][:k_use]; tools = data["tools"][:k_use]
    S = []
    for r in range(len(trajs)):
        T = trajs[r]                                  # (frames+1, M, 3)
        for t in range(H, T.shape[0] - 1):
            pos = torch.tensor(T[t])
            vel = torch.tensor(np.stack([T[t - i] - T[t - i - 1] for i in range(H)], 1))  # (M,H,3)
            tgt = torch.tensor(T[t + 1] - T[t])       # (M,3) displacement
            S.append((vel, _node_extra(pos, torch.tensor(float(tools[r][t]))), pos, tgt,
                      _knn_edges(pos, radius)))
    return S


def train(k_use=40, epochs=40, n_mp=3, hid=128, lr=1e-3, data_path=OUT / "rollouts.npz", log=True):
    data = np.load(data_path); dx = float(data["dx"])
    radius = 2.6 * dx * (5746 / M) ** (1 / 3)         # connectivity scaled to the subsample spacing
    S = _samples(data, k_use, radius)
    tgt_std = torch.stack([s[3] for s in S]).std() + 1e-9
    vel_std = torch.stack([s[0] for s in S]).std() + 1e-9
    model = GNS(n_mp=n_mp, hid=hid).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for ep in range(epochs):
        order = rng.permutation(len(S)); tot = 0.0
        for i in order:
            vel, nx, pos, tgt, edges = S[i]
            pred = model(vel / vel_std, nx, pos, edges, radius)
            loss = ((pred - tgt / tgt_std) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach())
        if log and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [train K={k_use} ep {ep:2d}] loss={tot/len(S):.4f}  [{time.time()-t0:.0f}s]", flush=True)
    return dict(model=model, radius=radius, tgt_std=float(tgt_std), vel_std=float(vel_std), dx=dx)


@torch.no_grad()
def gns_rollout(trained, init_pos, action, tool0_z, n_frames=60, dt_ctrl=8e-4):
    """Roll the GNS forward under a plate action; returns final particle positions (M,3)."""
    m = trained["model"]; radius = trained["radius"]; ts = trained["tgt_std"]; vs = trained["vel_std"]
    pos = torch.tensor(init_pos, dtype=torch.float32)
    hist = [pos.clone() for _ in range(H + 1)]                       # start at rest
    seg = int(np.ceil(n_frames / len(action))); vsched = np.repeat(np.asarray(action), seg)[:n_frames]
    zc = tool0_z
    for f in range(n_frames):
        vf = float(vsched[f]); zc -= vf * dt_ctrl
        vel = torch.stack([hist[-1 - i] - hist[-2 - i] for i in range(H)], 1)   # (M,H,3)
        edges = _knn_edges(pos, radius)
        disp = m(vel / vs, _node_extra(pos, torch.tensor(zc)), pos, edges, radius) * ts
        pos = pos + disp
        hist.append(pos.clone())
    return pos.numpy()

def _test_rollouts(sub_idx, actions, size=(0.12, 0.08, 0.06), params=None, n_frames=60,
                   device="auto"):
    """Warp rollouts for a set of (held-out) actions, recording subsampled final positions."""
    finals = []
    for a in actions:
        traj, tool, _, dx = _rollout(a, size=size, n_frames=n_frames, sub_idx=sub_idx,
                                     params=params, device=device)
        finals.append((traj[-1], traj[0], float(tool[0])))     # (final, init, tool0)
    return finals


def compare(Ks=(2, 5, 10, 20, 40), n_test=4, epochs=40, device="auto"):
    """Data-efficiency head-to-head: GNS prediction error vs #training rollouts K, against the
    one-probe identified MPM (K-independent). Plus a cross-size transfer probe."""
    import json
    data = np.load(OUT / "rollouts.npz"); sub_idx = data["sub_idx"]; dx = float(data["dx"])
    rng = np.random.default_rng(99)
    test_actions = rng.uniform(0.05, 0.35, size=(n_test, 2))
    print(f"=== GNS data-efficiency vs one-probe identified MPM ({n_test} held-out actions) ===", flush=True)
    # ground truth (true law) for the held-out actions
    gt = _test_rollouts(sub_idx, test_actions, device=device)   # list of (final, init, tool0)

    def err(pred_finals):
        return float(np.mean([np.sqrt(((p - g[0]) ** 2).sum(-1)).mean() * 1000 for p, g in zip(pred_finals, gt)]))

    # identified MPM (yield 1.5% err): E_hat=7.70e5, yield_hat=3045 (from #75) -- re-sim the test actions
    id_law = dict(E=7.70e5, nu=0.30, yield_stress=3045.3)
    mpm_finals = [f[0] for f in _test_rollouts(sub_idx, test_actions, params=id_law,
                                               device=device)]
    mpm_err = err(mpm_finals)
    print(f"  identified MPM (ONE force probe): prediction MAE = {mpm_err:.2f} mm  (K-independent)", flush=True)

    rows = [dict(method="identified-MPM (1 probe)", K=1, mae_mm=mpm_err)]
    for K in Ks:
        tr = train(k_use=K, epochs=epochs, log=False)
        gns_finals = [gns_rollout(tr, f[1], a, f[2], n_frames=int(data["n_frames"]))
                      for a, f in zip(test_actions, gt)]
        e = err(gns_finals)
        rows.append(dict(method="GNS", K=int(K), mae_mm=e))
        print(f"  GNS trained on K={K:2d} rollouts: prediction MAE = {e:.2f} mm", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(rows=rows, mpm_err=mpm_err), open(OUT / "data_efficiency.json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        gns = [r for r in rows if r["method"] == "GNS"]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot([r["K"] for r in gns], [r["mae_mm"] for r in gns], "o-", color="C0", label="GNS (learned, K rollouts)")
        ax.axhline(mpm_err, color="C2", lw=2, ls="--", label=f"identified MPM (1 force probe) = {mpm_err:.2f}mm")
        ax.set_xscale("log"); ax.set_xlabel("# training rollouts K (GNS)"); ax.set_ylabel("held-out prediction MAE (mm)")
        ax.set_title("Data efficiency: one force probe vs a learned GNS\n(predicting the shaped dough)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "data_efficiency.png", dpi=140)
        print(f"figure -> {OUT/'data_efficiency.png'}", flush=True)
    except Exception as e:
        print("plot skipped:", e, flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="gen", choices=("gen", "train", "compare"))
    parser.add_argument("K", nargs="?", type=int, default=40)
    parser.add_argument("--device", default="cuda:0", help="Warp MPM device, e.g. cuda:0 or cuda:1")
    args = parser.parse_args()
    if args.which == "gen":
        gen(K=args.K, device=args.device)
    elif args.which == "train":
        train(k_use=args.K)
        print("trained.", flush=True)
    elif args.which == "compare":
        compare(device=args.device)
