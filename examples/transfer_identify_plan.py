"""#74 -- cross-instance transfer: identify the law on ONE object, plan a shaping task on a
DIFFERENT (held-out) object. The convex-identified (G, yield) is a MATERIAL property, so it is
size-independent and transfers for free; a per-object GNN (RoboCraft/RoboCook) is trained on one
instance and does not transfer across object volume. This is the axis where the physical model
structurally wins.

Protocol:
  1. Identify (G, yield) from a force squeeze on a SMALL block A and (separately) a LARGE block B
     -- show the recovered yield matches across sizes (size-independence = the transfer property).
  2. On the LARGE block B, define a target shape (B pressed by a reference action with TRUE params).
  3. Plan the press on B through the A-identified law and through the B-true law (oracle); execute
     both in the B-true engine. Transfer is successful if the A-law plan reaches ~the oracle Chamfer.

Run:  ../.venv/bin/python examples/transfer_identify_plan.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `examples` importable when run as a script
from examples.vonmises_identify import probe, identify  # noqa: E402
from examples.shape_planning import PlateShapeScene, cem_plan, chamfer  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out" / "transfer_identify_plan"
NU = 0.30
TRUE = dict(E=5e5, nu=NU, yield_stress=3000.0)


def _law(idres):
    """(G_hat, yield_hat) -> material params dict for the engine."""
    E = 2.0 * idres["G_hat"] * (1.0 + NU)
    return dict(E=E, nu=NU, yield_stress=idres["yield_hat"])


def run(A=(0.09, 0.06, 0.045), B=(0.12, 0.08, 0.06), a_ref=(0.30, 0.18),
        device="auto"):
    print("=== #74 identify-on-A, plan-on-B transfer (von-Mises) ===", flush=True)
    # 1. identify on both sizes -> the recovered yield should be size-independent
    idA = identify(probe(size=A, n_frames=180, device=device))
    idB = identify(probe(size=B, n_frames=220, device=device))
    print(f"  identify on SMALL A {A}: yield={idA['yield_hat']:.1f} ({100*idA['yield_err']:.1f}%), "
          f"G={idA['G_hat']:.2e}", flush=True)
    print(f"  identify on LARGE B {B}: yield={idB['yield_hat']:.1f} ({100*idB['yield_err']:.1f}%), "
          f"G={idB['G_hat']:.2e}", flush=True)
    print(f"  -> yield A vs B differ by {100*abs(idA['yield_hat']/idB['yield_hat']-1):.1f}% "
          f"(material property, size-independent)", flush=True)

    # 2. target on the LARGE block B, generated with the TRUE law
    sc = PlateShapeScene(n_grid=32, ppc=2, size=B, n_seg=2, n_frames=80, sub=4,
                         device=device)
    a_ref = np.asarray(a_ref)
    target = sc.simulate(a_ref, params=TRUE)
    comp = 100 * (sc.z0 - target[:, 2].max()) / sc.z0
    print(f"  target on B: compression {comp:.1f}%", flush=True)

    # 3. plan B through the A-identified law, the B-identified law, and the true law (oracle)
    res = {}
    for tag, law in [("A-identified (transfer)", _law(idA)),
                     ("B-identified", _law(idB)),
                     ("true (oracle)", TRUE)]:
        a, _, _ = cem_plan(sc, target, [0, 0], [0.5, 0.5], params=law,
                           pop=16, elite=4, n_iter=4, seed=2, verbose=False)
        x = sc.simulate(a, params=TRUE)                       # always execute in the TRUE engine
        cd = chamfer(x[sc.match_idx], target[sc.match_idx]) * 1000
        res[tag] = dict(plan=[float(z) for z in a], executed_chamfer_mm=float(cd))
        print(f"  plan via {tag:24s}: executed Chamfer = {cd:.3f} mm", flush=True)

    transfer_gap = res["A-identified (transfer)"]["executed_chamfer_mm"] / res["true (oracle)"]["executed_chamfer_mm"]
    print(f"\n  TRANSFER GAP (A-law / oracle) = {transfer_gap:.2f}x  "
          f"(identify on a small blob, shape a large one)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(A=list(A), B=list(B), idA={k: idA[k] for k in ("G_hat", "yield_hat", "yield_err")},
                   idB={k: idB[k] for k in ("G_hat", "yield_hat", "yield_err")},
                   results=res, transfer_gap=float(transfer_gap)),
              open(OUT / "results.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Warp device: auto (cuda if available), cuda:N, or cpu")
    args = parser.parse_args()
    run(device=args.device)
