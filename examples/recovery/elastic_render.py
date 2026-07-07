"""Render the elastic gravity-drop bounce: truth-law vs recovered-law, surfaced + strain-coloured.

Reads the npz written by sim.elastic_drop (x, v, F per frame) and renders the deforming blob as
a shaded solid on a ground plane (density-field marching cubes + Taubin smoothing, the same
surfacing as the granular renders), coloured by the instantaneous deformation magnitude
||F - I|| (lights up where the blob is compressed -- the stored elastic energy at impact).

  nclaw_star_bounce.mp4   -- STAR blob: truth-law | recovered-law (learned on the RECTANGLE).
  nclaw_box_train.mp4     -- the RECTANGULAR training drop, single panel.

Run:  .venv/bin/python -m sim.elastic_render            # star (2-panel) + box (1-panel)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.recovery.nclaw_geom_render import _poly, _setup, _view   # reuse surfacing + scene helpers

OUT = ROOT / "out" / "elastic_drop"
VID = ROOT / "out" / "nclaw_compare"
H_SURF, SIGMA, AZ = 0.012, 1.25, -72.0                   # surfacing voxel, smoothing, camera az


def _load(name):
    d = np.load(OUT / name)
    X = d["x"].astype(np.float64)
    F = d["F"].astype(np.float64).reshape(X.shape[0], X.shape[1], 3, 3)
    strain = np.linalg.norm(F - np.eye(3), axis=(2, 3))   # ||F - I|| per particle per frame
    return X, strain, float(d["floor"])


def make_star_video(stride=3, fps=22):
    import pyvista as pv
    pv.OFF_SCREEN = True
    Xt, St, floor = _load("star_truth.npz")
    Xr, Sr, _ = _load("star_pred.npz")
    nf = min(len(Xt), len(Xr)); n = min(Xt.shape[1], Xr.shape[1])
    Xt, St, Xr, Sr = Xt[:nf, :n], St[:nf, :n], Xr[:nf, :n], Sr[:nf, :n]
    rmse = float(np.sqrt(((Xt - Xr) ** 2).sum(-1).mean()) * 1e3)
    clim = [0.0, float(np.percentile(St, 98))]
    focal, view_radius, zfloor = _view(np.concatenate([Xt, Xr]))
    frames = list(range(0, nf, stride))
    pl = pv.Plotter(off_screen=True, window_size=(1280, 720), shape=(1, 2),
                    lighting="light kit", border=False)
    VID.mkdir(parents=True, exist_ok=True)
    out = VID / "nclaw_star_bounce.mp4"
    pl.open_movie(str(out), framerate=fps, quality=8)
    panels = [(0, "truth law  E=2.0e5, nu=0.30", Xt, St),
              (1, "recovered law (learned on RECTANGLE)", Xr, Sr)]

    def frame(fi):
        pl.clear()
        for col, title, X, S in panels:
            pl.subplot(0, col)
            pd = _poly(X[fi], S[fi], H_SURF, SIGMA)
            _setup(pl, focal, view_radius, AZ, zfloor)
            pl.add_mesh(pd, scalars="strain", cmap="turbo", clim=clim, smooth_shading=True,
                        ambient=0.32, diffuse=0.85, specular=0.14,
                        show_scalar_bar=(col == 1),
                        scalar_bar_args=dict(title="||F - I||", n_labels=4, vertical=True,
                                             position_x=0.90, position_y=0.30, height=0.45,
                                             width=0.04, title_font_size=14, label_font_size=12))
            pl.add_text(title, position="upper_edge", font_size=10, color="#222222")
        pl.subplot(0, 0)
        pl.add_text(f"Elastic shape generalization: learn moduli on a rectangular blob, "
                    f"predict a STAR neo-Hookean bounce ({rmse:.1f} mm RMS, no backprop)",
                    position=(0.02, 0.02), viewport=True, font_size=10, color="#111111")

    frame(frames[0])
    for fi in frames:
        frame(fi); pl.write_frame()
    pl.close()
    print(f"wrote {out}  ({len(frames)} frames, star N={n}, {rmse:.1f}mm RMS)")
    return out


def make_box_video(stride=3, fps=22):
    import pyvista as pv
    pv.OFF_SCREEN = True
    X, S, floor = _load("box_truth.npz")
    nf = len(X)
    clim = [0.0, float(np.percentile(S, 98))]
    focal, view_radius, zfloor = _view(X)
    frames = list(range(0, nf, stride))
    pl = pv.Plotter(off_screen=True, window_size=(720, 720), lighting="light kit", border=False)
    VID.mkdir(parents=True, exist_ok=True)
    out = VID / "nclaw_box_train.mp4"
    pl.open_movie(str(out), framerate=fps, quality=8)

    def frame(fi):
        pl.clear()
        pd = _poly(X[fi], S[fi], H_SURF, SIGMA)
        _setup(pl, focal, view_radius, AZ, zfloor)
        pl.add_mesh(pd, scalars="strain", cmap="turbo", clim=clim, smooth_shading=True,
                    ambient=0.32, diffuse=0.85, specular=0.14, show_scalar_bar=True,
                    scalar_bar_args=dict(title="||F - I||", n_labels=4, vertical=True,
                                         position_x=0.88, position_y=0.30, height=0.45,
                                         width=0.04, title_font_size=14, label_font_size=12))
        pl.add_text("Rectangular training drop (moduli recovered from this bounce)",
                    position=(0.02, 0.02), viewport=True, font_size=10, color="#111111")

    frame(frames[0])
    for fi in frames:
        frame(fi); pl.write_frame()
    pl.close()
    print(f"wrote {out}  ({len(frames)} frames, box N={X.shape[1]})")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or "star" in args:
        make_star_video()
    if not args or "box" in args:
        make_box_video()
