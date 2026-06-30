"""SDF collider tests: a watertight mesh as a moving/rotating collision + coupling surface.

Containment: a fluid filled into a cup-shaped SDF must stay above the cup base over time (the
walls block it). Reaction: the wrench read back from the grid impulse must equal the contained
fluid's weight (calibrated, like the box tool). A tilt check confirms the rotating field drags
and spills the fluid (the pour mechanism).
"""
from __future__ import annotations

import functools

import numpy as np

from warpmpm import GridConfig, Solver
from warpmpm.geometry import build_sdf, make_cup_mesh
from warpmpm.materials import newtonian


@functools.lru_cache(maxsize=1)
def _cup_sdf():
    v, f = make_cup_mesh(inner_radius=0.04, wall_thickness=0.008, height=0.09,
                         base_thickness=0.01, n_theta=40)
    return build_sdf(v, f, res=48, margin_cells=4, interior_probe=np.array([0.044, 0.0, 0.04]))


def _fill_cavity(center, dx, radius=0.030, z_lo=0.015, z_hi=0.05):
    h = dx / 2
    pts = []
    for x in np.arange(-radius, radius, h):
        for y in np.arange(-radius, radius, h):
            for z in np.arange(z_lo, z_hi, h):
                if x * x + y * y < radius**2:
                    pts.append((x, y, z))
    pts = np.array(pts) + np.asarray(center)
    vol = np.full(len(pts), h**3, dtype=np.float32)
    return pts.astype(np.float32), vol


def _cup_solver(n_grid=48, cup_base_z=0.06, eta=5.0):
    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    sdf = _cup_sdf()
    center = np.array([0.2, 0.2, cup_base_z])
    pts, vol = _fill_cavity(center, grid.dx)
    s = Solver(grid=grid, device="cpu").load_particles(pts, vol)
    s.set_material(newtonian(eta=eta, density=1000.0, bulk_modulus=5.0e5))
    handle = s.add_sdf_collider(sdf, center=center, surface="separable", friction=0.3)
    mass = float(vol.sum() * 1000.0)
    return s, handle, center, mass


def test_fluid_stays_contained_in_cup():
    s, h, center, _ = _cup_solver()
    z0 = s.x()[:, 2].min()
    dt = 2.0e-4
    for _ in range(30):
        s.reset_sdf_force(h)
        s.step(dt, substeps=5)
    xf = s.x()
    # fluid must not leak through the base or walls (stays above the cup base, no escape)
    assert xf[:, 2].min() > center[2] - 0.01, "fluid leaked below the cup base"
    assert s.inverted_count() == 0
    # and it should not have collapsed to a point / flown apart
    assert abs(xf[:, 2].min() - z0) < 0.03


def test_reaction_equals_contained_weight():
    s, h, _, mass = _cup_solver()
    dt = 2.0e-4
    fz = []
    for it in range(30):
        s.reset_sdf_force(h)
        s.step(dt, substeps=5)
        if it >= 15:  # after a brief settle
            fz.append(s.sdf_wrench(h, dt * 5)["force"][2])
    weight = mass * 9.81
    mean_fz = float(np.mean(fz))
    # the material presses down on the cup with (its weight); magnitude within 25%
    assert abs(abs(mean_fz) - weight) < 0.25 * weight, (
        f"|Fz|={abs(mean_fz):.3f} vs weight {weight:.3f}")


def test_tilt_carries_fluid():
    # a rotating SDF must DRAG the contained fluid with it: tilting the cup about its base
    # (omega about +y, world) rotates the cavity, so fluid above the pivot is carried toward
    # +x. Checking the centroid shift verifies the oriented moving-SDF coupling.
    s, h, _, _ = _cup_solver(eta=2.0)
    dt = 2.0e-4
    for _ in range(10):                  # settle on the cavity floor
        s.step(dt, substeps=5)
    mean_x_before = float(s.x()[:, 0].mean())
    s.set_sdf_pose(h, omega=(0.0, 8.0, 0.0))   # ~37 deg over the next 0.08 s
    for _ in range(80):
        s.step(dt, substeps=5)
    mean_x_after = float(s.x()[:, 0].mean())
    assert mean_x_after - mean_x_before > 0.004, (
        f"rotating cup did not carry the fluid: dx={mean_x_after - mean_x_before:.4f}")
    assert s.inverted_count() == 0
