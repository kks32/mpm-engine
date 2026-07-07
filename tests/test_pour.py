"""Cup-collider (pouring) tests, physics-grounded like the engine tests: a static cup
must HOLD water with zero wall penetration and read the water's weight on its
Newton-exact wrench; a tilt below the spill angle must lose nothing; a hard tilt must
pour the water OUT while conserving the particle count and never leaving material
embedded in the glass. All on coarse grids for speed; the true-scale run lives in
examples/pour_franka.py."""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm import GlassProfile, GridConfig, Solver, cup_fill, newtonian
from warpmpm.colliders.glass import (
    angular_velocity_between,
    cavity_mask,
    quat_from_axis_angle,
    solid_mask,
)

PROF = GlassProfile()  # the Dogma95 glass, true scale
Q_ID = (1.0, 0.0, 0.0, 0.0)


def _water_in_cup(n_grid=40, grid_lim=0.35, fill=0.5, device="auto"):
    grid = GridConfig(n_grid=n_grid, grid_lim=grid_lim)
    h = grid.dx / 2
    pos_local, vol = cup_fill(PROF, h, fill_fraction=fill)
    cup_pos = np.array([0.5 * grid_lim, 0.5 * grid_lim, 0.16])
    s = Solver(grid=grid, device=device).load_particles(
        (pos_local + cup_pos).astype(np.float32), vol
    )
    s.set_material(newtonian(eta=1.0e-3, density=1000.0, bulk_modulus=9.0e5))
    s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "separate", friction=0.3)
    s.add_domain_walls()
    return s, cup_pos, vol, h


def test_static_cup_holds_water_and_reads_weight():
    # a settled cup of water: nothing penetrates the glass, everything stays in the
    # cavity, and the accumulated grid impulse reads the water's WEIGHT (Newton check)
    s, cup_pos, vol, h = _water_in_cup()
    cup = s.add_cup(PROF, cup_pos, Q_ID, friction=0.05)
    m_water = float(1000.0 * vol.sum())
    dt = 5.0e-5
    fz = []
    for _ in range(6):
        s.reset_cup_wrench(cup)
        s.step(dt, 500)  # 25 ms per batch
        fz.append(s.cup_wrench(cup, dt * 500)["force"][2])
        assert int(solid_mask(s.x(), cup_pos, Q_ID, PROF).sum()) == 0, "wall penetration"
    x = s.x()
    assert cavity_mask(x, cup_pos, Q_ID, PROF, pad=0.75 * h).all(), "water left the cup"
    f_settled = float(np.mean(fz[-3:]))
    assert abs(f_settled + 9.81 * m_water) < 0.1 * 9.81 * m_water, (
        f"cup wrench {f_settled:.2f} N != -weight {-9.81 * m_water:.2f} N"
    )


def test_static_cup_hydrostatic_pressure():
    s, cup_pos, _, _ = _water_in_cup()
    s.add_cup(PROF, cup_pos, Q_ID, friction=0.05)
    s.step(5.0e-5, 2500)
    z = s.x()[:, 2]
    c = s.cauchy()
    p = -(c[:, 0, 0] + c[:, 1, 1] + c[:, 2, 2]) / 3.0
    lo = z < np.quantile(z, 0.25)
    hi = z > np.quantile(z, 0.75)
    assert p[lo].mean() > p[hi].mean() > -50.0, "pressure must grow with depth in the cup"
    p_hydro = 1000.0 * 9.81 * (z.max() - z.min())
    assert 0.2 * p_hydro < p[lo].mean() < 3.0 * p_hydro


def test_tilt_below_spill_angle_holds():
    # freeboard at fill=0.5 allows ~40 deg before the surface meets the rim; a smooth
    # 20 deg tilt must lose nothing and never push particles into the glass wall
    s, cup_pos, _, h = _water_in_cup()
    cup = s.add_cup(PROF, cup_pos, Q_ID, friction=0.05)
    dt, fps = 5.0e-5, 50
    substeps = round(1.0 / fps / dt)
    s.step(dt, 6 * substeps)  # brief settle

    def q_at(t):  # 20 deg over 0.6 s, then hold
        a = np.deg2rad(20.0) * min(t / 0.6, 1.0)
        return quat_from_axis_angle([0, 1, 0], a)

    n_ticks = int(0.9 * fps)
    for tick in range(n_ticks):
        t = tick / fps
        q0, q1 = q_at(t), q_at(t + 1.0 / fps)
        omega = angular_velocity_between(q0, q1, 1.0 / fps)
        s.set_cup(cup, center=cup_pos, quat=q0, velocity=(0, 0, 0), omega=omega)
        s.step(dt, substeps)
        assert int(solid_mask(s.x(), cup_pos, q_at(t + 1.0 / fps), PROF).sum()) == 0
    q_end = q_at(n_ticks / fps)
    assert cavity_mask(s.x(), cup_pos, q_end, PROF, pad=0.75 * h).all(), "spilled below spill angle"


@pytest.mark.slow
def test_hard_tilt_pours_out_and_conserves_mass():
    # tilt to 120 deg: most of the water must LEAVE the cup (it pours), the particle
    # count is exactly conserved, nothing ends embedded in the glass, and everything
    # stays inside the domain (the floor plane + walls catch the spill)
    s, cup_pos, _, h = _water_in_cup(n_grid=48)
    cup = s.add_cup(PROF, cup_pos, Q_ID, friction=0.05)
    n0 = s.n_particles
    dt, fps = 5.0e-5, 50
    substeps = round(1.0 / fps / dt)
    s.step(dt, 6 * substeps)

    def q_at(t):  # 120 deg over 1.2 s, then hold
        a = np.deg2rad(120.0) * min(t / 1.2, 1.0)
        return quat_from_axis_angle([0, 1, 0], a)

    max_embedded = 0
    n_ticks = int(2.0 * fps)
    for tick in range(n_ticks):
        t = tick / fps
        q0, q1 = q_at(t), q_at(t + 1.0 / fps)
        omega = angular_velocity_between(q0, q1, 1.0 / fps)
        s.set_cup(cup, center=cup_pos, quat=q0, velocity=(0, 0, 0), omega=omega)
        s.step(dt, substeps)
        max_embedded = max(
            max_embedded, int(solid_mask(s.x(), cup_pos, q1, PROF).sum())
        )
    x = s.x()
    assert len(x) == n0, "particle count must be exactly conserved"
    assert np.isfinite(x).all()
    in_cup = int(cavity_mask(x, cup_pos, q_at(99.0), PROF, pad=0.75 * h).sum())
    assert in_cup < 0.4 * n0, f"cup should have poured out (still holds {in_cup}/{n0})"
    assert max_embedded <= 0.005 * n0, f"{max_embedded} particles embedded in the glass wall"
    assert int(solid_mask(x, cup_pos, q_at(99.0), PROF).sum()) == 0, "material stuck in the wall"
    assert x[:, 2].min() > 0.0 and x.max() < 0.35, "material escaped the domain"
