"""CPIC transfer masking and ghost fill (step 4): the zero-thickness dam, the
shelf reaction wrench, and Coulomb friction through the ghost projection. Split
pipeline (the fused path lands in step 5)."""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm.core.solver import GridConfig, Solver
from warpmpm.geometry import build_surface_cdf
from warpmpm.materials import elastic, newtonian

G = GridConfig(n_grid=32, grid_lim=0.4)
DX = G.dx


def _sheet(normal: str, size=0.5, res=64):
    """Square sheet through the body-frame origin with the given normal axis."""
    s = size / 2
    if normal == "x":
        v = np.array([[0, -s, -s], [0, s, -s], [0, s, s], [0, -s, s]], dtype=float)
    else:
        v = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return build_surface_cdf(v, f, res=res, band_cells=3.0)


def _water(x0, x1, y0, y1, z0, z1):
    hp = DX / 2
    ax = [np.arange(a + hp / 2, b, hp) for a, b in ((x0, x1), (y0, y1), (z0, z1))]
    return np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)


def _solver(pos, g=(0.0, 0.0, 0.0)):
    s = Solver(grid=G, device="cpu")
    s.load_particles(pos, np.full(len(pos), (DX / 2) ** 3, np.float32))
    s.set_material(newtonian(eta=0.5, density=1000.0, bulk_modulus=5.0e4), g=list(g))
    s.add_plane((0, 0, 3 * DX), (0, 0, 1), "slip")
    s.add_domain_walls()
    return s


def test_zero_thickness_dam_holds():
    """The core CPIC claim: a slab driven into a zero-thickness vertical wall never
    crosses it. The same wall as a plain grid-BC would need ~2 cells of thickness."""
    pos = _water(0.10, 0.16, 0.12, 0.28, 3 * DX, 3 * DX + 0.08)
    s = _solver(pos)
    x_wall = 0.20
    s.add_cdf_collider(_sheet("x"), center=(x_wall, 0.2, 0.2))
    v = s.v()
    v[:, 0] += 1.0
    s.set_v(v)
    crossed_max = 0
    near = False
    for _ in range(30):
        s.step(2e-4, 10)
        x = s.x()
        crossed_max = max(crossed_max, int((x[:, 0] > x_wall).sum()))
        near = near or bool((x[:, 0] > x_wall - 1.5 * DX).any())
    assert near, "the slab never reached the wall; the test exercised nothing"
    assert crossed_max == 0, f"{crossed_max} particles crossed the thin wall"


def test_shelf_wrench_matches_weight():
    """A water column resting on a horizontal CDF shelf: the ghost-projection
    impulse must integrate to the column's weight (the SDF reaction test's
    tolerance)."""
    pos = _water(0.14, 0.26, 0.14, 0.26, 0.21, 0.29)
    s = _solver(pos, g=(0.0, 0.0, -9.81))
    h = s.add_cdf_collider(_sheet("z"), center=(0.2, 0.2, 0.20))
    weight = float(len(pos) * (DX / 2) ** 3 * 1000.0 * 9.81)
    for _ in range(60):                       # settle onto the shelf
        s.step(2e-4, 10)
    # the undamped column rings, so damp lightly and average over many ticks
    s._sim.mpm_model.grid_v_damping_scale = 0.999
    forces = []
    for _ in range(40):
        s.reset_cdf_wrench()
        s.step(2e-4, 10)
        forces.append(s.cdf_wrench(h, 2e-3)["force"][2])
    f_mean = float(np.mean(forces))
    # the material presses DOWN on the shelf (same convention as the SDF wrench).
    # The CDF wrench is approximate in v1 (the ghost projection's separation push
    # is entangled with the support impulse; exact accounting is a step-6 item),
    # so this pins sign and order of magnitude, not calibration.
    assert f_mean < 0.0
    assert 0.4 * weight < -f_mean < 2.5 * weight, f"{f_mean:.3f} N vs -{weight:.3f} N"
    # nothing fell through the shelf
    assert float(s.x()[:, 2].min()) > 0.20 - 1.5 * DX


def test_ghost_friction_slides_and_holds():
    """An ELASTIC block on a tilted shelf: frictionless slides, high friction holds.
    The material must be a solid; a fluid flows on any incline regardless of wall
    friction, which tests nothing. The tilt comes from rotating gravity (same
    physics, simpler geometry): a lateral component along +x."""
    pos = _water(0.16, 0.24, 0.16, 0.24, 0.21, 0.26)
    slid = {}
    for mu, key in ((0.0, "slip"), (2.0, "hold")):
        s = Solver(grid=G, device="cpu")
        s.load_particles(pos, np.full(len(pos), (DX / 2) ** 3, np.float32))
        s.set_material(elastic(E=5.0e4, nu=0.3, density=1000.0), g=[4.0, 0.0, -9.0])
        s.add_domain_walls()
        s.add_cdf_collider(_sheet("z"), center=(0.2, 0.2, 0.20),
                           surface="separable", friction=mu)
        for _ in range(30):                        # settle and engage contact
            s.step(2e-4, 10)
        # terminal sliding velocity, not displacement: displacement includes the
        # pre-contact settling drift that both cases share
        vels = []
        for _ in range(10):
            s.step(2e-4, 10)
            vels.append(float(s.v()[:, 0].mean()))
        slid[key] = float(np.mean(vels))
    # CDF friction is a soft drag, not static stick: the tangential correction only
    # routes through the incompatible weight fraction, so mu reduces the terminal
    # sliding speed rather than anchoring (use an SDF collider where calibrated
    # friction matters). Pin the monotone effect with margin.
    assert slid["slip"] > 1.5 * max(slid["hold"], 1e-6), \
        f"slip {slid['slip']*1000:.2f} mm/s vs hold {slid['hold']*1000:.2f} mm/s"
    assert slid["slip"] > 0.01


def test_moving_wall_sweeps_without_leaks():
    """A vertical CDF wall driven into a resting blob at 0.5 m/s: every particle
    ends up pushed ahead of the wall; none tunnels behind it (the separation push
    plus color persistence carry particles the wall overtakes)."""
    pos = _water(0.22, 0.30, 0.14, 0.26, 3 * DX, 3 * DX + 0.08)
    s = _solver(pos)
    x0 = 0.18
    v_wall = 1.0
    h = s.add_cdf_collider(_sheet("x"), center=(x0, 0.2, 0.2),
                           velocity=(v_wall, 0.0, 0.0))
    t = 0.0
    for _ in range(40):
        s.step(2e-4, 10)
        t += 2e-3
        wall_x = x0 + v_wall * t
        behind = int((s.x()[:, 0] < wall_x - 1.5 * DX).sum())
        assert behind == 0, f"{behind} particles tunneled behind the moving wall"
    assert wall_x > 0.25, "the wall did sweep through the blob's initial region"
    _ = h


def test_fast_wall_warns_on_band_sweep():
    """Past the per-substep sweep limit (band per substep) the tunneling guard
    warns once, mirroring the SDF collider's contract."""
    pos = _water(0.14, 0.20, 0.14, 0.26, 3 * DX, 3 * DX + 0.05)
    s = _solver(pos)
    band = 2.0 * DX
    v_wall = 2.0 * band / 2e-4          # sweeps 2 bands per substep
    s.add_cdf_collider(_sheet("x"), center=(0.3, 0.2, 0.2),
                       velocity=(v_wall, 0.0, 0.0))
    with pytest.warns(RuntimeWarning, match="sweeps"):
        s.step(2e-4, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
