"""CDF geometry (step 1 of the CPIC plan): side-signed distance to an open oriented
mid-surface, with the rim exclusion that keeps trilerp from fabricating a phantom
surface past open edges."""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm.colliders.glass import GlassProfile, glass_mid_surface_profile
from warpmpm.geometry import (
    CDFData,
    build_surface_cdf,
    build_surface_cdf_cached,
    load_cdf,
    revolve_profile,
    revolve_profile_open,
    save_cdf,
)
from warpmpm.geometry.mesh_sdf import _boundary_loop_samples


def _flat_sheet(size=1.0):
    """Unit square sheet at z = 0, normals +z."""
    v = np.array([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return v, f


def _lookup(cdf: CDFData, pts: np.ndarray):
    """Nearest-voxel lookup; also returns the snapped voxel coordinates, since the
    surface generally sits between voxel planes and expectations must be evaluated
    at the voxel actually probed."""
    idx = np.rint((pts - cdf.origin) / cdf.cell).astype(int)
    idx = np.clip(idx, 0, cdf.res - 1)
    snapped = cdf.origin + idx * cdf.cell
    return (cdf.values[idx[:, 0], idx[:, 1], idx[:, 2]],
            cdf.valid[idx[:, 0], idx[:, 1], idx[:, 2]], snapped)


def test_flat_sheet_signed_distance_and_rim():
    v, f = _flat_sheet()
    cdf = build_surface_cdf(v, f, res=48, band_cells=3.0)
    h = cdf.cell
    # interior probes one/two cells above and below the sheet: d = signed z of the
    # probed voxel (the sheet lies between voxel planes, hence the snap)
    for z in (h, -h, 2 * h, -2 * h):
        pts = np.array([[0.5, 0.5, z], [0.4, 0.6, z]])
        d, ok, snapped = _lookup(cdf, pts)
        assert np.all(ok == 1.0), f"interior at z={z} should be valid"
        assert np.allclose(d, snapped[:, 2], atol=0.15 * h), f"d={d} vs {snapped[:, 2]}"
    # rim tube: within band of the sheet edge (x ~ 0), even at the sheet plane
    d, ok, _ = _lookup(cdf, np.array([[0.0, 0.5, h], [-0.5 * h, 0.5, 0.0]]))
    assert np.all(ok == 0.0), "rim neighborhood must be invalid"
    # far outside the band
    _, ok, _ = _lookup(cdf, np.array([[0.5, 0.5, 10 * h]]))
    assert np.all(ok == 0.0)


def test_open_cylinder_sides_and_rims():
    R = 0.3
    v, f = revolve_profile_open([(R, 0.0), (R, 0.5)], n_theta=64)
    assert len(_boundary_loop_samples(v, f, 0.01)) > 0      # two open rims
    cdf = build_surface_cdf(v, f, res=48, band_cells=3.0)
    h = cdf.cell
    mid_z = 0.25
    inside = np.array([[R - 1.5 * h, 0.0, mid_z], [0.0, R - 1.5 * h, mid_z]])
    outside = np.array([[R + 1.5 * h, 0.0, mid_z], [0.0, R + 1.5 * h, mid_z]])
    d_in, ok_in, s_in = _lookup(cdf, inside)
    d_out, ok_out, s_out = _lookup(cdf, outside)
    assert np.all(ok_in == 1.0) and np.all(ok_out == 1.0)
    # |d| matches the snapped radial distance (n_theta=64 faceting adds ~R(1-cos(pi/64)))
    tol = 0.2 * h + R * (1 - np.cos(np.pi / 64))
    assert np.allclose(np.abs(d_in), np.abs(np.linalg.norm(s_in[:, :2], axis=1) - R),
                       atol=tol)
    assert np.allclose(np.abs(d_out), np.abs(np.linalg.norm(s_out[:, :2], axis=1) - R),
                       atol=tol)
    # opposite sides get opposite signs, consistently around the axis
    assert np.sign(d_in[0]) == np.sign(d_in[1]) != np.sign(d_out[0]) == np.sign(d_out[1])
    # rim tubes at both ends invalid even next to the surface
    _, ok, _ = _lookup(cdf, np.array([[R + h, 0.0, 0.0], [R - h, 0.0, 0.5]]))
    assert np.all(ok == 0.0)


def test_closed_mesh_has_no_boundary():
    v, f = revolve_profile([(0.0, 0.0), (0.3, 0.0), (0.3, 0.4), (0.0, 0.4)], n_theta=32)
    assert len(_boundary_loop_samples(v, f, 0.01)) == 0


def test_glass_mid_surface_profile_geometry():
    p = GlassProfile()
    poly = glass_mid_surface_profile(p)
    assert poly[0][0] == 0.0                                 # starts on the axis
    assert np.isclose(poly[1][0], 0.5 * (p.inner_radius + p.outer_radius))
    assert np.isclose(poly[-1][1], p.rim_z)                  # ends at the rim
    v, f = revolve_profile_open(poly, n_theta=32)
    assert len(_boundary_loop_samples(v, f, 0.01)) > 0       # open at the rim


def test_cdf_cache_roundtrip(tmp_path):
    v, f = _flat_sheet()
    a = build_surface_cdf_cached(v, f, res=32, cache_dir=tmp_path)
    b = build_surface_cdf_cached(v, f, res=32, cache_dir=tmp_path)   # cache hit
    assert np.array_equal(a.values, b.values) and np.array_equal(a.valid, b.valid)
    p = tmp_path / "roundtrip.npz"
    save_cdf(a, p)
    c = load_cdf(p)
    assert np.array_equal(a.values, c.values) and a.band == c.band


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
