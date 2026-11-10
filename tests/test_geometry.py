import numpy as np
from datetime import datetime, timezone

from skyclock import geometry as G


def test_wgs84_round_trip():
    lat, lon, h = np.array([59.33, -33.9, 0.0, 89.9]), np.array([18.06, 151.2, -179.9, 10.0]), np.array([0.0, 0.5, 2.0, 0.1])
    x = G.geodetic_to_itrs(lat, lon, h)
    la, lo, hh = G.itrs_to_geodetic(x)
    assert np.allclose(la, lat, atol=1e-9) and np.allclose(lo, lon, atol=1e-9) and np.allclose(hh, h, atol=1e-6)


def test_aberration_inverse():
    ep = G.Epoch(datetime(2026, 3, 1, tzinfo=timezone.utc))
    beta = G.earth_beta(ep)
    assert 9.5e-5 < np.linalg.norm(beta) < 1.05e-4  # ~29.8 km/s / c
    u = G.unit(np.random.default_rng(0).normal(size=(50, 3)))
    back = G.wcs_to_geo(G.geo_to_wcs(u, beta), beta)
    assert np.degrees(np.max(np.arccos(np.clip(np.sum(back * u, 1), -1, 1)))) * 3600 < 0.01


def test_propagation_matches_skyfield(constellation):
    """Our SGP4 TEME->ITRS conversion against Skyfield's EarthSatellite (independent implementation)."""
    from skyfield.api import EarthSatellite, load
    from skyfield.framelib import itrs
    ts = load.timescale(builtin=True)
    when = datetime(2026, 11, 10, 16, 10, tzinfo=timezone.utc)
    ep = G.Epoch(when)
    k = 123
    r, v, ok = constellation.propagate(ep, [0.0, 37.5], idx=[k])
    sat = EarthSatellite.from_satrec(constellation.satrecs[k], ts)
    for j, sec in enumerate([0.0, 37.5]):
        t = ts.from_datetime(ep.datetime_at(sec))
        ref = sat.at(t).frame_xyz(itrs).km
        assert np.linalg.norm(r[0, j] - ref) < 0.01  # km


def test_gcrs_itrs_rotation_consistency():
    ep = G.Epoch(datetime(2026, 11, 10, 16, 10, tzinfo=timezone.utc))
    u = G.unit(np.random.default_rng(1).normal(size=(10, 3)))
    for sec in (0.0, 100.0, 3600.0):
        back = ep.itrs_to_gcrs_vecs(ep.gcrs_to_itrs_vecs(u, sec), sec)
        assert np.allclose(back, u, atol=1e-12)
    # the Earth turns ~15 arcsec per second
    a = ep.gcrs_to_itrs_vecs(np.array([1.0, 0, 0]), 0.0)
    b = ep.gcrs_to_itrs_vecs(np.array([1.0, 0, 0]), 1.0)
    assert abs(np.degrees(np.arccos(a @ b)) * 3600 - 15.04) < 0.05
