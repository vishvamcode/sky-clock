from datetime import timedelta

import numpy as np

from skyclock import zenith as Z
from skyclock.geometry import Epoch, earth_beta, enu_basis, geo_to_wcs, great_circle_distance_km, unit_to_radec

from conftest import LAT, LON, WHEN


def _pix_of_altaz(scene, alt_deg, az_deg):
    """Pixel of an Earth-fixed direction (alt, az) at the observer, as the simulated tripod photo shows it."""
    ep = Epoch(WHEN)
    t_mid = scene["truth"]["exposure_s"] / 2
    e, n, up = enu_basis(LAT, LON)
    a, z = np.radians(alt_deg), np.radians(az_deg)
    d = np.cos(a) * (np.sin(z) * e + np.cos(z) * n) + np.sin(a) * up
    g = ep.itrs_to_gcrs_vecs(d, t_mid)
    ra, dec = unit_to_radec(geo_to_wcs(g, earth_beta(ep, t_mid)))
    x, y = scene["wcs"].all_world2pix([ra], [dec], 0)
    return float(x[0]), float(y[0])


def _vertical(scene, az, alt1, alt2, jitter=0.0, rng=None):
    p = _pix_of_altaz(scene, alt1, az) + _pix_of_altaz(scene, alt2, az)
    if jitter and rng is not None:
        p = tuple(np.array(p) + rng.normal(0, jitter, 4))
    return list(p)


def test_verticals_give_the_place(scene):
    """Two marked vertical edges (pole, building corner) + the plate solution + the time give the place.
    Edges cover 20 deg of altitude, clicked with 1.5 px errors."""
    pt = scene["truth"]["pointing"]
    rng = np.random.default_rng(1)
    marks = Z.UpMarks(verticals=[_vertical(scene, pt["az"] - 20, pt["alt"] - 20, pt["alt"], 1.5, rng),
                                 _vertical(scene, pt["az"] + 25, pt["alt"] - 20, pt["alt"] + 5, 1.5, rng)])
    zen = Z.find_zenith(marks, scene["plate"])
    assert zen is not None
    t_mid = WHEN + timedelta(seconds=scene["truth"]["exposure_s"] / 2)
    lat, lon, cov = Z.latlon_from_zenith(zen, t_mid)
    err = great_circle_distance_km(LAT, LON, lat, lon)
    sig = np.sqrt(np.trace(cov))
    assert err < 3 * sig + 1.0, (err, sig)
    assert sig < 60  # 2 good edges on a 70-deg lens: tens of km at most


def test_wrong_edges_are_rejected_by_their_own_error_bars(scene):
    """Failure control: edges at the wrong azimuth tilt (not vertical) must not give the true place."""
    pt = scene["truth"]["pointing"]
    a = _pix_of_altaz(scene, pt["alt"] - 20, pt["az"] - 20) + _pix_of_altaz(scene, pt["alt"], pt["az"] - 10)
    b = _pix_of_altaz(scene, pt["alt"] - 20, pt["az"] + 25) + _pix_of_altaz(scene, pt["alt"] + 5, pt["az"] + 35)
    zen = Z.find_zenith(Z.UpMarks(verticals=[list(a), list(b)]), scene["plate"])
    lat, lon, cov = Z.latlon_from_zenith(zen, WHEN + timedelta(seconds=5))
    assert great_circle_distance_km(LAT, LON, lat, lon) > 3 * np.sqrt(np.trace(cov))


def test_time_from_zenith(scene):
    pt = scene["truth"]["pointing"]
    marks = Z.UpMarks(verticals=[_vertical(scene, pt["az"] - 20, pt["alt"] - 20, pt["alt"]),
                                 _vertical(scene, pt["az"] + 25, pt["alt"] - 20, pt["alt"] + 5)])
    zen = Z.find_zenith(marks, scene["plate"])
    t_mid = WHEN + timedelta(seconds=scene["truth"]["exposure_s"] / 2)
    t, sig_t, lat = Z.time_from_zenith(zen, LON, WHEN + timedelta(hours=2))
    assert abs((t - t_mid).total_seconds()) < max(3 * sig_t, 5.0), ((t - t_mid).total_seconds(), sig_t)
    assert abs(lat - LAT) < 0.5


def _marks(scene):
    pt = scene["truth"]["pointing"]
    rng = np.random.default_rng(1)
    return Z.UpMarks(verticals=[_vertical(scene, pt["az"] - 20, pt["alt"] - 20, pt["alt"], 1.5, rng),
                                _vertical(scene, pt["az"] + 25, pt["alt"] - 20, pt["alt"] + 5, 1.5, rng)])


def test_one_streak_plus_verticals_gives_a_point(scene, constellation):
    """One streak alone gives a line of places; adding two marked verticals picks the point on it."""
    from test_solver import _err, _run
    zen = Z.find_zenith(_marks(scene), scene["plate"])
    for st in scene["det"].streaks:
        st.enabled = st.id == 0
    try:
        r = _run(scene, constellation, mode="place", time=WHEN, sigma_time_s=2, zenith=zen)
        assert r.solutions, r.notes
        sol = r.solutions[0]
        km, s = _err(sol)
        sig = np.hypot(sol.sigma_east_km, sol.sigma_north_km)
        assert km < 3 * sig + 1.0, (km, sig)
        assert sig < 40
        assert abs(s) < 3.0
    finally:
        for st in scene["det"].streaks:
            st.enabled = True


def test_zenith_only_result(scene, constellation):
    """No streaks at all: the verticals still give the place at a known time."""
    zen = Z.find_zenith(_marks(scene), scene["plate"])
    from skyclock import solve as SV
    r = SV.analyze([], scene["plate"], constellation, SV.Settings(mode="place", time=WHEN, zenith=zen))
    assert r.zenith and "lat" in r.zenith and "lon" in r.zenith
    err = great_circle_distance_km(LAT, LON, r.zenith["lat"], r.zenith["lon"])
    assert err < 3 * np.hypot(r.zenith["sigma_east_km"], r.zenith["sigma_north_km"]) + 1.0


def test_wrong_verticals_are_flagged_with_satellites(scene, constellation):
    """Failure control for the combined fit: tilted 'verticals' plus all streaks must raise a warning."""
    from test_solver import _run
    pt = scene["truth"]["pointing"]
    a = _pix_of_altaz(scene, pt["alt"] - 20, pt["az"] - 20) + _pix_of_altaz(scene, pt["alt"], pt["az"] - 10)
    b = _pix_of_altaz(scene, pt["alt"] - 20, pt["az"] + 25) + _pix_of_altaz(scene, pt["alt"] + 5, pt["az"] + 35)
    bad = Z.find_zenith(Z.UpMarks(verticals=[list(a), list(b)]), scene["plate"])
    r = _run(scene, constellation, mode="place", time=WHEN, sigma_time_s=2, zenith=bad)
    assert any(n.startswith("CHECK THE MARKED LINES") for n in r.notes), r.notes
    good = _run(scene, constellation, mode="place", time=WHEN, sigma_time_s=2, zenith=Z.find_zenith(_marks(scene), scene["plate"]))
    assert not any(n.startswith("CHECK THE MARKED LINES") for n in good.notes), good.notes


def test_short_parallel_edges_are_not_used(scene, constellation):
    """Two short, nearly parallel edges side by side (like a chimney) give a zenith too uncertain to trust:
    the solver must say so and not feed it into the fit."""
    from skyclock import solve as SV
    pt = scene["truth"]["pointing"]
    marks = Z.UpMarks(verticals=[_vertical(scene, pt["az"], pt["alt"] - 12, pt["alt"] - 9),
                                 _vertical(scene, pt["az"] + 1.5, pt["alt"] - 12, pt["alt"] - 9)])
    r = SV.analyze([], scene["plate"], constellation, SV.Settings(mode="place", time=WHEN, up_marks=marks))
    assert r.zenith is None
    assert any("were not used" in n for n in r.notes), r.notes


def test_exact_verticals_are_exact(scene):
    """Noise-free edges: the zenith -> place chain (aberration, Earth rotation, frames) must be exact to
    well under a kilometre. Catches frame or timing slips that the noisy tests are too loose to see."""
    pt = scene["truth"]["pointing"]
    marks = Z.UpMarks(verticals=[_vertical(scene, pt["az"] - 25, pt["alt"] - 25, pt["alt"] + 20),
                                 _vertical(scene, pt["az"] + 25, pt["alt"] - 25, pt["alt"] + 20)])
    zen = Z.find_zenith(marks, scene["plate"])
    t_mid = WHEN + timedelta(seconds=scene["truth"]["exposure_s"] / 2)
    lat, lon, _ = Z.latlon_from_zenith(zen, t_mid)
    assert great_circle_distance_km(LAT, LON, lat, lon) < 0.3, great_circle_distance_km(LAT, LON, lat, lon)
