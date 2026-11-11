"""Moon and planet dating, checked against an independent ephemeris: JPL DE430 through skyfield (the
2015-03-02 excerpt shipped with skyfield's tests), while sky-clock uses astropy's built-in series."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from skyclock import bodies as B
from skyclock.geometry import unit_to_radec
from skyclock.platesolve import PlateSolution

sf = pytest.importorskip("skyfield.api")
DE430 = Path(sf.__file__).parent / "tests" / "data" / "de430-2015-03-02.bsp"
if not DE430.exists():
    pytest.skip("skyfield DE430 test excerpt not installed", allow_module_level=True)

WHEN = datetime(2015, 3, 2, 20, 30, tzinfo=timezone.utc)
LAT, LON = 59.33, 18.06
W, H = 3000, 2000


def _truth_dirs():
    eph = sf.load_file(str(DE430))
    t = sf.load.timescale(builtin=True).from_datetime(WHEN)
    obs = eph["earth"] + sf.wgs84.latlon(LAT, LON)
    out = {}
    for name, key in [("moon", "moon"), ("jupiter", "jupiter barycenter"), ("venus", "venus"),
                      ("mars", "mars barycenter")]:
        p = obs.at(t).observe(eph[key]).position.km
        out[name] = p / np.linalg.norm(p)
    return out


def _plate(center, fov_deg=70.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ra, dec = unit_to_radec(center)
    w.wcs.crval = [float(ra), float(dec)]
    w.wcs.crpix = [W / 2 + 0.5, H / 2 + 0.5]
    s = 2 * np.degrees(np.tan(np.radians(fov_deg / 2))) / W
    w.wcs.cd = [[-s, 0], [0, s]]
    return PlateSolution(w, (H, W), "test")


def _mark(plate, u, name=None):
    x, y = plate.unit2pix(u[None, :])
    return B.BodyMark(float(x[0]), float(y[0]), name)


@pytest.fixture(scope="module")
def truth():
    return _truth_dirs()


def test_jupiter_dates_the_photo(truth):
    plate = _plate(truth["jupiter"])
    m = _mark(plate, truth["jupiter"], "jupiter")
    c = B.date_candidates(m, plate, datetime(2014, 1, 1, tzinfo=timezone.utc), datetime(2016, 12, 31, tzinfo=timezone.utc))
    best = min(c, key=lambda x: abs((x.time - WHEN).total_seconds()))
    dt = (best.time - WHEN).total_seconds()
    assert abs(dt) < 3 * best.sigma_t_s, (dt, best.sigma_t_s)
    assert best.sigma_t_s < 5 * 86400
    assert 1 <= len(c) <= 4  # a retrograde loop gives up to 3 dates per year, not a smear


def test_moon_with_place_gives_minutes(truth):
    plate = _plate(truth["moon"])
    m = _mark(plate, truth["moon"], "moon")
    c = B.date_candidates(m, plate, WHEN - timedelta(days=40), WHEN + timedelta(days=40), LAT, LON)
    best = min(c, key=lambda x: abs((x.time - WHEN).total_seconds()))
    dt = (best.time - WHEN).total_seconds()
    assert abs(dt) < 3 * best.sigma_t_s and best.sigma_t_s < 15 * 60, (dt, best.sigma_t_s)
    # without the place the parallax (~40 arcmin here) widens it to hours, but the truth stays inside
    c2 = B.date_candidates(m, plate, WHEN - timedelta(days=5), WHEN + timedelta(days=5))
    b2 = min(c2, key=lambda x: abs((x.time - WHEN).total_seconds()))
    assert abs((b2.time - WHEN).total_seconds()) < 3 * b2.sigma_t_s and b2.sigma_t_s > 1800


def test_two_bodies_agree_on_one_date(truth):
    """Moon (no place given) + Jupiter in one 70-deg frame: together, one date in a 3-year window."""
    mid = truth["moon"] + truth["jupiter"]
    plate = _plate(mid / np.linalg.norm(mid))
    marks = [_mark(plate, truth["moon"], "moon"), _mark(plate, truth["jupiter"], "jupiter")]
    start, end = datetime(2014, 1, 1, tzinfo=timezone.utc), datetime(2016, 12, 31, tzinfo=timezone.utc)
    cands = [B.date_candidates(m, plate, start, end) for m in marks]
    comb = B.combine(cands)
    assert comb
    t, s, _, _ = comb[0]
    assert abs((t - WHEN).total_seconds()) < 3 * s, ((t - WHEN).total_seconds(), s)
    assert len(cands[0]) > 10 and len(comb) <= 3  # the Moon alone: dozens of dates; with Jupiter: one or few


def test_wrong_identification_finds_no_date(truth):
    """Failure control: calling Jupiter 'Mars' must not produce the photo's date."""
    plate = _plate(truth["jupiter"])
    m = _mark(plate, truth["jupiter"], "mars")
    c = B.date_candidates(m, plate, WHEN - timedelta(days=200), WHEN + timedelta(days=200))
    assert all(abs((x.time - WHEN).total_seconds()) > 3 * x.sigma_t_s for x in c)


def test_auto_find_skips_matched_stars(truth):
    from skyclock.detect import Detection, Star
    plate = _plate(truth["jupiter"])
    jx, jy = plate.unit2pix(truth["jupiter"][None, :])
    stars = [Star(float(jx[0]), float(jy[0]), 5e5), Star(100.0, 100.0, 9e5)]
    plate.matched_xy = np.array([[100.0, 100.0]])  # the brighter source is a catalogue star
    det = Detection([], stars, (H, W), 1.0)
    assert [(s.x, s.y) for s in B.bright_unmatched(det, plate)] == [(stars[0].x, stars[0].y)]
    hits = B.auto_find(det, plate, WHEN - timedelta(days=30), WHEN + timedelta(days=30))
    assert hits and any(c.body == "jupiter" and abs((c.time - WHEN).total_seconds()) < 3 * c.sigma_t_s
                        for c in hits[0][1])


def _src(plate, ra, dec, flux):
    from skyclock.detect import Star
    from skyclock.geometry import radec_to_unit
    x, y = plate.unit2pix(radec_to_unit(np.array([ra]), np.array([dec])))
    return Star(float(x[0]), float(y[0]), flux)


def test_auto_find_skips_unmatched_bright_stars(truth):
    """Regulus and Pollux in the frame with Jupiter, all three left unmatched by the solver (saturated): the
    finder must skip both stars and keep Jupiter. Failure control: with the star check off, Regulus (the
    brightest) comes first."""
    from skyclock.detect import Detection
    from skyclock.brightstars import index_available, _query_index
    plate = _plate(truth["jupiter"])
    jx, jy = plate.unit2pix(truth["jupiter"][None, :])
    from skyclock.detect import Star
    jup = Star(float(jx[0]), float(jy[0]), 5e5)
    stars = [jup, _src(plate, 152.0930, 11.9672, 9e5), _src(plate, 116.3290, 28.0262, 8e5)]  # Regulus, Pollux
    if index_available():  # a mag 2.5-3.8 star that only the Tycho-2 index knows
        ra, dec = (float(v) for v in unit_to_radec(truth["jupiter"]))
        q = _query_index(ra, dec, 20.0, 3.8)
        q = q[q[:, 2] > 2.6]
        assert len(q)
        stars.append(_src(plate, q[0, 0], q[0, 1], 7e5))
    plate.matched_xy = np.zeros((0, 2))
    det = Detection([], stars, (H, W), 1.0)
    assert [(s.x, s.y) for s in B.bright_unmatched(det, plate)] == [(jup.x, jup.y)]
    first = B.bright_unmatched(det, plate, star_check=False)[0]
    assert (first.x, first.y) == (stars[1].x, stars[1].y)
    hits = B.auto_find(det, plate, WHEN - timedelta(days=30), WHEN + timedelta(days=30))
    assert len(hits) == 1 and abs(hits[0][0].x - jup.x) < 1e-6


def test_builtin_bright_stars_agree_with_tycho2():
    """The built-in table (Hipparcos via PyEphem) against the Tycho-2 index, star by star where Tycho-2 has
    the star: catches a transcription error in the table."""
    from skyclock.brightstars import BRIGHT, index_available, _query_index
    if not index_available():
        pytest.skip("query-starkd / Tycho-2 index not installed")
    found = 0
    for name, ra, dec, mag in BRIGHT:
        q = _query_index(ra, dec, 0.1, 9.0)
        q = q[np.abs(q[:, 2] - mag) < 0.6] if len(q) else q
        if len(q):
            found += 1
            sep = np.hypot(((q[:, 0] - ra + 180) % 360 - 180) * np.cos(np.radians(dec)), q[:, 1] - dec).min()
            assert sep < 0.02, (name, sep)
    assert found >= 60, found
