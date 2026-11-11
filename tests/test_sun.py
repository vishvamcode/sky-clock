"""Daytime: Sun position against JPL DE430 (skyfield), and Sun altitude/azimuth from simulated photos of
poles and their shadows, then place and time from them."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from skyclock import sun as S
from skyclock.geometry import great_circle_distance_km

LAT, LON = 59.33, 18.06
WHEN = datetime(2015, 3, 2, 11, 0, tzinfo=timezone.utc)  # inside skyfield's DE430 test excerpt
W, H = 4000, 3000
FOV = 69.0  # ~26 mm equivalent, a typical phone main camera


def test_sun_position_matches_de430():
    sf = pytest.importorskip("skyfield.api")
    p = Path(sf.__file__).parent / "tests" / "data" / "de430-2015-03-02.bsp"
    if not p.exists():
        pytest.skip("DE430 excerpt missing")
    eph = sf.load_file(str(p))
    ts = sf.load.timescale(builtin=True)
    obs = eph["earth"] + sf.wgs84.latlon(LAT, LON)
    for h in (8, 11, 14):
        t = WHEN.replace(hour=h)
        alt, az, _ = obs.at(ts.from_datetime(t)).observe(eph["sun"]).apparent().altaz()
        a, z = S.sun_altaz([t], LAT, LON, refract=False)
        assert abs(a[0] - alt.degrees) < 0.01 and abs(z[0] - az.degrees) < 0.01, (a, alt.degrees, z, az.degrees)


def _scene(az_cam=200.0, pitch=-8.0, roll=3.0, when=WHEN, poles=((-3, 12, 2.5), (2, 9, 1.8), (6, 15, 3.0))):
    """Synthetic photo: camera 1.6 m up, poles (east, north offsets from a point ahead, height) on level ground."""
    alt, az = S.sun_altaz([when], LAT, LON)
    a, z = np.radians(alt[0]), np.radians(az[0])
    sun_h = np.array([np.sin(z), np.cos(z), 0.0])
    # camera axes in ENU
    A, P, R = np.radians(az_cam), np.radians(pitch), np.radians(roll)
    fwd = np.array([np.sin(A) * np.cos(P), np.cos(A) * np.cos(P), np.sin(P)])
    right0 = np.array([np.cos(A), -np.sin(A), 0.0])
    up0 = np.cross(right0, fwd)
    right = np.cos(R) * right0 + np.sin(R) * up0
    down = -(np.cos(R) * up0 - np.sin(R) * right0)
    cam = S.Camera((H, W), FOV)
    C = np.array([0, 0, 1.6])

    def pix(P3):
        v = P3 - C
        u = np.array([v @ right, v @ down, v @ fwd])
        x, y = cam.unit2pix(u / np.linalg.norm(u))
        return float(x[0]), float(y[0])

    marks = []
    for de, dn, h in poles:
        # place poles ahead of the camera, offset sideways/forward in the camera's own horizontal frame
        base = C * [1, 1, 0] + dn * np.array([np.sin(A), np.cos(A), 0]) + de * right0
        top = base + [0, 0, h]
        tip = base - h / np.tan(a) * sun_h
        marks.append(list(pix(base) + pix(top) + pix(tip)))
    return marks, alt[0], az[0]


def _jitter(marks, s, rng):
    return [list(np.array(m) + rng.normal(0, s, len(m))) for m in marks]


def test_shadows_give_altitude_and_azimuth():
    marks, alt, az = _scene()
    fix = S.measure_sun(S.SunMarks(poles=marks, heading_deg=200.0, click_sigma_px=0.01, fov_sigma_frac=0.0,
                                   heading_sigma_deg=0.01), (H, W), FOV)
    assert abs(fix.alt_deg - alt) < 0.02 and abs((fix.az_deg - az + 180) % 360 - 180) < 0.05, (fix, alt, az)


def test_shadow_altitude_with_click_errors_is_honest():
    rng = np.random.default_rng(3)
    errs = []
    for k in range(12):
        marks, alt, _ = _scene(az_cam=150 + 20 * k, roll=rng.normal(0, 3))
        fix = S.measure_sun(S.SunMarks(poles=_jitter(marks, 2.0, rng)), (H, W), FOV, n_mc=150, seed=k)
        errs.append((fix.alt_deg - alt) / fix.alt_sigma_deg)
    errs = np.array(errs)
    assert np.sqrt(np.mean(errs ** 2)) < 2.0, errs  # error bars not wildly too small


def test_wrong_focal_length_is_a_failure_control():
    """A 2x wrong field of view must move the altitude by far more than the stated error."""
    marks, alt, _ = _scene()
    fix = S.measure_sun(S.SunMarks(poles=marks, click_sigma_px=0.5), (H, W), FOV / 2, n_mc=100)
    assert abs(fix.alt_deg - alt) > 3 * fix.alt_sigma_deg


def test_place_from_altitude_and_azimuth():
    marks, alt, az = _scene()
    rng = np.random.default_rng(1)
    fix = S.measure_sun(S.SunMarks(poles=_jitter(marks, 2.0, rng), heading_deg=200.0 + 4.0), (H, W), FOV)
    r = S.place_from_sun([(WHEN, fix)])
    lat, lon, cov, _ = r
    err = great_circle_distance_km(LAT, LON, lat, lon)
    sig = np.sqrt(np.trace(cov))
    assert err < 3 * sig, (err, sig)
    # the altitude circle passes near the true place
    circ = np.array(S.altitude_circle(fix, WHEN))
    d = min(great_circle_distance_km(LAT, LON, a, b) for a, b in circ)
    assert d < 3 * fix.alt_sigma_deg * 111.2 + 25


def test_time_at_known_place():
    later = WHEN + timedelta(hours=2, minutes=17)
    marks, alt, az = _scene(when=later)
    fix = S.measure_sun(S.SunMarks(poles=_jitter(marks, 1.5, np.random.default_rng(2)), heading_deg=200.0),
                        (H, W), FOV)
    ts = S.times_for_altitude(fix, LAT, LON, later)
    best = ts[0]
    assert abs((best["time_utc"] - later).total_seconds()) < 3 * best["sigma_t_s"], (best, later)


def test_two_photos_fix_the_place():
    rng = np.random.default_rng(5)
    obs = []
    for dh in (0, 3):
        t = WHEN + timedelta(hours=dh)
        marks, _, _ = _scene(when=t)
        obs.append((t, S.measure_sun(S.SunMarks(poles=_jitter(marks, 1.5, rng)), (H, W), FOV)))
    lat, lon, cov, _ = S.place_from_sun(obs, guess=(50.0, 10.0))
    assert great_circle_distance_km(LAT, LON, lat, lon) < 3 * np.sqrt(np.trace(cov))


def test_refraction_matches_skyfield():
    """Shadows show the refracted Sun: compare with skyfield's refraction (standard atmosphere)."""
    sf = pytest.importorskip("skyfield.api")
    p = Path(sf.__file__).parent / "tests" / "data" / "de430-2015-03-02.bsp"
    if not p.exists():
        pytest.skip("DE430 excerpt missing")
    eph = sf.load_file(str(p))
    ts = sf.load.timescale(builtin=True)
    obs = eph["earth"] + sf.wgs84.latlon(LAT, LON)
    for t in (WHEN.replace(hour=7, minute=0), WHEN.replace(hour=7, minute=40), WHEN):
        alt, _, _ = obs.at(ts.from_datetime(t)).observe(eph["sun"]).apparent().altaz(
            temperature_C=10.0, pressure_mbar=1010.0)
        a, _ = S.sun_altaz([t], LAT, LON)
        assert abs(a[0] - alt.degrees) < 0.02, (t, a[0], alt.degrees)
