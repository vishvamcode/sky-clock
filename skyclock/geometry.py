"""Geometry core: time, Earth frames, WGS84, ray/ellipsoid, Sun, aberration.

Conventions
-----------
* Positions are in km, velocities in km/s.
* ITRS = Earth-fixed frame (polar motion ignored, < 15 m).
* "Sky" unit vectors are GCRS (~ICRS orientation). WCS solutions from star catalogues give
  *astrometric* directions; the conversion between a satellite's geometric GCRS direction and
  the WCS frame is ``geo_to_wcs`` / ``wcs_to_geo`` (annual aberration, up to ~20 arcsec).
* Times: a ``Epoch`` holds a reference instant; other instants are float seconds after it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from skyfield.api import load
from skyfield.framelib import itrs as _itrs

TS = load.timescale(builtin=True)

WGS84_A = 6378.137
WGS84_F = 1 / 298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = WGS84_F * (2 - WGS84_F)
EARTH_ROT = 7.292115146706979e-5  # rad / SI second (UT1 ~ UTC rate)
C_KM_S = 299792.458
AU_KM = 149597870.7
R_EARTH_MEAN = 6371.0


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rot_z(angle):
    """Active rotation by +angle about z (works for scalar angle)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Epoch:
    """Reference instant. All solver times are seconds after ``self``."""

    dt: datetime  # timezone-aware UTC

    def __post_init__(self):
        if self.dt.tzinfo is None:
            self.dt = self.dt.replace(tzinfo=timezone.utc)
        self.t = TS.from_datetime(self.dt)
        self.R0 = _itrs.rotation_at(self.t)  # GCRS -> ITRS at reference
        utc_jd = 2440587.5 + self.dt.timestamp() / 86400.0  # UTC-based JD, as SGP4 expects
        self.jd_utc = float(np.floor(utc_jd - 0.5) + 0.5)
        self.fr_utc = float(utc_jd - self.jd_utc)
        self.dut1 = float(self.t.dut1)

    # ---- time helpers
    def jd_fr(self, sec):
        sec = np.atleast_1d(np.asarray(sec, dtype=float))
        fr = self.fr_utc + sec / 86400.0
        whole = np.floor(fr)
        return self.jd_utc + whole, fr - whole

    def datetime_at(self, sec: float) -> datetime:
        from datetime import timedelta
        return self.dt + timedelta(seconds=float(sec))

    def seconds_from(self, dt: datetime) -> float:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (dt - self.dt).total_seconds()

    # ---- frames
    def gcrs_to_itrs(self, sec):
        """Matrix (3x3) for one time (sec scalar)."""
        return rot_z(-EARTH_ROT * float(sec)) @ self.R0

    def itrs_to_gcrs_vecs(self, v_itrs, sec):
        """Rotate ITRS vectors (..., 3) at times sec (broadcast over leading axis) into GCRS."""
        sec = np.asarray(sec, dtype=float)
        ang = EARTH_ROT * sec
        c, s = np.cos(ang), np.sin(ang)
        x, y, z = v_itrs[..., 0], v_itrs[..., 1], v_itrs[..., 2]
        # undo Earth rotation (rotate by +ang), then R0^T
        xr = c * x - s * y
        yr = s * x + c * y
        xr, yr, z = np.broadcast_arrays(xr, yr, z)
        w = np.stack([xr, yr, z], axis=-1)
        return w @ self.R0  # (R0^T w) as row vectors

    def gcrs_to_itrs_vecs(self, v_gcrs, sec):
        sec = np.asarray(sec, dtype=float)
        w = v_gcrs @ self.R0.T  # R0 v
        ang = -EARTH_ROT * sec
        c, s = np.cos(ang), np.sin(ang)
        x, y, z = w[..., 0], w[..., 1], w[..., 2]
        return np.stack(np.broadcast_arrays(c * x - s * y, s * x + c * y, z), axis=-1)


# ---------------------------------------------------------------- WGS84
def geodetic_to_itrs(lat_deg, lon_deg, h_km=0.0):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + h_km) * np.cos(lat) * np.cos(lon)
    y = (N + h_km) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + h_km) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def itrs_to_geodetic(xyz):
    """Vectorised (Bowring) inverse. Returns lat_deg, lon_deg, h_km."""
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(5):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.maximum(np.cos(lat), 1e-12) - N
        lat = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + h)))
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.maximum(np.cos(lat), 1e-12) - N
    return np.degrees(lat), np.degrees(lon), h


def enu_basis(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], -1)
    north = np.cross(up, east)
    return east, north, up


def ray_ellipsoid(origin, direction, h_km=0.0):
    """Nearest positive intersection of rays origin + t*direction with the WGS84 ellipsoid
    inflated by h_km (approximately: semi-axes a+h, b+h). Returns (point (...,3), t, hit mask)."""
    a, b = WGS84_A + h_km, WGS84_B + h_km
    o = np.asarray(origin, float) * np.array([1 / a, 1 / a, 1 / b])
    d = np.asarray(direction, float) * np.array([1 / a, 1 / a, 1 / b])
    A = np.sum(d * d, -1)
    B = 2 * np.sum(o * d, -1)
    C = np.sum(o * o, -1) - 1
    disc = B * B - 4 * A * C
    hit = disc >= 0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t1 = (-B - sq) / (2 * A)
    t2 = (-B + sq) / (2 * A)
    t = np.where(t1 > 0, t1, t2)
    hit &= t > 0
    pt = np.asarray(origin, float) + t[..., None] * np.asarray(direction, float)
    return pt, t, hit


def line_ellipsoid(p0, ell, h_km=0.0):
    """Both intersections of lines p0 + lam*ell with the ellipsoid. Returns (pts (...,2,3), hit)."""
    a, b = WGS84_A + h_km, WGS84_B + h_km
    s = np.array([1 / a, 1 / a, 1 / b])
    o, d = p0 * s, ell * s
    A = np.sum(d * d, -1)
    B = 2 * np.sum(o * d, -1)
    C = np.sum(o * o, -1) - 1
    disc = B * B - 4 * A * C
    hit = (disc >= 0) & (A > 0)
    sq = np.sqrt(np.where(hit, disc, 0.0))
    Ae = np.where(A > 0, A, 1.0)
    l1 = (-B - sq) / (2 * Ae)
    l2 = (-B + sq) / (2 * Ae)
    pts = np.stack([p0 + l1[..., None] * ell, p0 + l2[..., None] * ell], axis=-2)
    return pts, hit


# ---------------------------------------------------------------- Sun & aberration
def sun_gcrs(epoch: Epoch, sec=0.0):
    """Low-precision Sun unit vector in GCRS (~0.01 deg; Astronomical Almanac)."""
    jd = epoch.jd_utc + epoch.fr_utc + np.asarray(sec, float) / 86400.0
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360)
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4e-7 * n)
    v = np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], -1)
    return v


def earth_beta(epoch: Epoch, sec=0.0):
    """Earth's barycentric velocity / c in GCRS (approximation adequate to < 1 arcsec)."""
    h = 0.25 * 86400.0
    s1 = sun_gcrs(epoch, sec - h)
    s2 = sun_gcrs(epoch, sec + h)
    v = -AU_KM * (s2 - s1) / (2 * h)  # Earth moves opposite to the Sun's apparent motion
    return v / C_KM_S


def geo_to_wcs(u, beta):
    """Geometric GCRS direction of a near-Earth object -> direction in the star (astrometric) frame."""
    u = np.asarray(u, float)
    w = u - beta + np.sum(u * beta, -1, keepdims=True) * u
    return unit(w)


def wcs_to_geo(u, beta):
    u = np.asarray(u, float)
    w = u + beta - np.sum(u * beta, -1, keepdims=True) * u
    return unit(w)


def sunlit(r_itrs, sun_itrs, margin_km=0.0):
    """Cylindrical shadow test. margin_km > 0 is permissive (counts near-shadow objects as lit)."""
    proj = np.sum(r_itrs * sun_itrs, axis=-1)
    perp = np.linalg.norm(r_itrs - proj[..., None] * sun_itrs, axis=-1)
    return (proj > 0) | (perp > R_EARTH_MEAN - margin_km)


def radec_to_unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def unit_to_radec(u):
    u = np.asarray(u, float)
    ra = np.degrees(np.arctan2(u[..., 1], u[..., 0])) % 360
    dec = np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1)))
    return ra, dec


def great_circle_distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_MEAN * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
