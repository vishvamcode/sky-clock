"""Daytime: the Sun's altitude (and, with a compass heading, its azimuth) from a photo, and what it says
about place and time.

Measuring the Sun in a photo (no stars needed, only the camera's field of view, from EXIF):

* **the Sun itself** in the frame gives its direction directly;
* **pole + shadow pairs**: for a vertical object standing on level ground, the line from the tip of its
  shadow to its top points at the Sun. In the image, all such lines meet at the Sun's vanishing point,
  even when the Sun is outside the frame. Two or more pairs fix it. The same poles (base to top) are
  vertical edges, so they also fix the zenith; the Sun's altitude is the angle between the two.
* a **compass heading** of the camera (phones store it as GPSImgDirection) adds the Sun's azimuth.

Then, with the Sun's apparent altitude a (and azimuth A):

* time known, altitude only: the place lies on a circle around the sub-solar point, radius 90 deg - a;
* time known, altitude and azimuth: one place;
* place and date known: the time (one morning and one afternoon solution, the azimuth picks one);
* place known, date unknown, altitude and azimuth: date (two per year) and time;
* several photos at different known times from one place: the place.

Sun positions: astropy's built-in ephemeris (checked against JPL DE430 in the tests), annual aberration
and standard atmospheric refraction (Bennett) added, so they match what a shadow shows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.optimize import least_squares

from .geometry import Epoch, earth_beta, enu_basis, unit, wcs_to_geo

DEG = np.pi / 180


# ------------------------------------------------------------------ Sun position
def refraction_deg(alt_true_deg):
    """Bennett-style refraction (deg) for a true altitude, standard atmosphere (10 C, 1010 hPa)."""
    a = np.asarray(alt_true_deg, float)
    r = 1.02 / 60 / np.tan(np.radians(a + 10.3 / (a + 5.11)))
    return np.where(a > -1.0, r, 0.0)


def sun_enu(times, lat, lon, h_km=0.0):
    """Unit vectors (east, north, up) of the true (unrefracted, aberrated) Sun direction at each time."""
    from .bodies import body_dirs
    times = list(np.atleast_1d(times))
    ep = Epoch(times[0])
    sec = np.array([ep.seconds_from(t) for t in times])
    d, _ = body_dirs("sun", times, lat, lon, h_km)  # astrometric
    d = np.array([wcs_to_geo(di, earth_beta(ep, s)) for di, s in zip(d, sec)])  # apparent (aberration)
    itrs = ep.gcrs_to_itrs_vecs(d, sec)
    e, n, u = enu_basis(lat, lon)
    return np.stack([itrs @ e, itrs @ n, itrs @ u], -1)


def sun_altaz(times, lat, lon, h_km=0.0, refract=True):
    """Apparent altitude and azimuth (deg, azimuth from north through east) of the Sun's centre."""
    v = sun_enu(times, lat, lon, h_km)
    alt = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
    az = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360
    if refract:
        alt = alt + refraction_deg(alt)
    return alt, az


# ------------------------------------------------------------------ measuring the Sun in a photo
class Camera:
    """Pinhole camera from the field of view (EXIF 35 mm equivalent focal length). Lens distortion is
    ignored, so keep marks away from the extreme corners of wide lenses. Frame: x right, y down, z forward."""

    def __init__(self, shape, fov_x_deg):
        self.shape = tuple(shape)
        H, W = self.shape
        self.f = (W / 2) / np.tan(np.radians(fov_x_deg) / 2)
        self.cx, self.cy = W / 2 - 0.5, H / 2 - 0.5

    def pix2unit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return unit(np.stack([x - self.cx, y - self.cy, np.full_like(x, self.f)], -1))

    def unit2pix(self, u):
        u = np.atleast_2d(u)
        return self.cx + self.f * u[:, 0] / u[:, 2], self.cy + self.f * u[:, 1] / u[:, 2]

    def pixel_scale_rad(self, x=None, y=None):
        return 1.0 / self.f


@dataclass
class SunMarks:
    """Pixels (x right, y down). Each pole: [x_base, y_base, x_top, y_top, x_shadow_tip, y_shadow_tip]: a
    vertical object on level ground and the tip of its shadow. ``sun``: the Sun's centre if in frame.
    ``verticals``: extra vertical edges [x1, y1, x2, y2]."""
    poles: list = field(default_factory=list)
    sun: list | None = None
    verticals: list = field(default_factory=list)
    heading_deg: float | None = None  # compass azimuth of the camera's optical axis (true north)
    heading_sigma_deg: float = 8.0  # phone compasses are often off by 5 to 10 deg
    click_sigma_px: float = 2.0
    fov_sigma_frac: float = 0.02  # EXIF focal lengths are rounded


@dataclass
class SunFix:
    alt_deg: float
    alt_sigma_deg: float
    az_deg: float | None = None
    az_sigma_deg: float | None = None
    notes: list = field(default_factory=list)

    def to_dict(self):
        return dict(alt_deg=self.alt_deg, alt_sigma_deg=self.alt_sigma_deg, az_deg=self.az_deg,
                    az_sigma_deg=self.az_sigma_deg, notes=self.notes)


def _null_dir(normals):
    """Unit vector most nearly perpendicular to all the given plane normals (smallest eigenvector)."""
    M = sum(np.outer(n, n) for n in normals)
    return np.linalg.eigh(M)[1][:, 0]


def _measure(m: SunMarks, cam: Camera):
    """Sun direction s and zenith z in the camera frame, or None."""
    vn = []
    for p in m.poles:
        u = cam.pix2unit([p[0], p[2]], [p[1], p[3]])
        vn.append(unit(np.cross(u[0], u[1])))
    for v in m.verticals:
        u = cam.pix2unit([v[0], v[2]], [v[1], v[3]])
        vn.append(unit(np.cross(u[0], u[1])))
    if len(vn) < 2:
        return None
    z = _null_dir(vn)
    # "up" is from base to top: in the image, going from base to top moves toward the zenith's vanishing point
    p = m.poles[0] if m.poles else m.verticals[0]
    ub, ut = cam.pix2unit([p[0], p[2]], [p[1], p[3]])
    if z @ (ut - ub) < 0:
        z = -z
    if m.sun is not None:
        s = cam.pix2unit([m.sun[0]], [m.sun[1]])[0]
    else:
        if len(m.poles) < 2:
            return None
        sn = []
        for p in m.poles:
            u = cam.pix2unit([p[4], p[2]], [p[5], p[3]])  # shadow tip, top
            sn.append(unit(np.cross(u[0], u[1])))
        s = _null_dir(sn)
        if s @ z < 0:  # shadows exist only with the Sun above the horizon
            s = -s
    return s, z


def _altaz(s, z, heading):
    alt = np.degrees(np.arcsin(np.clip(s @ z, -1, 1)))
    az = None
    if heading is not None:
        f = unit(np.array([0.0, 0.0, 1.0]) - z[2] * z)  # optical axis projected on the horizontal plane
        r = np.cross(f, z)  # horizontal, 90 deg clockwise from f seen from above (like east from north)
        az = (heading + np.degrees(np.arctan2(s @ r, s @ f))) % 360
    return alt, az


def measure_sun(m: SunMarks, shape, fov_x_deg, n_mc=400, seed=0) -> SunFix | None:
    """Sun altitude (and azimuth if the heading is known) with Monte-Carlo errors from the clicks and
    the focal length."""
    cam = Camera(shape, fov_x_deg)
    r = _measure(m, cam)
    if r is None:
        return None
    alt0, az0 = _altaz(*r, m.heading_deg)
    rng = np.random.default_rng(seed)
    alts, azs = [], []
    for _ in range(n_mc):
        def jit(v):
            return [float(c) + rng.normal(0, m.click_sigma_px) for c in v]
        mm = SunMarks([jit(p) for p in m.poles], jit(m.sun) if m.sun is not None else None,
                      [jit(v) for v in m.verticals],
                      None if m.heading_deg is None else m.heading_deg + rng.normal(0, m.heading_sigma_deg))
        c2 = Camera(shape, fov_x_deg * (1 + rng.normal(0, m.fov_sigma_frac)))
        rr = _measure(mm, c2)
        if rr is None:
            continue
        a, b = _altaz(*rr, mm.heading_deg)
        alts.append(a)
        if b is not None:
            azs.append((b - az0 + 180) % 360 - 180)
    notes = []
    sa = float(np.std(alts)) if alts else float("nan")
    saz = float(np.std(azs)) if azs else None
    if m.sun is None and len(m.poles) >= 2:
        notes.append("Sun direction from shadows: assumes the poles are vertical and the ground between each "
                     "pole and its shadow tip is level.")
    if alt0 < 5:
        notes.append("Sun below 5 deg: refraction varies with the weather there; altitude errors grow.")
    return SunFix(float(alt0), sa, None if az0 is None else float(az0), saz, notes)


def shadow_ratio_altitude(height, length, rel_sigma=0.02):
    """Altitude (deg) from a measured object height and shadow length on level ground."""
    a = np.degrees(np.arctan2(height, length))
    # d(alt)/d(ln ratio) = sin(alt) cos(alt)
    return float(a), float(np.degrees(np.sin(np.radians(a)) * np.cos(np.radians(a)) * np.sqrt(2) * rel_sigma))


# ------------------------------------------------------------------ what the Sun says
def times_for_altitude(fix: SunFix, lat, lon, day: datetime, h_km=0.0):
    """Times (UTC) within 12 h of ``day`` (an approximate time, e.g. midday local) when the Sun had the measured altitude at (lat, lon), with
    1-sigma (s). With a measured azimuth, the matching time is ranked first."""
    if day.tzinfo is None:
        day = day.replace(tzinfo=timezone.utc)
    t0 = day - timedelta(hours=12)  # the 24 hours centred on the approximate time given
    grid = [t0 + timedelta(minutes=5 * k) for k in range(12 * 24 + 1)]
    alt, az = sun_altaz(grid, lat, lon, h_km)
    d = alt - fix.alt_deg
    out = []
    for i in np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]:
        f = d[i] / (d[i] - d[i + 1])
        t = grid[i] + timedelta(seconds=300 * float(f))
        rate = abs(d[i + 1] - d[i]) / 300.0  # deg/s
        a2, z2 = sun_altaz([t], lat, lon, h_km)
        sig = fix.alt_sigma_deg / max(rate, 1e-9)
        miss = None
        if fix.az_deg is not None:
            miss = abs((z2[0] - fix.az_deg + 180) % 360 - 180) / max(fix.az_sigma_deg or 5.0, 0.1)
        out.append(dict(time_utc=t, sigma_t_s=float(sig), az_deg=float(z2[0]), az_miss_sigma=miss))
    # near noon the measured altitude can sit just above the day's maximum (noise): then the time is the
    # culmination, known only to the width of the flat top of the altitude curve
    i = int(np.argmax(alt))
    if 0 < i < len(alt) - 1 and fix.alt_deg > alt[i] - 2 * fix.alt_sigma_deg and \
            not any(abs((o["time_utc"] - grid[i]).total_seconds()) < 3 * o["sigma_t_s"] for o in out):
        k = (alt[i - 1] - 2 * alt[i] + alt[i + 1]) / 300.0 ** 2  # deg/s^2 (negative)
        sig = float(np.sqrt(2 * max(fix.alt_sigma_deg, 1e-3) / max(-k, 1e-12)))
        z2 = az[i]
        miss = None if fix.az_deg is None else abs((z2 - fix.az_deg + 180) % 360 - 180) / max(fix.az_sigma_deg or 5.0, 0.1)
        out.append(dict(time_utc=grid[i], sigma_t_s=sig, az_deg=float(z2), az_miss_sigma=miss))
    out.sort(key=lambda o: (o["az_miss_sigma"] is not None and o["az_miss_sigma"] > 3, o["az_miss_sigma"] or 0))
    return out


def subsolar_point(when):
    """Latitude, longitude (deg) where the Sun is overhead (geocentric, no refraction)."""
    v = sun_enu([when], 0.0, 0.0)[0]  # at (0, 0): east = +y_itrs, north = +z_itrs, up = +x_itrs
    x, y, zc = v[2], v[0], v[1]
    return float(np.degrees(np.arcsin(zc))), float(np.degrees(np.arctan2(y, x)))


def altitude_circle(fix: SunFix, when, n=360):
    """Places where the Sun had the measured altitude at ``when``: [[lat, lon], ...] (the locus)."""
    la0, lo0 = np.radians(subsolar_point(when))
    rho = np.radians(90.0 - (fix.alt_deg - refraction_deg(fix.alt_deg)))
    b = np.linspace(0, 2 * np.pi, n)
    la = np.arcsin(np.sin(la0) * np.cos(rho) + np.cos(la0) * np.sin(rho) * np.cos(b))
    lo = lo0 + np.arctan2(np.sin(b) * np.sin(rho) * np.cos(la0), np.cos(rho) - np.sin(la0) * np.sin(la))
    return np.column_stack([np.degrees(la), (np.degrees(lo) + 180) % 360 - 180]).tolist()


def place_from_sun(obs, guess=None, h_km=0.0):
    """Least-squares place from Sun observations [(when, SunFix), ...]: altitude-only fixes at two or more
    times, or one fix with azimuth. Returns (lat, lon, cov_en_km 2x2, rms) or None."""
    def resid(p):
        r = []
        for when, fx in obs:
            a, z = sun_altaz([when], p[0], p[1], h_km)
            r.append((a[0] - fx.alt_deg) / fx.alt_sigma_deg)
            if fx.az_deg is not None:
                r.append(((z[0] - fx.az_deg + 180) % 360 - 180) / fx.az_sigma_deg)
        return np.array(r)

    n_res = sum(1 + (fx.az_deg is not None) for _, fx in obs)
    if n_res < 2:
        return None
    if guess is None:  # coarse global grid
        best = None
        for la in range(-80, 81, 10):
            for lo in range(-180, 180, 15):
                c = float(np.sum(resid([la, lo]) ** 2))
                if best is None or c < best[0]:
                    best = (c, la, lo)
        guess = best[1:]
    sol = least_squares(resid, np.array(guess, float), x_scale=[1.0, 1.0])
    J = sol.jac
    try:
        C = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return None
    k = 111.2
    T = np.diag([k * np.cos(np.radians(sol.x[0])), k])  # (lat, lon) -> (east, north) km
    T = np.array([[0, T[0, 0]], [T[1, 1], 0]])
    cov = T @ C @ T.T
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return float(sol.x[0]), float((sol.x[1] + 180) % 360 - 180), cov, rms


def date_and_time_from_sun(fix: SunFix, lat, lon, start: datetime, end: datetime, h_km=0.0):
    """Place known, date unknown: (time, sigma_t_s, miss) candidates when the Sun had the measured altitude
    AND azimuth. Usually two dates per year (the Sun passes each declination twice)."""
    if fix.az_deg is None:
        raise ValueError("needs the azimuth (camera compass heading)")
    step = timedelta(minutes=10)
    n = int((end - start) / step) + 1
    out = []
    for a in range(0, n, 20000):
        grid = [start + (a + k) * step for k in range(min(20000, n - a))]
        alt, az = sun_altaz(grid, lat, lon, h_km)
        d = np.hypot((alt - fix.alt_deg) / fix.alt_sigma_deg,
                     ((az - fix.az_deg + 180) % 360 - 180) / fix.az_sigma_deg)
        for i in range(1, len(d) - 1):
            if d[i] < 3 and d[i] <= d[i - 1] and d[i] <= d[i + 1]:
                out.append((grid[i], float(d[i])))
    # one entry per day (consecutive samples)
    res = []
    for t, d in sorted(out):
        if res and (t - res[-1][0]).total_seconds() < 6 * 3600:
            if d < res[-1][2]:
                res[-1] = (t, None, d)
            continue
        res.append((t, None, d))
    return [dict(time_utc=t, miss_sigma=d) for t, _, d in res]


def date_ranges(cands, gap_days=2.0):
    """Group date candidates into continuous ranges: [(first, last, best_candidate), ...]."""
    out = []
    for c in cands:
        if out and (c["time_utc"] - out[-1][1]).total_seconds() < gap_days * 86400:
            f, _, b = out[-1]
            out[-1] = (f, c["time_utc"], c if c["miss_sigma"] < b["miss_sigma"] else b)
        else:
            out.append((c["time_utc"], c["time_utc"], c))
    return out
