"""Which way is up: the zenith from the photo, and what it says about place and time.

The plate solution maps pixels to sky directions. If the photo also shows which way is "up", the
zenith's sky position follows, and the zenith is the point with declination = latitude and right
ascension = local sidereal time. So:

* latitude comes out directly (it barely depends on time),
* longitude follows once the time is known, or the time once the longitude is known: the Earth turns
  15.04 deg per hour under the fixed zenith direction.

Sources of "up", any combination:

* **vertical edges** (building corners, poles, door frames): in a rectilinear photo every vertical line
  points at the zenith, so each marked edge gives a great circle through it; two or more edges
  intersect there. Lens distortion is handled by the plate solution's distortion terms.
* **a level horizon** (sea, flat plain): all of it is at the same apparent altitude (slightly below 0
  because of the observer's height); the zenith is the pole of that circle. Refraction near the horizon
  is large and the plate solution is least accurate there, so this is the weaker source.
* **a level camera** (roll near 0): the image's centre column lies in a vertical plane, a weak but free
  constraint (default +-2 deg).

Directions from the plate solution are in the star (astrometric) frame; annual aberration is removed
before converting to the Earth-fixed frame. Refraction does not move the zenith (it acts along
verticals), so vertical edges need no refraction correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from scipy.optimize import least_squares

from .geometry import EARTH_ROT, Epoch, earth_beta, enu_basis, unit, unit_to_radec, wcs_to_geo

DEG = np.pi / 180


@dataclass
class UpMarks:
    """What the user (or a detector) marked in the photo, in pixels (x right, y down, 0-based)."""
    verticals: list = field(default_factory=list)  # [[x1, y1, x2, y2], ...] along vertical edges
    horizon: list = field(default_factory=list)  # [[x, y], ...] on a level (sea) horizon
    horizon_height_m: float = 2.0  # eye height above the horizon surface (sets the dip)
    camera_level: bool = False  # camera roll ~0 (image centre column is vertical)
    level_sigma_deg: float = 2.0
    click_sigma_px: float = 1.5

    def empty(self):
        return not (self.verticals or len(self.horizon) >= 3 or self.camera_level)


@dataclass
class Zenith:
    u_wcs: np.ndarray  # zenith unit vector in the star (WCS/ICRS) frame
    cov_t: np.ndarray  # 2x2 covariance (rad^2) in the tangent basis (e1, e2) at the zenith
    e1: np.ndarray
    e2: np.ndarray
    n_constraints: int
    rms: float  # rms of the normalised residuals
    notes: list = field(default_factory=list)

    @property
    def radec(self):
        ra, dec = unit_to_radec(self.u_wcs)
        return float(ra), float(dec)

    @property
    def sigma_deg(self):
        """Rough 1-sigma radius (deg) of the zenith position."""
        return float(np.sqrt(max(np.trace(self.cov_t), 0) / 2) / DEG)


def _tangent_basis(z):
    a = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = unit(np.cross(a, z))
    e2 = np.cross(z, e1)
    return e1, e2


def _dip_rad(height_m):
    """Apparent altitude of a sea horizon below the astronomical horizon (standard refraction):
    about 1.76 arcmin * sqrt(height in metres)."""
    return 1.76 / 60 * DEG * np.sqrt(max(height_m, 0.0))


def find_zenith(marks: UpMarks, plate) -> Zenith | None:
    """Least-squares zenith from marked vertical edges, horizon points and/or a level camera."""
    if marks.empty():
        return None
    H, W = plate.shape
    scale = plate.pixel_scale_rad()
    rows = []  # (kind, data, sigma)
    for x1, y1, x2, y2 in marks.verticals:
        u = plate.pix2unit([x1, x2], [y1, y2])
        L = np.arccos(np.clip(u[0] @ u[1], -1, 1))
        if L < 5 * scale:
            continue
        n = unit(np.cross(u[0], u[1]))
        # tilt of the great circle from clicking errors, at the far end of the edge (where it matters)
        sig = np.sqrt(2) * marks.click_sigma_px * scale / L
        rows.append(("v", n, sig))
    if marks.camera_level:
        u = plate.pix2unit([W / 2, W / 2], [0.25 * H, 0.75 * H])
        rows.append(("v", unit(np.cross(u[0], u[1])), marks.level_sigma_deg * DEG))
    hz = None
    if len(marks.horizon) >= 3:
        pts = np.asarray(marks.horizon, float)
        hz = plate.pix2unit(pts[:, 0], pts[:, 1])
        sin_a0 = np.sin(-_dip_rad(marks.horizon_height_m))
        # 0.1 deg extra per point: refraction varies with the weather, and the solution is weakest at the edge
        sig_h = np.hypot(marks.click_sigma_px * scale, 0.1 * DEG)
        for u in hz:
            rows.append(("h", (u, sin_a0), sig_h))
    if not rows:
        return None
    nv = sum(r[0] == "v" for r in rows)
    if nv < 2 and hz is None:
        return None  # one great circle alone does not fix a point
    # initial guess: smallest eigenvector of the vertical normals and/or the horizon plane normal
    M = np.zeros((3, 3))
    for kind, d, s in rows:
        v = d if kind == "v" else d[0]
        M += np.outer(v, v) / s ** 2
    w, V = np.linalg.eigh(M)
    z0 = V[:, 0]
    centre = plate.pix2unit([W / 2], [H / 2])[0]
    if z0 @ centre < 0:
        z0 = -z0  # the camera points above the horizon, so the zenith is on the same side
    e1, e2 = _tangent_basis(z0)

    def zvec(p):
        return unit(z0 + p[0] * e1 + p[1] * e2)

    def resid(p):
        z = zvec(p)
        out = []
        for kind, d, s in rows:
            if kind == "v":
                out.append((d @ z) / s)
            else:
                out.append((d[0] @ z - d[1]) / s)
        return np.array(out)

    sol = least_squares(resid, np.zeros(2), method="lm" if len(rows) > 2 else "trf")
    z = zvec(sol.x)
    J = sol.jac
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return None
    r = resid(sol.x)
    dof = max(len(r) - 2, 1)
    rms = float(np.sqrt(np.sum(r ** 2) / dof))
    notes = []
    if rms > 2 and len(r) > 2:
        cov = cov * rms ** 2  # marks disagree more than clicking errors explain: inflate honestly
        notes.append(f"The marked lines disagree by {rms:.1f}x the expected clicking error (not quite vertical, "
                     "or lens distortion not captured); the zenith uncertainty was enlarged to match.")
    e1n, e2n = _tangent_basis(z)
    # covariance was for (e1, e2) at z0; the bases differ by a tiny rotation, re-express
    R = np.array([[e1n @ e1, e1n @ e2], [e2n @ e1, e2n @ e2]])
    cov = R @ cov @ R.T
    return Zenith(z, cov, e1n, e2n, len(rows), rms, notes)


# ------------------------------------------------------------------ zenith -> place / time
def _zenith_itrs(z_wcs, epoch: Epoch, sec):
    z_geo = wcs_to_geo(z_wcs, earth_beta(epoch, sec))
    return epoch.gcrs_to_itrs_vecs(z_geo, sec)


def latlon_from_zenith(zen: Zenith, when: datetime, n_mc=400, seed=0):
    """Place (lat, lon) from the zenith at a known time, with a 2x2 (east, north) km covariance.

    Latitude is geodetic: the plumb line is the ellipsoid normal to within a few arcseconds."""
    ep = Epoch(when)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(zen.cov_t + 1e-30 * np.eye(2))
    d = rng.standard_normal((n_mc, 2)) @ L.T
    zs = unit(zen.u_wcs[None, :] + d[:, :1] * zen.e1 + d[:, 1:] * zen.e2)
    zi = _zenith_itrs(np.vstack([zen.u_wcs, zs]), ep, 0.0)
    lat = np.degrees(np.arctan2(zi[:, 2], np.hypot(zi[:, 0], zi[:, 1])))
    lon = np.degrees(np.arctan2(zi[:, 1], zi[:, 0]))
    lat0, lon0 = float(lat[0]), float(lon[0])
    dlon = (lon[1:] - lon0 + 180) % 360 - 180
    east = dlon * 111.2 * np.cos(np.radians(lat0))
    north = (lat[1:] - lat0) * 111.2
    cov = np.cov(np.vstack([east, north]))
    return lat0, lon0, cov


def time_from_zenith(zen: Zenith, lon_deg: float, near: datetime):
    """Times (UTC) when the zenith's longitude equals ``lon_deg``: one per sidereal day; the one nearest
    ``near`` is returned with its 1-sigma (s)."""
    ep = Epoch(near)
    zi = _zenith_itrs(zen.u_wcs[None, :], ep, 0.0)[0]
    lon_now = np.degrees(np.arctan2(zi[1], zi[0]))
    # the zenith direction is fixed on the sky; the ground under it moves west at the Earth's rate
    dlon = (lon_now - lon_deg + 180) % 360 - 180  # deg the Earth must still turn
    dt = np.radians(dlon) / EARTH_ROT
    t = near + timedelta(seconds=float(dt))
    _, _, cov = latlon_from_zenith(zen, t)
    lat = float(np.degrees(np.arctan2(zi[2], np.hypot(zi[0], zi[1]))))
    sig_e_km = np.sqrt(cov[0, 0])
    sig_t = sig_e_km / (111.2 * np.cos(np.radians(lat))) / np.degrees(EARTH_ROT)
    return t, float(sig_t), lat


def predicted_zenith_cam(epoch: Epoch, lat, lon, t_sec):
    """Model zenith direction in the camera (WCS) frame for an observer at (lat, lon) at t_sec, for the
    fit: the plate solution describes the sky at mid-exposure."""
    from .geometry import geo_to_wcs
    _, _, up = enu_basis(lat, lon)
    g = epoch.itrs_to_gcrs_vecs(up, t_sec)
    return geo_to_wcs(g, earth_beta(epoch, t_sec))


def zenith_residuals(zen: Zenith, epoch: Epoch, lat, lon, t_sec):
    """Two whitened residuals: model zenith minus measured, in the measured zenith's tangent plane."""
    zm = predicted_zenith_cam(epoch, lat, lon, t_sec)
    d = zm - zen.u_wcs
    v = np.array([d @ zen.e1, d @ zen.e2])
    Linv = np.linalg.inv(np.linalg.cholesky(zen.cov_t))
    return Linv @ v


def zenith_only(zen: Zenith, time: datetime | None = None, lat=None, lon=None, window=None):
    """What the zenith alone says, without satellites (a dict for the result, plus plain-language notes).

    * time known: the place (lat, lon) with an error ellipse;
    * place known: the time (one answer per sidereal day, the one nearest the given time / window centre);
    * neither: latitude only, plus the rule tying longitude to time.
    """
    out = dict(radec_deg=list(zen.radec), sigma_deg=zen.sigma_deg, n_marks=zen.n_constraints)
    notes = list(zen.notes)
    if time is not None:
        la, lo, cov = latlon_from_zenith(zen, time)
        se, sn = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
        out.update(lat=la, lon=lo, sigma_east_km=se, sigma_north_km=sn)
        notes.append(f"From the marked verticals/horizon alone, at the given time: {la:.3f}, {lo:.3f} "
                     f"(+-{sn:.0f} km north-south, +-{se:.0f} km east-west).")
    elif lon is not None:
        near = window or datetime.now().astimezone()
        t, sig_t, la = time_from_zenith(zen, lon, near)
        out.update(lat=la, time_utc=t.isoformat(), sigma_t_s=sig_t)
        notes.append(f"From the marked verticals/horizon alone, at the given longitude: {t:%Y-%m-%d %H:%M:%S} UTC "
                     f"+-{sig_t / 60:.0f} min (or any whole number of sidereal days, 23 h 56 min 4 s, earlier "
                     "or later). Latitude implied: {:.2f}.".format(la))
        if lat is not None and abs(la - lat) > 3 * zen.sigma_deg + 0.5:
            notes.append(f"The latitude implied by the verticals ({la:.2f}) disagrees with the given latitude "
                         f"({lat:.2f}): check the marked lines.")
    else:
        ref = window or datetime.now().astimezone()
        ep = Epoch(ref)
        zi = _zenith_itrs(zen.u_wcs[None, :], ep, 0.0)[0]
        la = float(np.degrees(np.arctan2(zi[2], np.hypot(zi[0], zi[1]))))
        out.update(lat=la)
        notes.append(f"From the marked verticals/horizon alone: latitude {la:.2f} +- {zen.sigma_deg:.2f} deg. "
                     "Longitude needs the time (each hour later moves it 15 deg west).")
    return out, notes
