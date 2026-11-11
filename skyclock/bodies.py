"""The Moon and planets as a calendar: which date (and hour) puts them where the photo shows them.

Stars do not move against each other, but the Moon and the planets do. Once the plate solution says
where in the sky a bright object in the photo is, the ephemeris says when that object was there:

* Moon: about 0.55 deg per hour against the stars. Its position also depends on where on Earth you stand
  (parallax up to about 1 deg), so with the place known it dates a photo to minutes; without it, to
  about +-2 hours. Its pattern almost repeats every 27.3 days.
* Venus (about 1 deg/day), Mercury, Mars (about 0.5 deg/day): date to hours.
* Jupiter and Saturn (0.03 to 0.2 deg/day): date to days; their paths cross themselves in retrograde loops,
  so they give 1 to 3 dates per year, repeating every 12 and 29 years.

Two or more bodies in one photo pin the date far more tightly than either alone.

Positions are astrometric (light-time corrected, no aberration), which is the frame the plate solution's
star positions are in: aberration moves the stars and the planets alike, so it cancels.
Ephemeris: astropy's built-in one (ERFA epv00 / plan94 and a Meeus lunar theory), which works offline and
is good to a few arcseconds for the planets and about 10 arcseconds for the Moon; the photo
measurement (a pixel or two, usually over an arcminute) dominates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.optimize import minimize_scalar

from .geometry import Epoch, geodetic_to_itrs, unit

C_KM_S = 299792.458
BODIES = ("moon", "venus", "jupiter", "mars", "saturn", "mercury")
# sampling step (hours) for the date search: well below the time over which the path curves
STEP_H = dict(moon=0.25, mercury=4.0, venus=6.0, mars=12.0, jupiter=48.0, saturn=72.0)
MOON_MAX_WINDOW_DAYS = 3 * 366  # the Moon's pattern repeats monthly: beyond this, the list is useless
EARTH_R_KM = 6378.0


def _times(ts):
    from astropy.time import Time
    ts = [t if t.tzinfo else t.replace(tzinfo=timezone.utc) for t in np.atleast_1d(ts)]
    return Time([t.astimezone(timezone.utc).replace(tzinfo=None) for t in ts], scale="utc")


def _bary_km(name, T):
    from astropy.coordinates import get_body_barycentric
    return np.atleast_2d(get_body_barycentric(name, T, ephemeris="builtin").xyz.to("km").value.T)


def observer_offset_km(lat, lon, h_km, times, ref=None):
    """Observer position relative to the geocentre, ICRS-aligned axes (km), at each time."""
    times = list(np.atleast_1d(times))
    ep = Epoch(ref or times[0])
    o = geodetic_to_itrs(lat, lon, h_km)
    sec = np.array([ep.seconds_from(t) for t in times])
    return ep.itrs_to_gcrs_vecs(np.broadcast_to(o, (len(sec), 3)), sec)


def body_dirs(name, times, lat=None, lon=None, h_km=0.0):
    """Astrometric unit vectors (ICRS axes) of ``name`` seen from (lat, lon), or from the geocentre if the
    place is not given, plus the distances (km)."""
    times = list(np.atleast_1d(times))
    T = _times(times)
    obs = _bary_km("earth", T)
    if lat is not None and lon is not None:
        obs = obs + observer_offset_km(lat, lon, h_km, times)
    r = _bary_km(name, T)
    for _ in range(2):  # light time: where the body was when the light left it
        tau = np.linalg.norm(r - obs, axis=1) / C_KM_S
        r = _bary_km(name, T - tau * _sec())
    d = r - obs
    dist = np.linalg.norm(d, axis=1)
    return d / dist[:, None], dist


def _sec():
    import astropy.units as u
    return u.s


@dataclass
class BodyMark:
    """A bright object in the photo the user (or the automatic search) says is the Moon or a planet."""
    x: float
    y: float
    name: str | None = None  # None: try every planet
    sigma_px: float = 1.5


@dataclass
class DateCandidate:
    body: str
    time: datetime
    sigma_t_s: float
    miss_sigma: float  # closest approach between the predicted and the measured position, in sigma
    rate_deg_per_day: float
    notes: list = field(default_factory=list)

    def to_dict(self):
        return dict(body=self.body, time_utc=self.time.isoformat(), sigma_t_s=self.sigma_t_s,
                    miss_sigma=self.miss_sigma, rate_deg_per_day=self.rate_deg_per_day, notes=self.notes)


def _sigma_rad(mark: BodyMark, plate, name, place_known):
    s = mark.sigma_px * plate.pixel_scale_rad(mark.x, mark.y)
    # ephemeris floor: measured against JPL DE430 (skyfield) on 2015-03-02: Moon, Mercury, Venus, Mars
    # within 2 arcsec, Saturn 17, Jupiter 39 (astropy's built-in plan94 series); plate model ~10 arcsec
    floor = dict(jupiter=60.0, saturn=30.0).get(name, 10.0)
    s = float(np.hypot(s, np.radians(floor / 3600)))
    if not place_known:
        # parallax from an unknown place on Earth: up to R_earth / distance (about 1 deg for the Moon)
        typical = dict(moon=384400.0, venus=1.0e8, mars=1.5e8, mercury=1.2e8, jupiter=7.0e8, saturn=1.4e9)[name]
        s = float(np.hypot(s, EARTH_R_KM / typical / np.sqrt(2)))
    return s


def date_candidates(mark: BodyMark, plate, start: datetime, end: datetime, lat=None, lon=None, h_km=0.0,
                    max_miss_sigma=3.0, names=None):
    """Every time in [start, end] when a body was where the mark is, best first."""
    u_meas = plate.pix2unit([mark.x], [mark.y])[0]
    names = [mark.name] if mark.name else list(names or BODIES)
    out = []
    for name in names:
        t0, t1 = start, end
        notes = []
        if name == "moon" and (t1 - t0).days > MOON_MAX_WINDOW_DAYS:
            t1 = t0 + timedelta(days=MOON_MAX_WINDOW_DAYS)
            notes.append("Moon searched over the first 3 years of the window only (its pattern repeats monthly).")
        place = lat is not None and lon is not None
        sig = _sigma_rad(mark, plate, name, place)
        step = timedelta(hours=STEP_H[name])
        n = int((t1 - t0) / step) + 2
        grid = [t0 + k * step for k in range(n)]
        sep = np.full(n, np.inf)
        for a in range(0, n, 20000):  # chunks keep memory small
            d, _ = body_dirs(name, grid[a:a + 20000], lat, lon, h_km)
            sep[a:a + 20000] = np.arccos(np.clip(d @ u_meas, -1, 1))
        # local minima of the separation that come anywhere near the mark
        tol = max(max_miss_sigma * sig, 0.0)
        reach = np.radians(3.0) if name == "moon" else np.radians(1.0)
        idx = [i for i in range(n) if sep[i] < tol + reach
               and (i == 0 or sep[i] <= sep[i - 1]) and (i == n - 1 or sep[i] <= sep[i + 1])]
        for i in idx:
            ta = grid[max(i - 1, 0)]
            span = (grid[min(i + 1, n - 1)] - ta).total_seconds()

            def f(x):
                d, _ = body_dirs(name, [ta + timedelta(seconds=float(x))], lat, lon, h_km)
                return float(np.arccos(np.clip(d[0] @ u_meas, -1, 1)))

            r = minimize_scalar(f, bounds=(0.0, span), method="bounded", options=dict(xatol=1.0))
            tm = ta + timedelta(seconds=float(r.x))
            miss = r.fun / sig
            if miss > max_miss_sigma:
                continue
            h = 3600.0
            d2, _ = body_dirs(name, [tm - timedelta(seconds=h), tm + timedelta(seconds=h)], lat, lon, h_km)
            rate = float(np.arccos(np.clip(d2[0] @ d2[1], -1, 1)) / (2 * h))  # rad/s
            sig_t = sig / max(rate, 1e-14)
            cn = list(notes)
            if rate * 86400 < np.radians(0.02):
                cn.append(f"{name.title()} was nearly stationary (turning in its retrograde loop): the date is "
                          "poorly fixed.")
            out.append(DateCandidate(name, tm, float(min(sig_t, 3.2e7)), float(miss),
                                     float(np.degrees(rate) * 86400), cn))
    out.sort(key=lambda c: (c.miss_sigma > 1.5, c.sigma_t_s))
    return out


def combine(cands_by_mark):
    """Dates on which every marked body agrees. ``cands_by_mark`` is a list (one per mark) of candidate
    lists. Returns (time, sigma_t_s, chi2, [candidates]) tuples, best first."""
    lists = [c for c in cands_by_mark if c]
    if not lists:
        return []
    if len(lists) == 1:
        return [(c.time, c.sigma_t_s, c.miss_sigma ** 2, [c]) for c in lists[0]]
    anchor = min(lists, key=lambda L: min(c.sigma_t_s for c in L))
    out = []
    for a in anchor:
        chosen, chi2 = [a], a.miss_sigma ** 2
        ok = True
        for L in lists:
            if L is anchor:
                continue
            best = None
            for c in L:
                dt = abs((c.time - a.time).total_seconds())
                s = np.hypot(c.sigma_t_s, a.sigma_t_s)
                z = dt / s
                if z < 3 and (best is None or z < best[0]):
                    best = (z, c)
            if best is None:
                ok = False
                break
            chosen.append(best[1])
            chi2 += best[0] ** 2 + best[1].miss_sigma ** 2
        if not ok:
            continue
        w = np.array([1 / c.sigma_t_s ** 2 for c in chosen])
        ts = np.array([(c.time - a.time).total_seconds() for c in chosen])
        tm = a.time + timedelta(seconds=float(np.sum(w * ts) / np.sum(w)))
        out.append((tm, float(1 / np.sqrt(np.sum(w))), float(chi2), chosen))
    out.sort(key=lambda r: (r[2] > 6 * len(lists), r[1]))
    return out


def bright_unmatched(det, plate, n=12, min_sep_px=6.0, star_check=True, max_star_mag=4.0):
    """The brightest detected point sources that are not known stars: the places to look for a planet.
    A source is a star when the plate solver matched it, or (``star_check``) when a catalogue star brighter
    than ``max_star_mag`` lies on it (see ``brightstars``: the very brightest stars are often missing
    from the solver's index, and a saturated star is often left unmatched)."""
    stars = sorted(det.stars, key=lambda s: -s.flux)[: 4 * n]
    mxy = getattr(plate, "matched_xy", None)
    cand = [s for s in stars
            if not (mxy is not None and len(mxy) and np.min(np.hypot(mxy[:, 0] - s.x, mxy[:, 1] - s.y)) < min_sep_px)]
    if star_check and cand:
        from .brightstars import near_star_mask
        near = near_star_mask(plate, [(s.x, s.y) for s in cand], max_star_mag)
        cand = [s for s, k in zip(cand, near) if not k]
    return cand[:n]


def auto_find(det, plate, start, end, lat=None, lon=None, h_km=0.0, n=8):
    """Try the brightest unmatched sources against every planet and the Moon over the window.
    Returns [(mark, candidates)], strongest first. Sources on a catalogue star are skipped, but every hit
    still needs a look: a satellite flare, an aircraft light or a hot pixel on a planet's path would also hit."""
    hits = []
    for s in bright_unmatched(det, plate, n):
        m = BodyMark(s.x, s.y)
        c = date_candidates(m, plate, start, end, lat, lon, h_km, max_miss_sigma=2.0,
                            names=[b for b in BODIES if b != "moon"])
        if c:
            hits.append((m, c))
    return hits


def summary_notes(marks, cands, combined):
    notes = []
    for m, c in zip(marks, cands):
        label = m.name.title() if m.name else f"Object at ({m.x:.0f}, {m.y:.0f})"
        if not c:
            notes.append(f"{label}: no matching date in the window (wrong body, or outside the window).")
            continue
        top = ", ".join(f"{x.body} {x.time:%Y-%m-%d %H:%M} UTC +-{_fmt_dt(x.sigma_t_s)}" for x in c[:4])
        more = f" and {len(c) - 4} more" if len(c) > 4 else ""
        notes.append(f"{label}: {top}{more}.")
    if len(marks) > 1:
        if combined:
            t, s, _, _ = combined[0]
            notes.append(f"All marked bodies agree on {t:%Y-%m-%d %H:%M} UTC +-{_fmt_dt(s)}"
                         + (f" ({len(combined) - 1} other date(s) also fit)" if len(combined) > 1 else "") + ".")
        else:
            notes.append("The marked bodies agree on no common date: check the identifications.")
    return notes


def _fmt_dt(s):
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    if s < 2 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"
