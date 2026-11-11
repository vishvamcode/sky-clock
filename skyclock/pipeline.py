"""High-level glue shared by the command line and the web app."""
from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import numpy as np

from . import catalog as cat_mod
from .detect import detect
from .imageio import exif_hints, load_image
from .observations import build as build_obs
from .platesolve import from_wcs, local_solver_available, solve_local_auto, solve_nova, zenith_radec
from .solve import Settings, analyze

SYNTHETIC_EPOCH = datetime(2026, 11, 1, tzinfo=timezone.utc)

CATALOG_CHOICES = {
    "celestrak": "CelesTrak current GP (all active) + Starlink SupGP: good for photos from the last ~2 days",
    "celestrak-starlink": "CelesTrak current GP, Starlink only (faster)",
    "github-archive": "Public GitHub archive of CelesTrak snapshots (no account; 2021-2023 all satellites, "
                      "Oct 2026 onward Starlink only)",
    "spacetrack": "Space-Track GP history around the photo time (free account; best for older photos)",
    "file": "Your own TLE / OMM file",
    "synthetic": "Synthetic Starlink-like constellation (only for simulated demo photos)",
}


def load_catalog(kind, when=None, file_bytes=None, file_name="upload.txt", min_rev_per_day=11.0,
                 spacetrack_user=None, spacetrack_password=None):
    if kind == "celestrak":
        c = cat_mod.fetch_default(prefer_supgp=True)
    elif kind == "celestrak-starlink":
        c = cat_mod.fetch_celestrak("starlink")
        try:
            c = cat_mod.merge(c, cat_mod.fetch_supgp("starlink"))
        except Exception:
            pass
    elif kind == "spacetrack":
        user = spacetrack_user or os.environ.get("SPACETRACK_USER")
        pw = spacetrack_password or os.environ.get("SPACETRACK_PASSWORD")
        if not (user and pw):
            raise RuntimeError("Space-Track needs SPACETRACK_USER and SPACETRACK_PASSWORD (environment) or the "
                               "credentials fields in the app")
        if when is None:
            raise RuntimeError("Space-Track history needs an approximate date (time, or search window)")
        c = cat_mod.fetch_spacetrack_history(user, pw, when)
    elif kind == "github-archive":
        if when is None:
            raise RuntimeError("the GitHub archive needs an approximate date (time, or search window)")
        c = cat_mod.fetch_github_archive(when)
    elif kind == "file":
        if file_bytes is None:
            raise RuntimeError("no catalogue file given")
        c = cat_mod.load_text(file_bytes.decode("utf-8", errors="replace"), source=f"file:{file_name}")
    elif kind == "synthetic":
        c = cat_mod.synthetic_constellation(SYNTHETIC_EPOCH)
    else:
        raise ValueError(f"unknown catalogue {kind}")
    return c.leo_only(min_rev_per_day) if min_rev_per_day else c


def gray_png_for_upload(gray):
    """EXIF-free 8-bit PNG of the image (what is sent to nova.astrometry.net, if used)."""
    from PIL import Image
    from .imageio import to_display
    b = io.BytesIO()
    Image.fromarray(to_display(gray)[..., 0]).save(b, format="PNG")
    return b.getvalue()


def plate_hint(settings):
    """(ra, dec, radius) around the zenith, when the time (to ~2 h) and place are roughly known."""
    if settings is None or settings.lat is None or settings.lon is None:
        return None
    if settings.time is not None and (settings.sigma_time_s or 0) <= 7200:
        when = settings.time
    elif settings.window_start is not None and settings.window_end is not None and \
            (settings.window_end - settings.window_start).total_seconds() <= 4 * 3600:
        when = settings.window_start + (settings.window_end - settings.window_start) / 2
    else:
        return None
    ra, dec = zenith_radec(when, settings.lat, settings.lon)
    return (ra, dec, 95.0)


def plate_solve(det, gray, method="local", nova_key=None, wcs_bytes=None, progress=None,
                scale_low_deg=3.0, scale_high_deg=180.0, hint=None, fov_deg=None):
    shape = gray.shape
    if method == "wcs":
        if wcs_bytes is None:
            raise RuntimeError("no WCS file given")
        return from_wcs(wcs_bytes, shape)
    if method == "local":
        if not local_solver_available():
            raise RuntimeError("astrometry.net is not installed here; use Docker, the nova API key, or a WCS file")
        return solve_local_auto(det.stars, shape, hint=hint, fov_deg=fov_deg, progress=progress)
    if method == "nova":
        key = nova_key or os.environ.get("ASTROMETRY_NET_API_KEY")
        if not key:
            raise RuntimeError("nova.astrometry.net needs an API key (ASTROMETRY_NET_API_KEY)")
        return solve_nova(key, gray_png_for_upload(gray), det.stars, shape, scale_low_deg, scale_high_deg,
                          progress=progress)
    raise ValueError(method)


def run_all(image_bytes, image_name, settings: Settings, catalog, plate_method="local", nova_key=None,
            wcs_bytes=None, streak_edits=None, progress=None):
    """Load -> detect -> plate solve -> measure -> analyze. Returns a dict of everything."""
    gray, display, exif = load_image(image_bytes, image_name)
    det = detect(gray)
    if streak_edits:
        for st in det.streaks:
            if st.id in streak_edits:
                st.enabled = bool(streak_edits[st.id])
    plate = plate_solve(det, gray, plate_method, nova_key, wcs_bytes, progress, hint=plate_hint(settings),
                        fov_deg=exif_hints(exif).get("fov_x_deg"))
    if plate is None:
        raise RuntimeError("plate solving failed: not enough stars matched")
    callable_window = False
    if callable(catalog):  # loader: lets the Moon/planet date choose the catalogue epoch first
        when = settings.time or settings.window_start
        if when is None and settings.mode != "date" and (settings.body_marks or settings.auto_bodies):
            from dataclasses import replace
            from .solve import date_from_bodies, window_from_dates
            dates, _ = date_from_bodies(plate, replace(settings, auto_bodies=False), det)
            if not dates or settings.mode not in ("time", "blind"):
                raise RuntimeError("no time given: use mode 'date', or 'time'/'blind' with a marked Moon/planet")
            settings = window_from_dates(settings, dates)
            when = settings.window_start
            callable_window = True
        catalog = catalog(when) if settings.mode != "date" else None
    epoch = settings.epoch()
    obs = build_obs(det.streaks, plate, epoch)
    res = analyze(obs, plate, catalog, settings, progress, det=det)
    if callable_window:
        res.notes.insert(0, f"Satellite search window set from the Moon/planet date: "
                            f"{settings.window_start:%Y-%m-%d %H:%M} to {settings.window_end:%H:%M} UTC.")
    return dict(gray=gray, display=display, exif=exif, hints=exif_hints(exif), detection=det, plate=plate,
                observations=obs, result=res)


def parse_time(s):
    if s is None or s == "":
        return None
    if isinstance(s, datetime):
        d = s
    else:
        d = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def nice_float(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)
