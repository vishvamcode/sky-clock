"""Plate solving: local astrometry.net (preferred), nova.astrometry.net API, or a supplied WCS.

The local path feeds astrometry.net *our own* star list (x, y in this program's pixel convention),
so the returned WCS is unambiguous: ``wcs.all_pix2world(x, y, 0)`` with 0-based x right, y down.
No image pixels or EXIF leave the machine.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import warnings

warnings.filterwarnings("ignore", category=FITSFixedWarning)

from .geometry import radec_to_unit, unit_to_radec


@dataclass
class PlateSolution:
    wcs: WCS
    shape: tuple  # (H, W)
    solver: str
    rms_px: float | None = None
    n_match: int | None = None

    def pix2unit(self, x, y):
        ra, dec = self.wcs.all_pix2world(np.asarray(x, float), np.asarray(y, float), 0)
        return radec_to_unit(ra, dec)

    def unit2pix(self, u):
        ra, dec = unit_to_radec(u)
        x, y = self.wcs.all_world2pix(ra, dec, 0, quiet=True)
        return np.asarray(x), np.asarray(y)

    def pixel_scale_rad(self, x=None, y=None):
        H, W = self.shape
        x = W / 2 if x is None else x
        y = H / 2 if y is None else y
        a = self.pix2unit([x, x + 1, x], [y, y, y + 1])
        return float(0.5 * (np.arccos(np.clip(a[0] @ a[1], -1, 1)) + np.arccos(np.clip(a[0] @ a[2], -1, 1))))

    def summary(self):
        H, W = self.shape
        c = self.pix2unit([W / 2], [H / 2])[0]
        ra, dec = unit_to_radec(c)
        corners = self.pix2unit([0, W - 1], [H / 2, H / 2])
        fov = np.degrees(np.arccos(np.clip(corners[0] @ corners[1], -1, 1)))
        return dict(solver=self.solver, ra_center=float(ra), dec_center=float(dec),
                    pixel_scale_arcmin=float(np.degrees(self.pixel_scale_rad()) * 60), fov_x_deg=float(fov),
                    rms_px=self.rms_px, n_match=self.n_match)

    def header_text(self):
        return self.wcs.to_header(relax=True).tostring(sep="\n")


def local_solver_available() -> bool:
    return shutil.which("solve-field") is not None


def zenith_radec(when, lat_deg, lon_deg):
    """RA/Dec (deg) of the local zenith: a good plate-solving hint for sky photos (the camera points
    somewhere above the horizon, i.e. within 90 deg of it)."""
    from astropy.coordinates import EarthLocation, SkyCoord, AltAz
    from astropy.time import Time
    import astropy.units as u
    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg)
    z = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=AltAz(obstime=Time(when), location=loc)).icrs
    return float(z.ra.deg), float(z.dec.deg)


def solve_local(stars, shape, scale_low_deg=3.0, scale_high_deg=180.0, cpulimit=120, max_stars=200,
                tweak_order=None, hint=None) -> PlateSolution | None:
    """Solve from a star list [(x, y, flux), ...] with a local astrometry.net install.

    ``hint`` = (ra_deg, dec_deg, radius_deg) limits the search to that part of the sky, which makes wide
    fields (where astrometry.net is slow) solve in seconds instead of minutes.
    ``tweak_order`` None picks 2 for normal lenses and 4 for wide/fisheye fields (more distortion)."""
    if not local_solver_available():
        raise RuntimeError("solve-field not found: install astrometry.net and wide-field index files "
                           "(see README) or use the nova.astrometry.net API key option")
    H, W = shape
    st = sorted(stars, key=lambda s: -s.flux)[:max_stars]
    if len(st) < 8:
        return None
    with tempfile.TemporaryDirectory(prefix="skyclock_solve_") as wd:
        wd = Path(wd)
        cols = [fits.Column(name="X", format="D", array=np.array([s.x + 1 for s in st])),
                fits.Column(name="Y", format="D", array=np.array([s.y + 1 for s in st])),
                fits.Column(name="FLUX", format="D", array=np.array([s.flux for s in st]))]
        fits.BinTableHDU.from_columns(cols).writeto(wd / "xy.fits")
        cmd = ["solve-field", str(wd / "xy.fits"), "--width", str(W), "--height", str(H),
               "--x-column", "X", "--y-column", "Y", "--sort-column", "FLUX",
               "--scale-units", "degwidth", "--scale-low", str(scale_low_deg), "--scale-high", str(scale_high_deg),
               "--no-plots", "--overwrite", "--dir", str(wd), "--cpulimit", str(cpulimit),
               "--tweak-order", str(tweak_order or (4 if scale_low_deg >= 40 else 3)), "--corr", str(wd / "out.corr"), "--wcs", str(wd / "out.wcs"),
               "--new-fits", "none", "--rdls", "none", "--index-xyls", "none", "--match", "none",
               "--solved", str(wd / "out.solved"), "--temp-axy", "--no-remove-lines", "--uniformize", "0"]
        if hint is not None:
            cmd += ["--ra", f"{hint[0]:.4f}", "--dec", f"{hint[1]:.4f}", "--radius", f"{hint[2]:.2f}"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=cpulimit + 60, check=False)
        except subprocess.TimeoutExpired:
            return None
        if not (wd / "out.wcs").exists():
            return None
        hdr = fits.getheader(wd / "out.wcs")
        wcs = WCS(hdr)
        rms, nm = _corr_rms(wcs, wd / "out.corr", flip_h=None)
        return PlateSolution(wcs, (H, W), "astrometry.net (local)", rms, nm)


SCALE_BANDS = [(40.0, 180.0), (12.0, 45.0), (3.0, 14.0)]  # field width (deg): wide/fisheye, normal, tele


def solve_local_auto(stars, shape, hint=None, fov_deg=None, progress=None) -> PlateSolution | None:
    """Try the most likely field widths first, each with a short time limit.

    astrometry.net is slow on very wide fields when the scale range is open (minutes); split into bands,
    a 97-degree field solves in a few seconds. ``fov_deg``: horizontal field width if known (EXIF focal
    length); ``hint``: (ra, dec, radius) from a claimed time and place."""
    bands = []
    if fov_deg:
        bands.append((0.8 * fov_deg, 1.25 * fov_deg))
    bands += SCALE_BANDS
    tries = [(b, hint) for b in bands]
    if hint is not None:  # the hint may be wrong (wrong time/place claim): retry without it
        tries += [(b, None) for b in bands]
    for k, (band, h) in enumerate(tries):
        if progress:
            progress(k / len(tries), f"plate solving: field {band[0]:.0f}-{band[1]:.0f} deg"
                     + (" near the zenith" if h else ""))
        sol = solve_local(stars, shape, band[0], band[1], cpulimit=25 if h else 45, hint=h)
        if sol is not None:
            return sol
    return None


def _corr_rms(wcs, corr_path, flip_h=None):
    try:
        t = fits.getdata(corr_path, 1)
        fx, fy = t["field_x"] - 1, t["field_y"] - 1
        if flip_h is not None:
            fy = flip_h - 1 - fy
        px, py = wcs.all_world2pix(t["index_ra"], t["index_dec"], 0, quiet=True)
        r = np.hypot(px - fx, py - fy)
        return float(np.sqrt(np.mean(r**2))), int(len(r))
    except Exception:
        return None, None


def from_wcs(src, shape) -> PlateSolution:
    """Load a WCS from a FITS header file / bytes (e.g. downloaded from nova, or simulator output)."""
    if isinstance(src, (bytes, bytearray)):
        hdr = fits.Header.fromfile(io.BytesIO(src)) if not src[:6] == b"SIMPLE" else fits.getheader(io.BytesIO(src))
    else:
        hdr = fits.getheader(src)
    return PlateSolution(WCS(hdr), tuple(shape), "supplied WCS")


def from_wcs_object(wcs, shape, solver="supplied WCS") -> PlateSolution:
    return PlateSolution(wcs, tuple(shape), solver)


# ------------------------------------------------------------------ nova.astrometry.net
NOVA = "https://nova.astrometry.net"


def solve_nova(api_key: str, gray_display_png: bytes, stars, shape, scale_low_deg=3.0, scale_high_deg=180.0,
               timeout_s=600, progress=None) -> PlateSolution | None:
    """Upload an EXIF-free grayscale PNG (made by this program) to nova.astrometry.net, private.

    The returned WCS is checked against our own star list, and its y-axis flipped if needed.
    """
    import requests
    s = requests.Session()
    r = s.post(f"{NOVA}/api/login", data={"request-json": json.dumps({"apikey": api_key})}, timeout=60).json()
    if r.get("status") != "success":
        raise RuntimeError(f"nova login failed: {r}")
    sess = r["session"]
    req = {"session": sess, "publicly_visible": "n", "allow_modifications": "n", "allow_commercial_use": "n",
           "scale_units": "degwidth", "scale_type": "ul", "scale_lower": scale_low_deg, "scale_upper": scale_high_deg,
           "tweak_order": 3}
    files = {"file": ("frame.png", gray_display_png, "image/png")}
    r = s.post(f"{NOVA}/api/upload", data={"request-json": json.dumps(req)}, files=files, timeout=300).json()
    if r.get("status") != "success":
        raise RuntimeError(f"nova upload failed: {r}")
    subid = r["subid"]
    t0 = time.time()
    job = None
    while time.time() - t0 < timeout_s:
        time.sleep(5)
        sub = s.get(f"{NOVA}/api/submissions/{subid}", timeout=60).json()
        jobs = [j for j in sub.get("jobs", []) if j]
        if jobs:
            job = jobs[0]
            st = s.get(f"{NOVA}/api/jobs/{job}", timeout=60).json().get("status")
            if progress:
                progress(f"nova job {job}: {st}")
            if st == "success":
                break
            if st == "failure":
                return None
    else:
        return None
    wcs_bytes = s.get(f"{NOVA}/wcs_file/{job}", timeout=120).content
    corr_bytes = s.get(f"{NOVA}/corr_file/{job}", timeout=120).content
    wcs = WCS(fits.getheader(io.BytesIO(wcs_bytes)))
    H, W = shape
    with tempfile.NamedTemporaryFile(suffix=".corr") as tf:
        tf.write(corr_bytes)
        tf.flush()
        best = None
        t = fits.getdata(tf.name, 1)
        ours = np.array([[st.x, st.y] for st in stars]) if stars else None
        for flip in (None, H):
            fx = t["field_x"] - 1
            fy = t["field_y"] - 1 if flip is None else H - t["field_y"]
            if ours is not None and len(ours):
                d = np.min(np.hypot(ours[:, None, 0] - fx[None, :], ours[:, None, 1] - fy[None, :]), axis=0)
                score = float(np.median(d))
            else:
                score = 0.0 if flip is None else 1.0
            if best is None or score < best[0]:
                best = (score, flip)
        flip = best[1]
    if flip is not None:
        # Re-express WCS for y_ours = H-1-y_nova: build by fitting via pixel transformation
        wcs = _flip_wcs_y(wcs, H)
    rms, nm = _corr_rms_bytes(wcs, corr_bytes, flip)
    return PlateSolution(wcs, (H, W), "nova.astrometry.net", rms, nm)


def _corr_rms_bytes(wcs, corr_bytes, flip):
    with tempfile.NamedTemporaryFile(suffix=".corr") as tf:
        tf.write(corr_bytes)
        tf.flush()
        return _corr_rms(wcs, tf.name, flip_h=flip)


def _flip_wcs_y(wcs, H):
    """Return a WCS w' with w'(x, y) = w(x, H-1-y) (0-based), refitting a TAN-SIP model."""
    from astropy.wcs.utils import fit_wcs_from_points
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    W = int(wcs.pixel_shape[0]) if wcs.pixel_shape else int(2 * wcs.wcs.crpix[0])
    xs, ys = np.meshgrid(np.linspace(0, W - 1, 25), np.linspace(0, H - 1, 25))
    ra, dec = wcs.all_pix2world(xs.ravel(), (H - 1 - ys).ravel(), 0)
    sc = SkyCoord(ra * u.deg, dec * u.deg)
    return fit_wcs_from_points((xs.ravel(), ys.ravel()), sc, projection="TAN", sip_degree=3)
