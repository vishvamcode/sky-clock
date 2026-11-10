"""Synthetic night-sky photos with satellite streaks and known truth (v0 simulator).

Uses the same propagation and frame code as the solver, so it validates the pipeline's internal
consistency and the solver's statistics. It does not validate the physics against the real sky:
for that, use real photos with ground truth (see README, "Validation protocol").
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from astropy.wcs import WCS

from .catalog import Catalog
from .geometry import (Epoch, earth_beta, enu_basis, geo_to_wcs, geodetic_to_itrs, radec_to_unit, sun_gcrs,
                       sunlit, unit, unit_to_radec)


@dataclass
class Camera:
    width: int = 3000
    height: int = 2000
    fov_x_deg: float = 60.0
    psf_sigma_px: float = 1.1
    roll_deg: float = 0.0

    @property
    def scale_deg(self):
        return self.fov_x_deg / self.width


def make_wcs(ra_deg, dec_deg, cam: Camera) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra_deg, dec_deg]
    w.wcs.crpix = [(cam.width + 1) / 2, (cam.height + 1) / 2]
    r = np.radians(cam.roll_deg)
    s = cam.scale_deg
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    w.wcs.cd = -s * rot  # numpy rows grow downward: this keeps the natural (non-mirrored) sky parity
    return w


def pointing_radec(epoch: Epoch, lat, lon, alt_deg, az_deg, sec=0.0):
    e, n, u = enu_basis(lat, lon)
    a, z = np.radians(alt_deg), np.radians(az_deg)
    d_itrs = np.cos(a) * (np.sin(z) * e + np.cos(z) * n) + np.sin(a) * u
    g = epoch.itrs_to_gcrs_vecs(d_itrs, sec)
    w = geo_to_wcs(g, earth_beta(epoch))
    ra, dec = unit_to_radec(w)
    return float(ra), float(dec)


def real_star_field(ra_deg, dec_deg, radius_deg, mag_limit=7.0, index_dir="/usr/share/astrometry"):
    """Tycho-2 stars (unit vectors, V mags) around a direction, read from astrometry.net index files
    with ``query-starkd``. Raises RuntimeError if astrometry.net or the index files are missing."""
    import glob
    import shutil
    import subprocess
    import tempfile
    from astropy.io import fits
    if shutil.which("query-starkd") is None:
        raise RuntimeError("query-starkd (astrometry.net) not installed")
    files = sorted(glob.glob(f"{index_dir}/index-tycho2-1[0-9]*.fits") + glob.glob(f"{index_dir}/index-41[0-9][0-9]*.fits"))
    if not files:
        raise RuntimeError(f"no index files in {index_dir}")
    us, ms = [], []
    with tempfile.TemporaryDirectory() as td:
        for k, f in enumerate(files):
            out = f"{td}/q{k}.fits"
            subprocess.run(["query-starkd", "-r", str(ra_deg), "-d", str(dec_deg), "-R", str(radius_deg), "-T",
                            "-o", out, f], capture_output=True, check=False)
            try:
                t = fits.getdata(out, 1)
            except Exception:
                continue
            mag = t["MAG_VT"] if "MAG_VT" in t.columns.names else np.full(len(t), 8.0)
            us.append(radec_to_unit(t["RA"], t["DEC"]))
            ms.append(np.asarray(mag, float))
    if not us:
        raise RuntimeError("no stars returned")
    u = np.concatenate(us)
    m = np.concatenate(ms)
    # de-duplicate stars present in several index scales
    key = np.round(u * 2e4).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    u, m = u[first], m[first]
    keep = m <= mag_limit
    return u[keep], m[keep]


def _splat(img, xs, ys, flux, sigma):
    H, W = img.shape
    r = int(np.ceil(3 * sigma))
    ix, iy = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - ix, ys - iy
    for dy in range(-r, r + 2):
        for dx in range(-r, r + 2):
            px, py = ix + dx, iy + dy
            wgt = np.exp(-((dx - fx) ** 2 + (dy - fy) ** 2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            np.add.at(img, (py[ok], px[ok]), (flux * wgt)[ok])


def simulate_photo(catalog: Catalog, when: datetime, lat: float, lon: float, h_km: float = 0.0,
                   exposure_s: float = 10.0, alt_deg: float = 55.0, az_deg: float = 200.0,
                   cam: Camera | None = None, n_stars: int = 1500, star_mag_limit: float = 6.5,
                   sat_mag_range=(2.0, 4.5), sky_level: float = 200.0, read_noise: float = 4.0,
                   require_sunlit: bool = True, dashed_gap: tuple | None = None, seed: int = 0,
                   max_sats: int | None = None, tracking: bool = False, real_stars: bool = False):
    """Render a photo. ``dashed_gap=(on_s, off_s)`` mimics stacked night modes.

    ``tracking=False`` (default) simulates a fixed tripod: the camera is Earth-fixed, stars trail
    slightly and the WCS describes the frame at mid-exposure. ``real_stars=True`` uses Tycho-2 stars
    from the installed astrometry.net index files (needed to test plate solving end to end).

    Returns (image float32, wcs, truth dict).
    """
    cam = cam or Camera()
    rng = np.random.default_rng(seed)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    ep = Epoch(when)
    beta = earth_beta(ep)
    t_m = exposure_s / 2  # the WCS describes the frame at mid-exposure
    ra0, dec0 = pointing_radec(ep, lat, lon, alt_deg, az_deg, t_m)
    wcs = make_wcs(ra0, dec0, cam)
    img = np.full((cam.height, cam.width), sky_level, dtype=np.float64)
    # gentle sky gradient (light pollution)
    yy = np.linspace(0, 1, cam.height)[:, None]
    img += 0.3 * sky_level * yy

    def to_cam(d_gcrs, tau):
        """Sky direction at time tau -> direction in the camera's (WCS) frame."""
        if tracking:
            return d_gcrs
        return ep.itrs_to_gcrs_vecs(ep.gcrs_to_itrs_vecs(d_gcrs, tau), np.full(np.shape(tau), t_m))

    f0 = 4e6
    if real_stars:
        star_u, mags = real_star_field(ra0, dec0, 0.75 * np.hypot(cam.fov_x_deg, cam.fov_x_deg * cam.height / cam.width),
                                       star_mag_limit)
    else:
        u = rng.uniform(size=n_stars)
        mags = star_mag_limit + 2.0 * np.log10(np.maximum(u, 1e-6))
        xs = rng.uniform(-50, cam.width + 50, n_stars)
        ys = rng.uniform(-50, cam.height + 50, n_stars)
        sra, sdec = wcs.all_pix2world(xs, ys, 0)
        star_u = radec_to_unit(sra, sdec)
    n_sub = 9
    for tau in np.linspace(0, exposure_s, n_sub):  # stars trail on a fixed tripod
        su = to_cam(star_u, np.full(len(star_u), tau))
        sra, sdec = unit_to_radec(su)
        sx, sy = wcs.all_world2pix(sra, sdec, 0, quiet=True)
        front = su @ radec_to_unit(ra0, dec0) > 0.3
        ok = front & (sx > -10) & (sx < cam.width + 10) & (sy > -10) & (sy < cam.height + 10)
        _splat(img, sx[ok], sy[ok], f0 * 10 ** (-0.4 * mags[ok]) * exposure_s / 10.0 / n_sub, cam.psf_sigma_px)

    o = geodetic_to_itrs(lat, lon, h_km)
    e_, n_, up = enu_basis(lat, lon)
    sun_i = ep.gcrs_to_itrs_vecs(sun_gcrs(ep), 0.0)
    coarse = np.linspace(0, exposure_s, 21)
    r, v, ok = catalog.propagate(ep, coarse)
    truth_streaks = []
    cand = []
    a_, z_ = np.radians(alt_deg), np.radians(az_deg)
    centre = geo_to_wcs(ep.itrs_to_gcrs_vecs(np.cos(a_) * (np.sin(z_) * e_ + np.cos(z_) * n_) + np.sin(a_) * up, t_m), beta)
    allok = ok.all(axis=1)
    d = r - o
    rho = np.linalg.norm(d, axis=2)
    el = np.degrees(np.arcsin(np.clip(np.einsum("ntk,k->nt", d, up) / rho, -1, 1)))
    good = allok & (np.nanmax(np.where(np.isfinite(el), el, -90), axis=1) > 5)
    if require_sunlit:
        good &= sunlit(np.nan_to_num(r[:, 10]), sun_i)
    half_diag = np.radians(0.5 * np.hypot(cam.fov_x_deg, cam.fov_x_deg * cam.height / cam.width) + 3)
    idx = np.where(good)[0]
    if len(idx):
        g = ep.itrs_to_gcrs_vecs(d[idx] / rho[idx][..., None], coarse)
        wv = geo_to_wcs(to_cam(g, np.broadcast_to(coarse, g.shape[:-1])), beta)
        near = (wv @ centre > np.cos(half_diag)).any(axis=1)
        for jj, j in enumerate(idx):
            if not near[jj]:
                continue
            ra, dec = unit_to_radec(wv[jj])
            px, py = wcs.all_world2pix(ra, dec, 0)
            inside = (px > -5) & (px < cam.width + 5) & (py > -5) & (py < cam.height + 5)
            if np.any(inside & ((wv[jj] @ centre) > 0.2)):
                cand.append((int(j), float(rho[j].mean())))
    if max_sats is not None:
        cand = cand[:max_sats]
    for j, rho_m in cand:
        # dense render
        r2, _, _ = catalog.propagate(ep, coarse, idx=[j])
        g = ep.itrs_to_gcrs_vecs(unit(r2[0] - o), coarse)
        ra, dec = unit_to_radec(geo_to_wcs(to_cam(g, coarse), beta))
        px, py = wcs.all_world2pix(ra, dec, 0)
        L = np.hypot(px[-1] - px[0], py[-1] - py[0])
        nfine = int(max(200, L / 0.25))
        tf = np.linspace(0, exposure_s, nfine)
        rf, _, _ = catalog.propagate(ep, tf, idx=[j])
        gf = ep.itrs_to_gcrs_vecs(unit(rf[0] - o), tf)
        ra, dec = unit_to_radec(geo_to_wcs(to_cam(gf, tf), beta))
        fx, fy = wcs.all_world2pix(ra, dec, 0)
        m = rng.uniform(*sat_mag_range)
        flux_total = f0 * 10 ** (-0.4 * m) * exposure_s / 10.0 * 3.0
        w_per = np.full(nfine, flux_total / nfine)
        if dashed_gap:
            on_s, off_s = dashed_gap
            phase = tf % (on_s + off_s)
            w_per = np.where(phase < on_s, w_per, 0.0)
        _splat(img, fx, fy, w_per, cam.psf_sigma_px)
        truth_streaks.append(dict(norad=int(catalog.norad[j]), name=catalog.names[j],
                                  start_px=[float(fx[0]), float(fy[0])], end_px=[float(fx[-1]), float(fy[-1])],
                                  mag=float(m), range_km=rho_m))
    noisy = rng.poisson(np.clip(img, 0, None)).astype(np.float64) + rng.normal(0, read_noise, img.shape)
    noisy = np.clip(noisy, 0, 65535).astype(np.float32)
    truth = dict(time_utc=when.isoformat(), exposure_s=exposure_s, lat=lat, lon=lon, h_km=h_km,
                 pointing=dict(alt=alt_deg, az=az_deg, ra=ra0, dec=dec0), camera=cam.__dict__, tracking=tracking,
                 streaks=truth_streaks, catalog_source=catalog.source)
    return noisy, wcs, truth


def save_outputs(prefix, image, wcs, truth):
    """Write <prefix>.png (16-bit), <prefix>.wcs (FITS header) and <prefix>.truth.json."""
    from PIL import Image
    arr = np.clip(image, 0, 65535).astype(np.uint16)
    Image.fromarray(arr).save(f"{prefix}.png")
    wcs.to_fits().writeto(f"{prefix}.wcs", overwrite=True)
    with open(f"{prefix}.truth.json", "w") as f:
        json.dump(truth, f, indent=1)
    return f"{prefix}.png", f"{prefix}.wcs", f"{prefix}.truth.json"
