"""Match streaks to catalogue satellites for a hypothesised observer and time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .fit import SIG_CROSS_KM, camera_dirs, gnomonic
from .geometry import EARTH_ROT, Epoch, enu_basis, geodetic_to_itrs, sun_gcrs, sunlit, unit


@dataclass
class Match:
    obs_index: int
    sat_index: int
    cost: float
    cross_rad: float
    angle_rad: float
    tau_mid: float  # time at which the satellite crosses the arc midpoint (s after epoch)
    range_km: float


def predicted_in_view(catalog, epoch: Epoch, lat, lon, t, plate, h_km=0.0, half_window=None, T=10.0,
                      tracking=False, el_min_deg=0.0, require_sunlit=True, beta=None):
    """Satellites whose predicted track crosses the frame during [t-T/2, t+T/2].

    Returns list of dicts with sat index, pixel track and mean range (used for overlays).
    """
    from .geometry import earth_beta, geo_to_wcs, unit_to_radec
    beta = earth_beta(epoch) if beta is None else beta
    half = T / 2 if half_window is None else half_window
    tau = np.linspace(t - half, t + half, 21)
    o = geodetic_to_itrs(lat, lon, h_km)
    _, _, up = enu_basis(lat, lon)
    r, _, ok = catalog.propagate(epoch, tau)
    rel = r - o
    rho = np.linalg.norm(rel, axis=2)
    el = np.degrees(np.arcsin(np.clip(np.einsum("ntk,k->nt", rel, up) / rho, -1, 1)))
    good = ok.all(axis=1) & (np.nanmax(np.where(np.isfinite(el), el, -90), axis=1) > el_min_deg)
    if require_sunlit:
        sun_i = epoch.gcrs_to_itrs_vecs(sun_gcrs(epoch, t), t)
        good &= sunlit(np.nan_to_num(r[:, 10]), sun_i, margin_km=30)
    H, W = plate.shape
    out = []
    for j in np.where(good)[0]:
        d = camera_dirs(epoch, rel[j] / rho[j][:, None], tau, t, tracking)
        ra, dec = unit_to_radec(geo_to_wcs(d, beta))
        x, y = plate.wcs.all_world2pix(ra, dec, 0, quiet=True)
        x, y = np.asarray(x), np.asarray(y)
        cen = plate.pix2unit([W / 2], [H / 2])[0]
        wv = geo_to_wcs(d, beta)
        front = wv @ cen > 0.1
        # distortion polynomials (SIP) are only valid near the image: require a clean round trip
        with np.errstate(all="ignore"):
            back = plate.pix2unit(x, y)
        front &= np.sum(back * wv, axis=-1) > np.cos(np.radians(0.2))
        inside = front & (x > 0) & (x < W - 1) & (y > 0) & (y < H - 1)
        if inside.any():
            out.append(dict(sat=int(j), name=catalog.names[j], norad=int(catalog.norad[j]), x=x.tolist(), y=y.tolist(),
                            front=front.tolist(), range_km=float(rho[j].mean())))
    return out


def identify(obs_list, catalog, epoch: Epoch, lat, lon, t, sigma_loc_km=20.0, sigma_t=5.0, T_max=30.0, T=None,
             h_km=0.0, tracking=False, require_sunlit=True, el_min_deg=0.0, step=1.0, n_sigma=3.0,
             candidates=None):
    """Assign each streak at most one satellite. Returns (list[Match|None] per obs, info dict).

    ``candidates`` optionally restricts the catalogue indices considered.
    """
    half = n_sigma * sigma_t + (T_max if T is None else T) / 2 + 2.0
    tau = np.arange(t - half, t + half + step, step)
    o = geodetic_to_itrs(lat, lon, h_km)
    _, _, up = enu_basis(lat, lon)
    idx = np.arange(len(catalog)) if candidates is None else np.asarray(candidates, int)
    chunk = max(1, int(2e6 // max(len(tau), 1)))
    cand = {i: [] for i in range(len(obs_list))}
    sun_i = epoch.gcrs_to_itrs_vecs(sun_gcrs(epoch, t), t)
    for c0 in range(0, len(idx), chunk):
        sub = idx[c0:c0 + chunk]
        r, _, ok = catalog.propagate(epoch, tau, idx=sub)
        rel = r - o
        rho = np.linalg.norm(rel, axis=2)
        d_itrs = rel / rho[..., None]
        el = np.einsum("ntk,k->nt", d_itrs, up)
        good = ok.all(axis=1) & (np.nanmax(el, axis=1) > np.sin(np.radians(el_min_deg)))
        if require_sunlit:
            good &= sunlit(np.nan_to_num(r[:, len(tau) // 2]), sun_i, margin_km=50)
        if not good.any():
            continue
        gi = np.where(good)[0]
        d = camera_dirs(epoch, d_itrs[gi], tau[None, :].repeat(len(gi), 0), t, tracking)
        for i, ob in enumerate(obs_list):
            near = (d @ ob.mid) > np.cos(ob.half_len + np.radians(8.0))
            hits = np.where(near.any(axis=1))[0]
            for h in hits:
                k = sub[gi[h]]
                dd = d[h]
                front = dd @ ob.mid > 0.2
                if front.sum() < 3:
                    continue
                xp, yp = gnomonic(dd[front], ob)
                tp = tau[front]
                order = np.argsort(xp)
                xp, yp, tp = xp[order], yp[order], tp[order]
                xo, yo = gnomonic(ob.g_samples, ob)
                if xo.min() < xp[0] or xo.max() > xp[-1]:
                    continue  # predicted path over the time window does not span the streak
                yc = np.interp(xo, xp, yp)
                cross = float(np.median(np.abs(yo - yc)))
                slope = np.gradient(yp, xp)
                ang = float(abs(np.arctan(np.interp(0.0, xp, slope))))
                tau_mid = float(np.interp(0.0, xp, tp))
                rr = float(np.interp(0.0, xp, rho[gi[h]][front][order]))
                tol_c = n_sigma * np.sqrt((sigma_loc_km / rr) ** 2 + (SIG_CROSS_KM / rr) ** 2 +
                                          (EARTH_ROT * sigma_t) ** 2 + ob.sigma_offset_rad ** 2 +
                                          ob.sigma_rad ** 2 + (2e-4) ** 2)
                tol_a = n_sigma * np.sqrt(ob.orient_sigma ** 2 + (2 * sigma_loc_km / rr) ** 2 +
                                          (EARTH_ROT * sigma_t) ** 2) + 0.01
                ends_ok = all(ob.trusted_ends)
                tol_t = n_sigma * np.sqrt(sigma_t ** 2 + (sigma_loc_km / 7.0) ** 2 + 1.0) + \
                    (0.0 if ends_ok else T_max / 2)
                cost = (cross / tol_c) ** 2 + (ang / tol_a) ** 2 + ((tau_mid - t) / tol_t) ** 2
                if cost < 3.0:
                    cand[i].append(Match(i, int(k), float(cost), cross, ang, tau_mid, rr))
    # one satellite per streak, one streak per satellite
    sats = sorted({m.sat_index for v in cand.values() for m in v})
    result = [None] * len(obs_list)
    if sats:
        sidx = {s: j for j, s in enumerate(sats)}
        C = np.full((len(obs_list), len(sats)), 1e6)
        best = {}
        for i, v in cand.items():
            for m in v:
                j = sidx[m.sat_index]
                if m.cost < C[i, j]:
                    C[i, j] = m.cost
                    best[(i, j)] = m
        ri, ci = linear_sum_assignment(C)
        for i, j in zip(ri, ci):
            if C[i, j] < 1e5:
                result[i] = best[(i, j)]
    info = dict(n_candidates={int(i): len(v) for i, v in cand.items()})
    return result, info
