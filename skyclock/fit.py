"""Measurement model and least-squares fit.

Parameters: observer latitude/longitude (height fixed), mid-exposure time t (s after the reference
epoch), exposure length T, and per-streak nuisances: along-track timing offset dt_i (catalogue error,
prior sigma ``SIG_DT``) and cross-track angular offset c_i (catalogue cross-track error at the
satellite's range plus plate-solution systematics).

Residuals are computed per streak in a gnomonic tangent plane centred on the observed arc:
  * cross-track: observed centre-line samples vs predicted track,
  * along-track: trusted (unclipped, undashed) endpoints vs predicted exposure start/end; untrusted
    endpoints only have to lie inside the predicted exposure track (one-sided).

Camera frame: a fixed tripod is Earth-fixed, so predicted directions are rotated into the sky frame
at mid-exposure; on a tracking mount the camera follows the stars, so each instant uses its own
rotation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from .geometry import Epoch, geodetic_to_itrs, unit

# Public-GP error model (see docs/feasibility/report.md; ESTIMATED, to be calibrated on real photos)
SIG_DT = 0.28  # s, per-satellite along-track timing error
SIG_SHARED_DT = 0.30  # s, shared along-track bias (all satellites move together)
SIG_CROSS_KM = 0.5  # km, per-satellite cross-track error


def _load_calibration():
    """Override the three numbers above from skyclock/calibration.json (written by
    ``skyclock benchmark FOLDER --apply`` after scoring photos with a known time and place)."""
    import json
    from pathlib import Path
    p = Path(__file__).with_name("calibration.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    g = globals()
    for k in ("SIG_DT", "SIG_SHARED_DT", "SIG_CROSS_KM"):
        if k in d:
            g[k] = float(d[k])
    return d


CALIBRATION = _load_calibration()
KM_PER_DEG_LAT = 111.2


class SatTrack:
    """Cubic-spline interpolation of one satellite's ITRS position, re-gridded on demand."""

    def __init__(self, catalog, idx, epoch: Epoch, t_center, half=150.0, step=2.0):
        self.catalog, self.idx, self.epoch = catalog, int(idx), epoch
        self.half, self.step = half, step
        self._build(t_center)

    def _build(self, tc):
        g = np.arange(tc - self.half, tc + self.half + self.step, self.step)
        r, _, ok = self.catalog.propagate(self.epoch, g, idx=[self.idx])
        if not ok.all():
            raise ValueError("propagation failed")
        self.lo, self.hi = g[0], g[-1]
        self.spline = CubicSpline(g, r[0], axis=0)

    def __call__(self, tau):
        tau = np.asarray(tau, float)
        if tau.min() < self.lo + 1 or tau.max() > self.hi - 1:
            self._build(0.5 * (tau.min() + tau.max()))
        return self.spline(tau)


def camera_dirs(epoch: Epoch, d_itrs, tau, t_mid, tracking):
    """Rotate ITRS directions into the camera's (WCS) sky frame."""
    rot_t = tau if tracking else np.full(np.shape(tau), t_mid)
    return epoch.itrs_to_gcrs_vecs(d_itrs, rot_t)


def _interp_extrap(x, xp, fp):
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    if lo.any():
        y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    if hi.any():
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def gnomonic(d, ob):
    z = d @ ob.mid
    return (d @ ob.tangent) / z, (d @ ob.pole) / z


@dataclass
class FitResult:
    x: np.ndarray  # full parameter vector [lat, lon, t, T, dt..., c...]
    cov: np.ndarray | None  # covariance of the full vector (fixed params: zero rows)
    chi2_data: float
    chi2_total: float
    n_data: int
    n_free: int
    free: np.ndarray
    streak_chi2: list
    streak_rms_px: list
    streak_n: list
    success: bool
    degenerate: bool = False
    notes: list = field(default_factory=list)

    @property
    def lat(self):
        return float(self.x[0])

    @property
    def lon(self):
        return float(self.x[1])

    @property
    def t(self):
        return float(self.x[2])

    @property
    def T(self):
        return float(self.x[3])

    def sigma(self, i):
        if self.cov is None or not self.free[i]:
            return None
        v = self.cov[i, i]
        return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else None

    def loc_cov_km(self):
        """2x2 covariance of (east, north) in km, or None."""
        if self.cov is None or not (self.free[0] and self.free[1]):
            return None
        c = self.cov[:2, :2]
        J = np.diag([KM_PER_DEG_LAT * np.cos(np.radians(self.lat)), KM_PER_DEG_LAT])  # (lon, lat) order fix below
        c2 = np.array([[c[1, 1], c[1, 0]], [c[0, 1], c[0, 0]]])  # (lon, lat)
        return J @ c2 @ J.T

    @property
    def dof(self):
        return max(self.n_data - self.n_free, 1)


class Fitter:
    def __init__(self, obs_list, sat_idx, catalog, epoch: Epoch, h_km=0.0, tracking=False, scale_rad=None,
                 T_max=3600.0, zenith=None):
        self.obs = obs_list
        self.zenith = zenith  # optional zenith.Zenith measured from vertical edges / horizon
        self.T_max = float(T_max)
        self.sat_idx = [int(k) for k in sat_idx]
        self.catalog, self.epoch, self.h, self.tracking = catalog, epoch, h_km, tracking
        self.m = len(obs_list)
        self.tracks = {}
        self.scale = scale_rad
        self._obs_xy = [gnomonic(ob.g_samples, ob) for ob in obs_list]
        self._obs_end_x = [gnomonic(ob.g_ends, ob)[0] for ob in obs_list]
        self.u = np.linspace(-1.0, 1.0, 61)  # track sampling grid, in units of the half-span
        self._span = [0.0] * self.m  # per streak: half-span (s) needed to cover the observed samples

    def track(self, k, tc):
        if k not in self.tracks:
            self.tracks[k] = SatTrack(self.catalog, k, self.epoch, tc)
        return self.tracks[k]

    def ranges(self, x):
        o = geodetic_to_itrs(x[0], x[1], self.h)
        return [float(np.linalg.norm(self.track(k, x[2])(np.array([x[2]]))[0] - o)) for k in self.sat_idx]

    def sigma_c(self, ob, rho):
        return float(np.hypot(SIG_CROSS_KM / rho, ob.sigma_offset_rad))

    def residuals(self, x, priors=None, parts=False):
        lat, lon, t, T = x[:4]
        T = max(T, 0.05)
        m = self.m
        dts, cs = x[4:4 + m], x[4 + m:4 + 2 * m]
        o = geodetic_to_itrs(lat, lon, self.h)
        out, per = [], []
        for i, (ob, k) in enumerate(zip(self.obs, self.sat_idx)):
            # sample the predicted track over a span that covers the whole observed streak, whatever the
            # trial exposure T (otherwise a short T would extrapolate the track and bias T upwards)
            xo, yo = self._obs_xy[i]
            for _ in range(3):
                Hs = max(0.75 * T, self._span[i], 1.0)
                tau = t + dts[i] + np.concatenate([Hs * self.u, [-T / 2, T / 2]])
                r = self.track(k, t)(tau)
                rel = r - o
                d = camera_dirs(self.epoch, unit(rel), tau, t, self.tracking)
                xp, yp = gnomonic(d, ob)
                xg, yg = xp[:-2], yp[:-2]
                margin = 0.05 * (xo.max() - xo.min() + 1e-9)
                if xg.min() <= xo.min() - margin and xg.max() >= xo.max() + margin:
                    break
                rate = (xg.max() - xg.min()) / (2 * Hs) + 1e-12  # gnomonic units per second
                xc = xg[len(xg) // 2]
                need = max(abs(xo.min() - margin - xc), abs(xo.max() + margin - xc)) / rate
                self._span[i] = min(1.3 * need, 3600.0)
            rho = float(np.linalg.norm(rel[len(self.u) // 2]))
            order = np.argsort(xg)
            yc = _interp_extrap(xo, xg[order], yg[order])
            rc = (yo - yc - cs[i]) / ob.sigma_rad
            xs, xe = xp[-2], xp[-1]
            xo1, xo2 = self._obs_end_x[i]
            same = (xe - xs) * (xo2 - xo1) >= 0
            pred = (xs, xe) if same else (xe, xs)
            ra = [(xo_ - xp_) / ob.sigma_end_rad for xo_, xp_, ok in zip((xo1, xo2), pred, ob.trusted_ends) if ok]
            # an untrusted end (frame edge, trees, fading, dashes) is not the exposure boundary, but the
            # visible streak must still lie inside the predicted exposure track: a satellite too slow to
            # cover the visible arc within the exposure is penalised (one-sided)
            lo, hi = min(xs, xe), max(xs, xe)
            ra += [(max(lo - xo_, 0.0) + max(xo_ - hi, 0.0)) / ob.sigma_end_rad
                   for xo_, ok in zip((xo1, xo2), ob.trusted_ends) if not ok]
            rpri = [dts[i] / SIG_DT, cs[i] / self.sigma_c(ob, rho)]
            seg = np.concatenate([rc, np.array(ra, float)])
            per.append(seg)
            out.append(seg)
            out.append(np.array(rpri))
        if self.zenith is not None:
            from .zenith import zenith_residuals
            out.append(zenith_residuals(self.zenith, self.epoch, lat, lon, t))
        res = np.concatenate(out)
        if priors:
            pr = []
            for j, (mu, sg) in priors.items():
                pr.append((x[j] - mu) / sg)
            res = np.concatenate([res, np.array(pr, float)])
        if parts:
            return res, per
        return res

    def fit(self, x0, free, priors=None, max_nfev=200):
        x0 = np.asarray(x0, float).copy()
        free = np.asarray(free, bool)
        x_scale = np.concatenate([[0.01, 0.01, 0.5, 0.5], np.full(self.m, 0.2), np.full(self.m, 1e-5)])
        diff = np.concatenate([[1e-6, 1e-6, 1e-3, 1e-3], np.full(self.m, 1e-3), np.full(self.m, 1e-7)])
        lb = np.concatenate([[-90, -540, -np.inf, 0.05], np.full(2 * self.m, -np.inf)])
        ub = np.concatenate([[90, 540, np.inf, max(self.T_max, 0.1)], np.full(2 * self.m, np.inf)])

        def f(xf):
            x = x0.copy()
            x[free] = xf
            return self.residuals(x, priors)

        xf0 = np.clip(x0[free], lb[free] + 1e-9, ub[free] - 1e-9)
        try:
            res = least_squares(f, xf0, x_scale=x_scale[free], diff_step=None, bounds=(lb[free], ub[free]),
                                method="trf", max_nfev=max_nfev, jac="2-point")
        except Exception as e:  # propagation failure etc.
            return FitResult(x0, None, np.inf, np.inf, 0, int(free.sum()), free, [], [], [], False, notes=[str(e)])
        x = x0.copy()
        x[free] = res.x
        x[1] = (x[1] + 180) % 360 - 180
        full, per = self.residuals(x, priors, parts=True)
        chi2_data = float(sum(np.sum(p**2) for p in per))
        n_data = int(sum(len(p) for p in per))
        if self.zenith is not None:  # the zenith's two residuals are data too
            from .zenith import zenith_residuals
            zr = zenith_residuals(self.zenith, self.epoch, x[0], x[1], x[2])
            chi2_data += float(zr @ zr)
            n_data += 2
        J = res.jac
        cov = np.zeros((len(x), len(x)))
        degenerate = False
        try:
            JTJ = J.T @ J
            w, V = np.linalg.eigh(JTJ)
            if w.min() <= w.max() * 1e-12:
                degenerate = True
            wi = np.where(w > w.max() * 1e-12, 1 / np.maximum(w, 1e-300), np.inf)
            # infinite variance along degenerate directions
            Cf = (V * np.where(np.isfinite(wi), wi, 0)) @ V.T
            if degenerate:
                bad = V[:, ~np.isfinite(wi)]
                Cf = Cf + bad @ bad.T * 1e12
            idx = np.where(free)[0]
            cov[np.ix_(idx, idx)] = Cf
        except np.linalg.LinAlgError:
            cov = None
        rms_px, chi2_s, n_s = [], [], []
        for ob, p in zip(self.obs, per):
            nc = len(ob.g_samples)
            rc = p[:nc] * ob.sigma_rad
            rms_px.append(float(np.sqrt(np.mean(rc**2)) / self.scale) if self.scale else None)
            chi2_s.append(float(np.sum(p**2)))
            n_s.append(len(p))
        return FitResult(x, cov, chi2_data, float(np.sum(full**2)), n_data, int(free.sum()), free, chi2_s, rms_px,
                         n_s, bool(res.success), degenerate)


def initial_T(obs_list, sat_idx, catalog, epoch, lat, lon, t, h_km=0.0, default=10.0):
    """Exposure guess from trusted streak lengths divided by predicted angular speed."""
    o = geodetic_to_itrs(lat, lon, h_km)
    guesses = []
    for ob, k in zip(obs_list, sat_idx):
        if not all(ob.trusted_ends):
            continue
        r, _, ok = catalog.propagate(epoch, [t - 0.5, t + 0.5], idx=[k])
        if not ok.all():
            continue
        d = unit(r[0] - o)
        w = np.arccos(np.clip(d[0] @ d[1], -1, 1))
        if w > 0:
            guesses.append(2 * ob.half_len / w)
    return float(np.median(guesses)) if guesses else default
