"""Turn detected streaks + a plate solution into sky measurements.

Each streak becomes a set of *geometric* GCRS unit vectors (annual aberration removed, so they can be
compared with satellite positions), a fitted great circle (pole), an arc (midpoint, tangent,
half-length) and uncertainties.

The camera-to-sky mapping in the WCS is valid at mid-exposure (it was fitted to star images, which are
averaged over the exposure). The solver accounts for Earth rotation during the exposure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Epoch, earth_beta, unit, wcs_to_geo


@dataclass
class StreakObs:
    id: int
    g_samples: np.ndarray  # (n, 3) geometric GCRS unit vectors along the streak
    g_ends: np.ndarray  # (2, 3) endpoints
    clipped: tuple  # (bool, bool) endpoint touches the frame edge
    dashed: bool
    pole: np.ndarray  # (3,) unit normal of the best-fit great circle
    mid: np.ndarray  # (3,) arc midpoint (on the great circle)
    tangent: np.ndarray  # (3,) unit, from end 1 towards end 2, perpendicular to mid and pole
    half_len: float  # rad
    sigma_rad: float  # effective per-sample cross-track sigma (rad), inflated for correlated errors
    sigma_end_rad: float  # endpoint along-track sigma (rad)
    sigma_offset_rad: float  # plate-solution systematic (rad), added to the cross-track nuisance prior
    orient_sigma: float  # sigma of the great-circle orientation (rad)
    pix_samples: np.ndarray
    pix_ends: np.ndarray
    length_px: float
    width_px: float
    # regular dashes (stacked night mode): geometric unit vectors of the interior dash centres and the time
    # of each as a fraction of the exposure from mid-exposure (-0.5 .. 0.5), ordered from end 1 to end 2
    dash_g: np.ndarray = None
    dash_f: np.ndarray = None

    @property
    def regular_dashes(self):
        return self.dash_f is not None

    @property
    def trusted_ends(self):
        # an irregular dash pattern (blinking, a dropped frame, stars in the gaps) says nothing about where
        # the exposure began; a regular one does: the first dash starts it and the last ends it
        return [(not c) and (not self.dashed or self.regular_dashes) for c in self.clipped]

    def arc_point(self, s):
        """Unit vector at arc coordinate s (rad) from the midpoint along the tangent."""
        s = np.asarray(s, float)[..., None]
        return np.cos(s) * self.mid + np.sin(s) * self.tangent


def dash_timing(dashes, p1, p2, min_dashes=3):
    """Check that a dashed streak looks like a stacked exposure, and time its dashes.

    A stacked night mode adds N sub-exposures of equal length ``on`` taken every ``P`` seconds, so the
    satellite leaves N dashes, evenly spaced in time, each covering the same fraction ``d = on / P`` of a
    cycle; the whole exposure is T = (N - 1 + d) P. Dash k (0 .. N-1) is centred at
    (k - (N - 1) / 2) / (N - 1 + d) of T from mid-exposure, whatever the frame rate. The satellite's angular
    speed varies slowly over a streak, so spacings may change smoothly; a missing dash, a merged pair, a
    star bridging a gap or blinking breaks the pattern and the streak is not timed.

    ``dashes``: [[[x, y], [x, y]], ...] in order from p1 to p2. Returns (fractions, centres_xy) or None."""
    if len(dashes) < min_dashes:
        return None
    u = np.subtract(p2, p1, dtype=float)
    L = float(np.linalg.norm(u))
    if L <= 0:
        return None
    u /= L
    ab = np.array([[(np.asarray(e, float) - p1) @ u for e in d] for d in dashes])
    ab.sort(axis=1)
    if np.any(np.diff(ab[:, 0]) <= 0) or np.any(ab[1:, 0] <= ab[:-1, 1]):
        return None
    N = len(ab)
    cen, lens = ab.mean(axis=1), ab[:, 1] - ab[:, 0]
    k = np.arange(N)
    deg = 2 if N >= 5 else 1
    coef = np.polyfit(k, cen, deg)
    spacing = np.polyval(np.polyder(coef), k)
    if np.any(spacing <= 0):
        return None
    med = float(np.median(np.diff(cen)))
    if np.max(np.abs(cen - np.polyval(coef, k))) > max(1.5, 0.05 * med):
        return None
    sp = np.diff(cen)
    if np.any(sp < 0.75 * med) or np.any(sp > 1.25 * med):  # a dropped or split dash
        return None
    duty = lens / spacing
    d = float(np.median(duty))
    if not 0.2 <= d <= 0.98 or np.any(np.abs(duty - d) > max(0.12 * d, 2.0 / med)):
        return None
    f = (k - (N - 1) / 2) / (N - 1 + d)
    xy = np.asarray(p1, float)[None, :] + cen[:, None] * u[None, :]
    return f, xy


def build(streaks, plate, epoch: Epoch, endpoint_sigma_px=1.5, max_eff_samples=40):
    """``streaks``: list of detect.Streak (enabled ones are used). Returns list[StreakObs]."""
    beta = earth_beta(epoch)
    scale = plate.pixel_scale_rad()
    wcs_rms_px = plate.rms_px if plate.rms_px is not None else 0.5
    wcs_rms_px = float(np.clip(wcs_rms_px, 0.2, 5.0))
    out = []
    for st in streaks:
        if not st.enabled:
            continue
        pts = np.asarray(st.samples, float)
        if len(pts) < 5:
            continue
        ends = np.array([st.p1, st.p2], float)
        g = wcs_to_geo(plate.pix2unit(pts[:, 0], pts[:, 1]), beta)
        ge = wcs_to_geo(plate.pix2unit(ends[:, 0], ends[:, 1]), beta)
        # great-circle fit: smallest right-singular vector of the sample matrix
        _, _, vt = np.linalg.svd(g, full_matrices=False)
        n = vt[-1]
        e1 = unit(ge[0] - (ge[0] @ n) * n)
        e2 = unit(ge[1] - (ge[1] @ n) * n)
        mid = unit(e1 + e2)
        tan = unit(np.cross(n, mid))
        if (e2 - e1) @ tan < 0:
            tan = -tan
            n = -n
        half = 0.5 * float(np.arccos(np.clip(e1 @ e2, -1, 1)))
        n_s = len(pts)
        n_eff = min(n_s, max_eff_samples, max(5, int(st.length_px / max(3 * st.width_px, 3))))
        sig_px = max(float(st.sigma_px), 0.15)
        sigma_rad = sig_px * scale * np.sqrt(n_s / n_eff)
        orient = np.hypot(sig_px * scale * np.sqrt(12.0 / n_eff) / max(2 * half, 1e-6),
                          wcs_rms_px * scale / max(half, 1e-6))
        dash_g = dash_f = None
        # only a streak seen whole: with an end at the frame edge, the first visible dash need not be the first
        timing = (dash_timing(getattr(st, "dashes", None) or [], ends[0], ends[1])
                  if st.dashed and not (st.clipped1 or st.clipped2) else None)
        if timing is not None:
            f, xy = timing
            dash_f = f[1:-1]  # the outer dashes' far ends are the streak ends, already used
            dash_g = wcs_to_geo(plate.pix2unit(xy[1:-1, 0], xy[1:-1, 1]), beta)
        out.append(StreakObs(st.id, g, ge, (bool(st.clipped1), bool(st.clipped2)), bool(st.dashed), n, mid, tan, half,
                             sigma_rad, endpoint_sigma_px * scale, wcs_rms_px * scale, float(orient), pts, ends,
                             float(st.length_px), float(st.width_px), dash_g, dash_f))
    return out
