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

    @property
    def trusted_ends(self):
        return [(not c) and (not self.dashed) for c in self.clipped]

    def arc_point(self, s):
        """Unit vector at arc coordinate s (rad) from the midpoint along the tangent."""
        s = np.asarray(s, float)[..., None]
        return np.cos(s) * self.mid + np.sin(s) * self.tangent


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
        out.append(StreakObs(st.id, g, ge, (bool(st.clipped1), bool(st.clipped2)), bool(st.dashed), n, mid, tan, half,
                             sigma_rad, endpoint_sigma_px * scale, wcs_rms_px * scale, float(orient), pts, ends,
                             float(st.length_px), float(st.width_px)))
    return out
