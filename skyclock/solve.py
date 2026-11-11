"""End-to-end analysis: modes, candidate evaluation, ranking, verdicts."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.stats import chi2 as chi2_dist

from .fit import KM_PER_DEG_LAT, SIG_SHARED_DT, Fitter, initial_T
from .geometry import (Epoch, geodetic_to_itrs, great_circle_distance_km, itrs_to_geodetic)
from .identify import identify
from .search import blind_candidates, time_candidates

MODES = {
    "identify": "Time and place roughly known: identify the satellites and refine both",
    "check": "Test a claimed time and place (e.g. from EXIF or a caption) for consistency",
    "time": "Place known: find the time",
    "place": "Time known: find the place",
    "blind": "Neither known (within a search window): find both",
}


@dataclass
class Settings:
    mode: str = "identify"
    time: datetime | None = None  # claimed / approximate start of exposure (UTC)
    lat: float | None = None
    lon: float | None = None
    sigma_time_s: float = 60.0  # how uncertain the given time is
    sigma_loc_km: float = 10.0  # how uncertain the given place is
    window_start: datetime | None = None  # search window for 'time' and 'blind'
    window_end: datetime | None = None
    exposure_s: float | None = None  # known exposure; None = fit it
    exposure_max_s: float = 30.0
    tracking: bool = False  # camera on a star-tracking mount
    h_km: float = 0.0  # observer height (fixed)
    el_min_deg: float = 8.0
    sun_max_alt_deg: float | None = -3.0  # observer must be in (civil-ish) darkness; None disables
    require_sunlit: bool = True  # satellites must be in sunlight
    region: tuple | None = None  # (lat, lon, radius_km) limit for place/blind searches
    max_solutions: int = 5
    max_clusters: int = 25

    def epoch(self):
        ref = self.time or self.window_start
        if ref is None:
            raise ValueError("a time or a search window is required")
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        return Epoch(ref)


@dataclass
class Solution:
    lat: float
    lon: float
    sigma_east_km: float | None
    sigma_north_km: float | None
    loc_cov_km: list | None
    t_mid: datetime
    t_start: datetime
    sigma_t_s: float | None  # statistical + shared catalogue bias
    exposure_s: float
    sigma_exposure_s: float | None
    matches: list  # per used streak
    unmatched: list
    chi2: float
    dof: int
    degenerate: bool
    score: float
    notes: list = field(default_factory=list)
    samples: dict | None = None

    def to_dict(self):
        d = dict(self.__dict__)
        d["t_mid"] = self.t_mid.isoformat()
        d["t_start"] = self.t_start.isoformat()
        d.pop("samples", None)
        return d


@dataclass
class Result:
    mode: str
    solutions: list
    verdict: dict | None = None
    notes: list = field(default_factory=list)
    catalog_info: dict | None = None
    locus: list | None = None  # single-streak degenerate loci: list of [[lat, lon], ...]
    epoch: Epoch | None = None

    def to_dict(self):
        return dict(mode=self.mode, solutions=[s.to_dict() for s in self.solutions], verdict=self.verdict,
                    notes=self.notes, catalog=self.catalog_info, locus=self.locus)


# ---------------------------------------------------------------------------------------------
def _fit_assignment(obs, assign, catalog, epoch, lat, lon, t, s: Settings, scale, priors_loc=None,
                    priors_t=None, prior_T=None):
    items = sorted(assign.items())
    o_list = [obs[i] for i, _ in items]
    k_list = [k for _, k in items]
    T0 = s.exposure_s or min(initial_T(o_list, k_list, catalog, epoch, lat, lon, t, s.h_km, default=10.0),
                             s.exposure_max_s)
    fitter = Fitter(o_list, k_list, catalog, epoch, s.h_km, s.tracking, scale,
                    T_max=s.exposure_s or s.exposure_max_s)
    m = len(o_list)
    x0 = np.concatenate([[lat, lon, t, T0], np.zeros(2 * m)])
    free = np.ones(len(x0), bool)
    priors = {}
    if priors_loc is not None:
        (la, lo, sig_km) = priors_loc
        if sig_km <= 0:
            free[0] = free[1] = False
        else:
            priors[0] = (la, sig_km / KM_PER_DEG_LAT)
            priors[1] = (lo, sig_km / (KM_PER_DEG_LAT * max(np.cos(np.radians(la)), 0.05)))
    if priors_t is not None:
        (tt, sig) = priors_t
        if sig <= 0:
            free[2] = False
        else:
            priors[2] = (tt, sig)
    if s.exposure_s:
        free[3] = False
    elif prior_T is not None:
        priors[3] = prior_T
    if m == 1 and free[0] and free[1] and 0 not in priors:
        # one streak: the location is a 1-D locus; a weak prior keeps the problem well posed
        priors[0] = (lat, 30.0 / KM_PER_DEG_LAT)
        priors[1] = (lon, 30.0 / (KM_PER_DEG_LAT * max(np.cos(np.radians(lat)), 0.05)))
    res = fitter.fit(x0, free, priors)
    return res, fitter, [i for i, _ in items], k_list


def _solution_from_fit(res, fitter, obs_idx, k_list, catalog, epoch, obs, notes=None, n_samples=400,
                       loc_constrained=False):
    lat, lon, t, T = res.x[:4]
    C = res.loc_cov_km()
    se = sn = None
    if C is not None and np.all(np.isfinite(C)):
        se, sn = float(np.sqrt(max(C[0, 0], 0))), float(np.sqrt(max(C[1, 1], 0)))
    st = res.sigma(2)
    sig_t = float(np.hypot(st, SIG_SHARED_DT)) if st is not None else None
    sT = res.sigma(3)
    rhos = fitter.ranges(res.x)
    matches = []
    for j, (oi, k) in enumerate(zip(obs_idx, k_list)):
        matches.append(dict(streak=int(obs[oi].id), norad=int(catalog.norad[k]), name=catalog.names[k],
                            rms_px=res.streak_rms_px[j], chi2=res.streak_chi2[j], n=res.streak_n[j],
                            range_km=rhos[j], dt_s=float(res.x[4 + j]),
                            catalog_age_days=float(catalog.epoch_age_days(epoch)[k] - 0)))
    used = {obs[i].id for i in obs_idx}
    unmatched = [int(o.id) for o in obs if o.id not in used]
    degenerate = res.degenerate or (len(obs_idx) < 2 and not loc_constrained) or (se is not None and max(se, sn) > 200)
    samples = None
    if res.cov is not None and not res.degenerate:
        c3 = res.cov[:3, :3]
        try:
            rng = np.random.default_rng(1)
            smp = rng.multivariate_normal(res.x[:3], c3 + np.diag([1e-12, 1e-12, SIG_SHARED_DT**2]),
                                          n_samples, check_valid="ignore")
            samples = dict(lat=smp[:, 0].tolist(), lon=smp[:, 1].tolist(), t=smp[:, 2].tolist())
        except Exception:
            samples = None
    score = 100.0 * len(obs_idx) - res.chi2_total / res.dof
    return Solution(float(lat), float((lon + 180) % 360 - 180), se, sn, C.tolist() if C is not None else None,
                    epoch.datetime_at(t), epoch.datetime_at(t - T / 2), sig_t, float(T), sT, matches, unmatched,
                    float(res.chi2_data), int(res.dof), bool(degenerate), float(score), list(notes or []), samples)


def evaluate(obs, catalog, epoch, lat, lon, t, s: Settings, scale, assign=None, sig_loc=20.0, sig_t=10.0,
             priors_loc=None, priors_t=None, rounds=2):
    """Identify (if needed), fit, prune outliers, re-identify with the refined state, refit."""
    notes = []
    if assign is None:
        m, _ = identify(obs, catalog, epoch, lat, lon, t, sig_loc, sig_t, s.exposure_max_s, s.exposure_s,
                        s.h_km, s.tracking, s.require_sunlit)
        assign = {i: mm.sat_index for i, mm in enumerate(m) if mm is not None}
    if not assign:
        return None
    best = None
    for _round in range(rounds + 1):
        res, fitter, oi, kl = _fit_assignment(obs, assign, catalog, epoch, lat, lon, t, s, scale,
                                              priors_loc, priors_t)
        if not res.success and not np.isfinite(res.chi2_data):
            return best
        # outlier rejection: a mis-identified streak has a huge chi2 per point
        per = np.array(res.streak_chi2) / np.maximum(np.array(res.streak_n), 1)
        if len(assign) > 2 and per.max() > 25:
            drop = oi[int(np.argmax(per))]
            notes.append(f"streak {obs[drop].id}: dropped (did not fit, chi2/pt={per.max():.0f})")
            assign = {i: k for i, k in assign.items() if i != drop}
            continue
        best = (res, fitter, oi, kl)
        lat, lon, t = res.x[0], res.x[1], res.x[2]
        if _round == rounds:
            break
        # re-identify with tighter tolerances: may add streaks or correct identifications
        sl = res.loc_cov_km()
        sl = float(np.sqrt(np.trace(sl))) if sl is not None and np.all(np.isfinite(sl)) else sig_loc
        stt = res.sigma(2) or sig_t
        m2, _ = identify(obs, catalog, epoch, lat, lon, t, float(np.clip(sl, 1.0, sig_loc)),
                         float(np.clip(stt, 1.0, sig_t)), s.exposure_max_s, res.T, s.h_km, s.tracking,
                         s.require_sunlit)
        new = {i: mm.sat_index for i, mm in enumerate(m2) if mm is not None}
        merged = dict(assign)
        for i, k in new.items():
            merged.setdefault(i, k)
        if merged == assign:
            break
        assign = merged
    if best is None:
        return None
    res, fitter, oi, kl = best
    return _solution_from_fit(res, fitter, oi, kl, catalog, epoch, obs, notes, loc_constrained=priors_loc is not None)


def _dedupe(sols, km=25.0, sec=8.0):
    out = []
    for s in sorted(sols, key=lambda z: -z.score):
        if any(great_circle_distance_km(s.lat, s.lon, o.lat, o.lon) < km and
               abs((s.t_mid - o.t_mid).total_seconds()) < sec for o in out):
            continue
        out.append(s)
    return out


def analyze(obs, plate, catalog, s: Settings, progress=None) -> Result:
    result = _analyze(obs, plate, catalog, s, progress)
    _plain_language_notes(result, obs, s)
    return result


def _plain_language_notes(result: Result, obs, s: Settings):
    """Up-front warnings when the answer is ambiguous or the fit is poor (added after the search)."""
    sols = result.solutions
    if not sols:
        return
    best = sols[0]
    ties = [x for x in sols[1:] if len(x.matches) == len(best.matches)
            and x.chi2 - best.chi2 < max(10.0, 0.1 * best.chi2)]
    if ties:
        names = "; ".join(f"{', '.join(m['name'] for m in x.matches)} at {x.t_start:%H:%M:%S} UTC"
                          for x in [best] + ties)
        result.notes.insert(0, f"AMBIGUOUS: {len(ties) + 1} solutions fit about equally well ({names}). The photo "
                               "alone cannot choose between them. A second streak from another orbit, the time to "
                               "within a few minutes, or the exact place would.")
    if best.dof > 0 and best.chi2 / best.dof > 3:
        result.notes.insert(0, f"POOR FIT: even the best solution has chi2/dof = {best.chi2 / best.dof:.1f} "
                               "(about 1 is expected). The streak may be from an uncatalogued object or an aircraft, "
                               "the catalogue may be too old, or the exposure length, time window or place you gave "
                               "may be wrong.")
    if s.exposure_s is None and all(not any(o.trusted_ends) for o in obs) and best.exposure_s >= s.exposure_max_s - 0.5:
        result.notes.append(f"No streak shows a real start or end, so the exposure length is only bounded from "
                            f"below; the fit stopped at the maximum you allowed ({s.exposure_max_s:.0f} s).")


def _analyze(obs, plate, catalog, s: Settings, progress=None) -> Result:
    if s.mode not in MODES:
        raise ValueError(f"unknown mode {s.mode}")
    epoch = s.epoch()
    scale = plate.pixel_scale_rad()
    result = Result(s.mode, [], epoch=epoch)
    result.catalog_info = dict(source=catalog.source, **catalog.describe_age(epoch))
    if (catalog.source != "synthetic" and result.catalog_info.get("median_age_days") is not None
            and abs(result.catalog_info["median_age_days"]) > 3):
        result.notes.append("Catalogue epochs are more than 3 days from the photo: expect km-level and "
                            "second-level errors (use Space-Track history for old photos).")
    if not obs:
        result.notes.append("No usable streaks.")
        return result
    if len(obs) == 1:
        result.notes.append("Only one streak: the place is constrained to a line (or set of lines), not a "
                            "point. Time can still be found if the place is known, and vice versa.")
    T_guess = s.exposure_s or 10.0

    def t_rel(dt):
        return epoch.seconds_from(dt)

    if s.mode in ("identify", "check"):
        if s.time is None or s.lat is None or s.lon is None:
            raise ValueError("identify/check need a time and a place")
        t0 = t_rel(s.time) + T_guess / 2
        # identification tolerances are kept wide in 'check' mode, so that a wrong claim is still
        # matched to the satellites and then *measured* as wrong by the likelihood-ratio test
        wide = s.mode == "check"
        sol = evaluate(obs, catalog, epoch, s.lat, s.lon, t0, s, scale,
                       sig_loc=max(s.sigma_loc_km, 25.0 if wide else 2.0),
                       sig_t=max(s.sigma_time_s, 30.0 if wide else 2.0), priors_loc=None if s.mode == "check" else
                       (s.lat, s.lon, s.sigma_loc_km), priors_t=None if s.mode == "check" else (t0, s.sigma_time_s))
        if sol is None:
            result.notes.append("No catalogued satellite matches any streak at the given time and place.")
            if s.mode == "check":
                result.verdict = dict(status="no-match",
                                      text="No catalogued satellite explains the streaks at the claimed time "
                                           "and place. Either the claim is wrong, or the streaks are not from "
                                           "catalogued satellites (aircraft, meteors, uncatalogued objects), or "
                                           "the catalogue is stale. Try the 'time' or 'blind' mode to see where "
                                           "the streaks do fit.")
            return result
        result.solutions = [sol]
        if s.mode == "check":
            result.verdict = _verdict(obs, catalog, epoch, s, scale, sol, t0)
        return result

    if s.mode == "time":
        if s.lat is None or s.window_start is None or s.window_end is None:
            raise ValueError("time mode needs a place and a search window")
        o = geodetic_to_itrs(s.lat, s.lon, s.h_km)
        clusters, _ = time_candidates(obs, catalog, epoch, o, t_rel(s.window_start), t_rel(s.window_end),
                                      s.exposure_max_s, max(s.sigma_loc_km, 1.0), s.require_sunlit,
                                      el_min_deg=max(0.0, s.el_min_deg - 3), progress=progress)
        sols = []
        for cl in clusters[: s.max_clusters]:
            sol = evaluate(obs, catalog, epoch, s.lat, s.lon, cl.t, s, scale, assign=cl.assignment,
                           sig_loc=max(s.sigma_loc_km, 2.0), sig_t=10.0,
                           priors_loc=(s.lat, s.lon, s.sigma_loc_km))
            if sol is not None:
                sols.append(sol)
        result.solutions = _dedupe(sols)[: s.max_solutions]
        if not result.solutions:
            result.notes.append("No time in the window fits the streaks from this place.")
        return result

    # place / blind
    if s.mode == "place":
        if s.time is None:
            raise ValueError("place mode needs a time")
        tc = t_rel(s.time) + T_guess / 2
        lo, hi = tc - 3 * s.sigma_time_s - s.exposure_max_s / 2, tc + 3 * s.sigma_time_s + s.exposure_max_s / 2
        pt = (tc, s.sigma_time_s)
    else:
        if s.window_start is None or s.window_end is None:
            raise ValueError("blind mode needs a search window")
        lo, hi = t_rel(s.window_start), t_rel(s.window_end)
        pt = None
    region = None
    if s.region is not None:
        region = (geodetic_to_itrs(s.region[0], s.region[1], 0.0), float(s.region[2]))
    clusters, roots, segs = blind_candidates(obs, catalog, epoch, lo, hi, s.exposure_max_s, s.h_km, s.el_min_deg,
                                             sun_max_alt_deg=s.sun_max_alt_deg, require_sunlit=s.require_sunlit,
                                             region=region, progress=progress)
    if len(obs) == 1:
        result.locus = _locus(segs[0], roots[0], catalog, epoch)
        result.notes.append(f"{len(result.locus)} possible ground tracks (one per satellite that could have made "
                            "the streak). Add a second streak, the place, or a narrower time to resolve.")
        return result
    sols = []
    for ci, cl in enumerate(clusters[: s.max_clusters]):
        if progress:
            progress(min(1.0, (ci + 1) / max(1, min(len(clusters), s.max_clusters))), f"fitting candidate {ci + 1}")
        la, lo_, _ = itrs_to_geodetic(cl.o)
        sol = evaluate(obs, catalog, epoch, float(la), float(lo_), cl.t, s, scale, assign=cl.assignment,
                       sig_loc=40.0, sig_t=10.0, priors_t=pt)
        if sol is not None:
            sols.append(sol)
    sols = _dedupe(sols)
    result.solutions = sols[: s.max_solutions]
    if not sols:
        result.notes.append("No consistent solution found in the search window.")
    elif len(sols) > 1 and sols[0].matches and len(sols[1].matches) == len(sols[0].matches):
        result.notes.append("More than one solution explains the same number of streaks: compare chi2 values "
                            "and check the alternatives (aliases are expected when all streaks come from one "
                            "constellation shell).")
    return result


def _verdict(obs, catalog, epoch, s: Settings, scale, sol: Solution, t0):
    """Likelihood-ratio test of the claim (place +- sigma_loc_km, time +- sigma_time_s)."""
    assign = {}
    for mt in sol.matches:
        i = next(j for j, o in enumerate(obs) if o.id == mt["streak"])
        k = int(np.where(catalog.norad == mt["norad"])[0][0])
        assign[i] = k
    t_mid = epoch.seconds_from(sol.t_mid)
    best, *_ = _fit_assignment(obs, assign, catalog, epoch, sol.lat, sol.lon, t_mid, s, scale)
    tol_loc = max(s.sigma_loc_km, 0.05)
    tol_t = max(s.sigma_time_s, 0.05)
    t_claim_mid = epoch.seconds_from(s.time) + sol.exposure_s / 2
    claim, *_ = _fit_assignment(obs, assign, catalog, epoch, s.lat, s.lon, t_claim_mid, s, scale,
                                priors_loc=(s.lat, s.lon, tol_loc),
                                priors_t=(t_claim_mid, float(np.hypot(tol_t, SIG_SHARED_DT))))
    d = max(claim.chi2_total - best.chi2_total, 0.0)
    k = 3
    p = float(chi2_dist.sf(d, k))
    dist = float(great_circle_distance_km(s.lat, s.lon, best.lat, best.lon))
    dt = float(best.t - best.T / 2 - epoch.seconds_from(s.time))
    if p >= 0.01:
        status, text = "consistent", "The streaks are consistent with the claimed time and place."
    elif p >= 1e-4:
        status, text = "tension", "The streaks are in tension with the claim (marginal)."
    else:
        status, text = "inconsistent", "The streaks are inconsistent with the claimed time and place."
    gof = float(chi2_dist.sf(best.chi2_data, best.dof))
    if gof < 1e-3:
        text += (" Note: even the best fit is poor (chi2/dof = %.1f), so the error model may be too optimistic; "
                 "treat the verdict with caution." % (best.chi2_data / best.dof))
    return dict(status=status, text=text, p_value=p, delta_chi2=d, dof=k,
                best_minus_claim_km=dist, best_start_minus_claim_s=dt,
                tolerances=dict(loc_km=tol_loc, time_s=tol_t), fit_quality_p=gof)


def _locus(seg, roots, catalog, epoch, max_tracks=300):
    P, ri = seg
    if len(P) == 0:
        return []
    lat, lon, _ = itrs_to_geodetic(P)
    tracks = []
    for r in np.unique(ri)[:max_tracks]:
        m = ri == r
        pts = np.stack([lat[m], lon[m]], 1)
        tracks.append(dict(points=pts.tolist(), sat=catalog.names[int(roots.sat[r])],
                           time=epoch.datetime_at(float(roots.t[r])).isoformat()))
    return tracks
