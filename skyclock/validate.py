"""Benchmark: score sky-clock against photos whose true time and place are known.

A benchmark folder holds photos, each with a sidecar ``<photo stem>.truth.json``::

    {
      "time_utc": "2026-11-20T16:42:05Z",   # exposure START, UTC (required)
      "lat": 59.3293, "lon": 18.0686,       # observer, degrees (required)
      "height_m": 20,                        # optional, metres above the ellipsoid (default 0)
      "exposure_s": 15,                      # optional: if given, also scored with the exposure known
      "time_uncertainty_s": 0.5,             # optional: how well YOU know the true time
      "catalog": "github-archive",           # optional: celestrak | github-archive | spacetrack | synthetic | file
      "catalog_file": "elements.tle",        # with "catalog": "file", relative to the folder
      "tracking": false,                     # optional: star-tracking mount
      "notes": "phone night mode"            # optional, free text
    }

``score_folder`` runs every photo through the modes below and writes ``results.csv`` and
``report.md``:

* ``time``  - the place is given (+-1 km), the time is searched in a +-1 h window: time error;
* ``place`` - the time is given (+-2 s), the place is fitted: distance error;
* ``check`` - the true claim with tight tolerances must come out *consistent*;
* ``blind`` - optional (slow): +-2 h window, no place, both errors.

Each error is also divided by the uncertainty the tool reported (a z-score). If the error model is
right, about 68 % of |z_time| are below 1, and the squared location z (2 degrees of freedom) has a
median of 1.39. ``calibrate`` turns the z-scores into scale factors for the error model in fit.py.
"""
from __future__ import annotations

import csv
import json
import math
import time as _time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np

from .geometry import great_circle_distance_km
from .pipeline import load_catalog, parse_time, plate_hint, plate_solve
from .solve import Settings, analyze

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".fits", ".fit", ".dng", ".cr2", ".nef", ".arw"}
CHI2_2DOF_MEDIAN = 2 * math.log(2)  # 1.386: median of a chi-square with 2 degrees of freedom


@dataclass
class Row:
    photo: str
    mode: str
    ok: bool
    n_streaks: int = 0
    n_matched: int = 0
    time_err_s: float | None = None
    sigma_t_s: float | None = None
    loc_err_km: float | None = None
    loc_z2: float | None = None  # squared Mahalanobis distance of the location error
    chi2_dof: float | None = None
    verdict: str | None = None
    p_value: float | None = None
    ambiguous: bool = False
    runtime_s: float = 0.0
    note: str = ""

    @property
    def z_time(self):
        if self.time_err_s is None or not self.sigma_t_s:
            return None
        return self.time_err_s / self.sigma_t_s


@dataclass
class Truth:
    path: Path
    time: object
    lat: float
    lon: float
    h_km: float = 0.0
    exposure_s: float | None = None
    catalog: str = "celestrak"
    catalog_file: str | None = None
    tracking: bool = False
    extra: dict = field(default_factory=dict)


def read_truth(photo: Path) -> Truth | None:
    tp = photo.with_name(photo.stem + ".truth.json")
    if not tp.exists():
        return None
    d = json.loads(tp.read_text())
    h_km = float(d.get("h_km", float(d.get("height_m", 0.0)) / 1000.0))
    return Truth(photo, parse_time(d["time_utc"]), float(d["lat"]), float(d["lon"]), h_km,
                 d.get("exposure_s"), d.get("catalog", "celestrak"), d.get("catalog_file"),
                 bool(d.get("tracking", False)), d)


def find_photos(folder) -> list[Truth]:
    folder = Path(folder)
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in PHOTO_EXT and not p.name.endswith(".annotated.png"):
            t = read_truth(p)
            if t is not None:
                out.append(t)
    return out


def _loc_z2(sol, lat, lon):
    """Squared Mahalanobis distance of the true place from the solution, using its 2x2 (east, north) km cov."""
    if sol.loc_cov_km is None:
        return None
    C = np.asarray(sol.loc_cov_km, float)
    if not np.all(np.isfinite(C)) or np.linalg.det(C) <= 0:
        return None
    de = (lon - sol.lon) * 111.2 * math.cos(math.radians(sol.lat))
    dn = (lat - sol.lat) * 111.2
    v = np.array([de, dn])
    return float(v @ np.linalg.solve(C, v))


def _row_from(result, photo, mode, truth: Truth, n_streaks, t0):
    r = Row(photo=photo, mode=mode, ok=bool(result.solutions), n_streaks=n_streaks,
            runtime_s=_time.time() - t0)
    r.ambiguous = any(n.startswith("AMBIGUOUS") for n in result.notes)
    if result.verdict:
        r.verdict = result.verdict.get("status")
        r.p_value = result.verdict.get("p_value")
    if not result.solutions:
        r.note = "; ".join(result.notes)[:200]
        return r
    s = result.solutions[0]
    r.n_matched = len(s.matches)
    r.time_err_s = (s.t_start - truth.time).total_seconds()
    r.sigma_t_s = s.sigma_t_s
    r.loc_err_km = great_circle_distance_km(truth.lat, truth.lon, s.lat, s.lon)
    r.loc_z2 = _loc_z2(s, truth.lat, truth.lon)
    r.chi2_dof = s.chi2 / max(s.dof, 1)
    if s.degenerate:
        r.note = "degenerate (one streak?)"
    return r


def mode_settings(truth: Truth, mode: str, exposure_known: bool) -> Settings:
    T = truth.exposure_s if exposure_known else None
    common = dict(exposure_s=T, exposure_max_s=max(30.0, (truth.exposure_s or 0) * 1.5), tracking=truth.tracking,
                  h_km=truth.h_km)
    if mode == "check":
        return Settings(mode="check", time=truth.time, lat=truth.lat, lon=truth.lon, sigma_time_s=2.0,
                        sigma_loc_km=1.0, **common)
    if mode == "time":
        return Settings(mode="time", lat=truth.lat, lon=truth.lon, sigma_loc_km=1.0,
                        window_start=truth.time - timedelta(hours=1), window_end=truth.time + timedelta(hours=1),
                        **common)
    if mode == "place":
        return Settings(mode="place", time=truth.time, sigma_time_s=2.0, **common)
    if mode == "blind":
        return Settings(mode="blind", window_start=truth.time - timedelta(hours=2),
                        window_end=truth.time + timedelta(hours=2), **common)
    raise ValueError(mode)


def score_photo(truth: Truth, modes=("check", "time", "place"), catalog_cache=None, progress=None) -> list[Row]:
    from .detect import detect
    from .imageio import exif_hints, load_image
    from .observations import build as build_obs
    rows = []
    name = truth.path.name
    t0 = _time.time()
    try:
        gray, _, exif = load_image(truth.path.read_bytes(), truth.path.name)
        det = detect(gray)
        wcs_side = truth.path.with_suffix(".wcs")
        if wcs_side.exists():  # simulated photos ship their true WCS
            plate = plate_solve(det, gray, "wcs", wcs_bytes=wcs_side.read_bytes())
        else:
            plate = plate_solve(det, gray, "local", hint=plate_hint(mode_settings(truth, "check", False)),
                                fov_deg=exif_hints(exif).get("fov_x_deg"))
        if plate is None:
            return [Row(name, m, False, len(det.streaks), note="plate solve failed",
                        runtime_s=_time.time() - t0) for m in modes]
        key = (truth.catalog, truth.catalog_file, truth.time.strftime("%Y-%m-%dT%H"))
        catalog_cache = catalog_cache if catalog_cache is not None else {}
        if key not in catalog_cache:
            fb = (truth.path.parent / truth.catalog_file).read_bytes() if truth.catalog_file else None
            catalog_cache[key] = load_catalog(truth.catalog, truth.time, fb, truth.catalog_file or "file")
        catalog = catalog_cache[key]
    except Exception as e:  # keep scoring the other photos
        return [Row(name, m, False, note=f"setup failed: {e}"[:200], runtime_s=_time.time() - t0) for m in modes]
    variants = [False] + ([True] if truth.exposure_s else [])
    for exposure_known in variants:
        for mode in modes:
            t1 = _time.time()
            label = mode + ("+exp" if exposure_known else "")
            if progress:
                progress(f"{name}: {label}")
            try:
                s = mode_settings(truth, mode, exposure_known)
                obs = build_obs(det.streaks, plate, s.epoch())
                res = analyze(obs, plate, catalog, s)
                rows.append(_row_from(res, name, label, truth, len(det.streaks), t1))
            except Exception as e:
                rows.append(Row(name, label, False, len(det.streaks), note=f"error: {e}"[:200],
                                runtime_s=_time.time() - t1))
    return rows


# ------------------------------------------------------------------ summary and calibration
def _pct(a, q):
    a = [x for x in a if x is not None and np.isfinite(x)]
    return float(np.percentile(a, q)) if a else None


def summarize(rows: list[Row]) -> dict:
    out = {}
    for mode in sorted({r.mode for r in rows}):
        rs = [r for r in rows if r.mode == mode]
        ok = [r for r in rs if r.ok]
        zt = [abs(r.z_time) for r in ok if r.z_time is not None]
        z2 = [r.loc_z2 for r in ok if r.loc_z2 is not None]
        d = dict(n=len(rs), solved=len(ok), ambiguous=sum(r.ambiguous for r in ok))
        if mode.startswith("time") or mode.startswith("blind"):
            d["time_err_abs_median_s"] = _pct([abs(r.time_err_s) for r in ok if r.time_err_s is not None], 50)
            d["time_err_abs_p90_s"] = _pct([abs(r.time_err_s) for r in ok if r.time_err_s is not None], 90)
            d["frac_time_within_1sigma"] = float(np.mean(np.array(zt) <= 1)) if zt else None
            d["frac_time_within_3sigma"] = float(np.mean(np.array(zt) <= 3)) if zt else None
            d["z_time_p68"] = _pct(zt, 68.27)
        if mode.startswith("place") or mode.startswith("blind"):
            d["loc_err_median_km"] = _pct([r.loc_err_km for r in ok], 50)
            d["loc_err_p90_km"] = _pct([r.loc_err_km for r in ok], 90)
            d["loc_z2_median"] = _pct(z2, 50)
        if mode.startswith("check"):
            d["consistent"] = sum(r.verdict == "consistent" for r in rs)
        d["chi2_dof_median"] = _pct([r.chi2_dof for r in ok], 50)
        out[mode] = d
    return out


def calibrate(rows: list[Row], min_photos=8) -> dict:
    """Scale factors for the error model, from the z-scores of the known-place time fits and known-time
    place fits. A factor above 1 means the tool's error bars are too small by that much.

    Only ``time`` and ``place`` rows with a unique (non-ambiguous) solution are used: the blind and check
    modes reuse the same fits and would count photos twice."""
    zt = [abs(r.z_time) for r in rows if r.ok and not r.ambiguous and r.mode.startswith("time")
          and r.z_time is not None]
    z2 = [r.loc_z2 for r in rows if r.ok and not r.ambiguous and r.mode.startswith("place") and r.loc_z2 is not None]
    chi = [r.chi2_dof for r in rows if r.ok and r.mode.startswith("time") and r.chi2_dof is not None]
    res = dict(n_time=len(zt), n_place=len(z2), enough=min(len(zt), len(z2)) >= min_photos)
    if zt:
        res["time_scale"] = max(float(np.percentile(zt, 68.27)), 1e-3)
    if z2:
        res["place_scale"] = max(math.sqrt(float(np.median(z2)) / CHI2_2DOF_MEDIAN), 1e-3)
    if chi:
        res["chi2_dof_median"] = float(np.median(chi))
    from . import fit
    if "time_scale" in res:
        res["suggested"] = dict(SIG_DT=round(fit.SIG_DT * res["time_scale"], 3),
                                SIG_SHARED_DT=round(fit.SIG_SHARED_DT * res["time_scale"], 3))
    if "place_scale" in res:
        res.setdefault("suggested", {})["SIG_CROSS_KM"] = round(fit.SIG_CROSS_KM * res["place_scale"], 3)
    return res


def write_outputs(folder, rows, summary, calib):
    folder = Path(folder)
    with open(folder / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["photo", "mode", "ok", "n_streaks", "n_matched", "time_err_s", "sigma_t_s", "z_time",
                "loc_err_km", "loc_z2", "chi2_dof", "verdict", "p_value", "ambiguous", "runtime_s", "note"]
        w.writerow(cols)
        for r in rows:
            w.writerow([getattr(r, c) if c != "z_time" else r.z_time for c in cols])
    lines = ["# sky-clock benchmark", "",
             "Errors are tool minus truth. z = error / the uncertainty the tool reported. If the error model is "
             "right, about 68 % of |z_time| are at most 1, and the median squared location z is 1.39.", "",
             "| mode | photos | solved | ambiguous | time err median / p90 (s) | within 1 / 3 sigma | "
             "place err median / p90 (km) | loc z2 median | consistent | chi2/dof median |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def f(x, nd=2):
        return "" if x is None else f"{x:.{nd}f}"
    for mode, d in summary.items():
        lines.append(
            f"| {mode} | {d['n']} | {d['solved']} | {d['ambiguous']} | "
            f"{f(d.get('time_err_abs_median_s'))} / {f(d.get('time_err_abs_p90_s'))} | "
            f"{f(d.get('frac_time_within_1sigma'))} / {f(d.get('frac_time_within_3sigma'))} | "
            f"{f(d.get('loc_err_median_km'))} / {f(d.get('loc_err_p90_km'))} | {f(d.get('loc_z2_median'))} | "
            f"{d.get('consistent', '')} | {f(d.get('chi2_dof_median'))} |")
    lines += ["", "## Calibration", ""]
    if not calib.get("n_time") and not calib.get("n_place"):
        lines.append("No uniquely solved time or place fits, so nothing to calibrate.")
    else:
        lines.append(f"From {calib['n_time']} time fits and {calib['n_place']} place fits: error bars should be "
                     f"scaled by {f(calib.get('time_scale'))} (time) and {f(calib.get('place_scale'))} (place).")
        if calib.get("suggested"):
            lines.append("Suggested error model: " + ", ".join(f"{k} = {v}" for k, v in calib["suggested"].items())
                         + ". Apply with `skyclock benchmark FOLDER --apply`.")
        if not calib.get("enough"):
            lines.append("Fewer than 8 photos per mode: treat these factors as a rough first look, not a calibration.")
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    (folder / "summary.json").write_text(json.dumps(dict(summary=summary, calibration=calib), indent=1))
    return folder / "report.md"


def score_folder(folder, modes=("check", "time", "place"), progress=None):
    photos = find_photos(folder)
    if not photos:
        raise RuntimeError(f"no photos with a .truth.json sidecar in {folder}")
    rows, cache = [], {}
    for t in photos:
        rows += score_photo(t, modes, cache, progress)
    summary = summarize(rows)
    calib = calibrate(rows)
    report = write_outputs(folder, rows, summary, calib)
    return rows, summary, calib, report


def make_simulated_set(folder, n=6, seed=0, fov=70.0, exposure=10.0, dashed=False, catalog_errors=True):
    """A benchmark folder of simulated photos with known truth (synthetic catalogue).

    With ``catalog_errors`` the satellites are drawn where the catalogue is wrong by the amounts the error
    model assumes (fit.SIG_DT, SIG_SHARED_DT, SIG_CROSS_KM), so a correct error model gives calibration
    factors near 1. Without it the catalogue is exact and the factors come out well below 1."""
    from datetime import datetime, timezone
    from . import simulate as S
    from .catalog import synthetic_constellation
    from .pipeline import SYNTHETIC_EPOCH
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cat = synthetic_constellation(SYNTHETIC_EPOCH)
    sites = [(59.33, 18.06), (55.70, 13.19), (48.85, 2.35), (40.42, -3.70), (52.52, 13.40), (60.17, 24.94)]
    made = 0
    for i in range(n * 4):
        if made >= n:
            break
        lat, lon = sites[i % len(sites)]
        when = datetime(2026, 11, 10, 16, 0, tzinfo=timezone.utc) + timedelta(minutes=float(rng.uniform(0, 90)))
        from . import fit
        render_cat = S.PerturbedCatalog(cat, fit.SIG_DT, fit.SIG_SHARED_DT, fit.SIG_CROSS_KM,
                                        seed=int(rng.integers(1 << 30))) if catalog_errors else cat
        img, wcs, truth = S.simulate_photo(render_cat, when, lat, lon, exposure_s=exposure,
                                           alt_deg=float(rng.uniform(40, 75)), az_deg=float(rng.uniform(0, 360)),
                                           cam=S.Camera(fov_x_deg=fov), seed=int(rng.integers(1 << 30)),
                                           dashed_gap=(exposure / 6, exposure / 24) if dashed else None)
        if len(truth["streaks"]) < 2:
            continue
        prefix = folder / f"sim{made:02d}"
        S.save_outputs(str(prefix), img, wcs, truth)
        t = json.loads(Path(f"{prefix}.truth.json").read_text())
        t.update(catalog="synthetic", notes="simulated" + (" with catalogue errors" if catalog_errors else ""))
        if catalog_errors:  # what was injected, for diagnosing the error model
            sel = {int(sk["norad"]) for sk in t["streaks"]}
            t["injected"] = dict(shared_dt_s=render_cat.shared_dt,
                                 per_sat={str(int(n)): dict(dt_s=float(render_cat.dt[k]), dx_km=float(render_cat.dx[k]))
                                          for k, n in enumerate(render_cat.norad) if int(n) in sel})
        Path(f"{prefix}.truth.json").write_text(json.dumps(t, indent=1))
        made += 1
    return made


# ------------------------------------------------------------------ applying a calibration
CALIBRATION_FILE = Path(__file__).with_name("calibration.json")


def apply_calibration(calib: dict, path=CALIBRATION_FILE):
    if not calib.get("suggested"):
        raise RuntimeError("nothing to apply")
    d = dict(calib["suggested"])
    d["_from"] = dict(n_time=calib.get("n_time"), n_place=calib.get("n_place"))
    Path(path).write_text(json.dumps(d, indent=1) + "\n")
    return path
