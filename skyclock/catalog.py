"""Satellite element catalogues: loading, fetching, merging and fast vectorised propagation.

Sources
-------
* CelesTrak GP (OMM JSON; no account): ``fetch_celestrak("active")``
* CelesTrak SupGP for Starlink (derived from SpaceX ephemerides; no account): ``fetch_supgp("starlink")``
* Space-Track GP history for old photos (free account): ``fetch_spacetrack_history(...)``
* Local TLE / 3LE / OMM JSON / OMM CSV files: ``load_file(path)``
* Synthetic Walker constellations for tests and demos: ``synthetic_constellation(...)``

CelesTrak asks clients not to download the same data more than once every 2 hours; the
cache below enforces that. Space-Track's user agreement restricts redistribution: cached
files stay on your machine and are never committed (see .gitignore).
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sgp4 import omm
from sgp4.api import WGS72, Satrec, SatrecArray
from skyfield.sgp4lib import theta_GMST1982

from .geometry import Epoch

CACHE_DIR = Path(os.environ.get("SKYCLOCK_CACHE", Path.home() / ".cache" / "skyclock"))
USER_AGENT = "sky-clock/0.1 (satellite streak analysis; https://github.com/vishvamcode/sky-clock)"


@dataclass
class Catalog:
    satrecs: list
    names: list
    norad: np.ndarray
    source: str = "unknown"
    _arr: SatrecArray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.norad = np.asarray(self.norad, dtype=np.int64)
        self.epoch_jd = np.array([s.jdsatepoch + s.jdsatepochF for s in self.satrecs])
        self.mean_motion = np.array([s.no_kozai * 1440 / (2 * np.pi) for s in self.satrecs])  # rev/day
        self._arr = SatrecArray(self.satrecs) if self.satrecs else None
        self._sub_cache = {}

    def __len__(self):
        return len(self.satrecs)

    def subset(self, idx) -> "Catalog":
        idx = np.asarray(idx, dtype=int)
        return Catalog([self.satrecs[i] for i in idx], [self.names[i] for i in idx], self.norad[idx], self.source)

    def leo_only(self, min_rev_per_day=6.0) -> "Catalog":
        return self.subset(np.where(self.mean_motion >= min_rev_per_day)[0])

    def epoch_age_days(self, epoch: Epoch):
        return (epoch.jd_utc + epoch.fr_utc) - self.epoch_jd

    def describe_age(self, epoch: Epoch) -> dict:
        age = self.epoch_age_days(epoch)
        return {
            "n": int(len(self)),
            "median_age_days": float(np.median(age)) if len(age) else None,
            "frac_older_than_3d": float(np.mean(np.abs(age) > 3)) if len(age) else None,
            "frac_newer_than_photo": float(np.mean(age < -0.5)) if len(age) else None,
        }

    def _array_for(self, idx):
        if idx is None:
            return self._arr
        key = tuple(np.asarray(idx).tolist())
        if key not in self._sub_cache:
            if len(self._sub_cache) > 64:
                self._sub_cache.clear()
            self._sub_cache[key] = SatrecArray([self.satrecs[i] for i in key])
        return self._sub_cache[key]

    def propagate(self, epoch: Epoch, secs, idx=None):
        """Earth-fixed (ITRS) position and rotating-frame velocity.

        Returns r (N, T, 3) km, v (N, T, 3) km/s, ok (N, T). ``v`` is the velocity relative to
        an observer fixed on the ground, i.e. exactly what determines apparent motion.
        """
        secs = np.atleast_1d(np.asarray(secs, dtype=float))
        arr = self._array_for(idx)
        jd, fr = epoch.jd_fr(secs)
        e, r, v = arr.sgp4(jd, fr)
        ok = e == 0
        theta, theta_dot = theta_GMST1982(jd, fr + epoch.dut1 / 86400.0)
        c, s = np.cos(theta), np.sin(theta)  # (T,)
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        xp = c * x + s * y
        yp = -s * x + c * y
        vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
        w = theta_dot / 86400.0
        vxp = c * vx + s * vy + w * yp
        vyp = -s * vx + c * vy - w * xp
        R = np.stack([xp, yp, z], -1)
        V = np.stack([vxp, vyp, vz], -1)
        R[~ok] = np.nan
        V[~ok] = np.nan
        return R, V, ok


# ------------------------------------------------------------------ parsing
def _from_records(records, source):
    sats, names, ids = [], [], []
    for rec in records:
        try:
            s = Satrec()
            omm.initialize(s, rec)
        except Exception:
            continue
        sats.append(s)
        names.append(str(rec.get("OBJECT_NAME", rec.get("NORAD_CAT_ID", "?"))).strip())
        ids.append(int(rec["NORAD_CAT_ID"]))
    return Catalog(sats, names, np.array(ids, dtype=np.int64), source)


def parse_omm_json(text_or_obj, source="omm-json") -> Catalog:
    data = json.loads(text_or_obj) if isinstance(text_or_obj, (str, bytes)) else text_or_obj
    return _from_records(data, source)


def parse_omm_csv(text, source="omm-csv") -> Catalog:
    return _from_records(list(csv.DictReader(io.StringIO(text))), source)


def parse_tle(text, source="tle") -> Catalog:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    sats, names, ids = [], [], []
    i = 0
    while i < len(lines):
        name = None
        if not lines[i].startswith("1 ") and i + 2 < len(lines) + 1 and i + 1 < len(lines) and lines[i + 1].startswith("1 "):
            name = lines[i].lstrip("0 ").strip()
            i += 1
        if i + 1 < len(lines) and lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            s = Satrec.twoline2rv(lines[i], lines[i + 1])
            sats.append(s)
            ids.append(int(s.satnum))
            names.append(name or str(s.satnum))
            i += 2
        else:
            i += 1
    return Catalog(sats, names, np.array(ids, dtype=np.int64), source)


def load_text(text: str, source="file") -> Catalog:
    t = text.lstrip()
    if t.startswith("[") or t.startswith("{"):
        return parse_omm_json(t if t.startswith("[") else "[" + t + "]", source)
    if t.upper().startswith("OBJECT_NAME") or t.upper().startswith("CCSDS_OMM_VERS"):
        return parse_omm_csv(text, source)
    return parse_tle(text, source)


def load_file(path) -> Catalog:
    p = Path(path)
    return load_text(p.read_text(), source=f"file:{p.name}")


def merge(primary: Catalog, override: Catalog) -> Catalog:
    """Replace entries of ``primary`` by ``override`` where NORAD IDs coincide; add new ones."""
    by_id = {int(n): (s, nm) for s, nm, n in zip(primary.satrecs, primary.names, primary.norad)}
    for s, nm, n in zip(override.satrecs, override.names, override.norad):
        by_id[int(n)] = (s, nm)
    ids = sorted(by_id)
    return Catalog([by_id[i][0] for i in ids], [by_id[i][1] for i in ids], np.array(ids), f"{primary.source}+{override.source}")


# ------------------------------------------------------------------ fetching
def _cached_get(url, name, max_age_h=2.0, session=None, **kw):
    import requests
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_h * 3600:
        return path.read_text()
    getter = session or requests
    resp = getter.get(url, headers={"User-Agent": USER_AGENT}, timeout=120, **kw)
    if resp.status_code != 200:
        if path.exists():  # fall back to stale cache rather than hammering the server
            return path.read_text()
        raise RuntimeError(f"GET {url} -> HTTP {resp.status_code}")
    path.write_text(resp.text)
    return resp.text


def fetch_celestrak(group="active", max_age_h=2.0) -> Catalog:
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=json"
    return parse_omm_json(_cached_get(url, f"celestrak_{group}.json", max_age_h), source=f"celestrak:{group}")


def fetch_supgp(file="starlink", max_age_h=2.0) -> Catalog:
    url = f"https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE={file}&FORMAT=json"
    return parse_omm_json(_cached_get(url, f"supgp_{file}.json", max_age_h), source=f"supgp:{file}")


def fetch_default(prefer_supgp=True, group="active") -> Catalog:
    cat = fetch_celestrak(group)
    if prefer_supgp:
        try:
            cat = merge(cat, fetch_supgp("starlink"))
        except Exception:
            pass
    return cat


def fetch_spacetrack_history(user, password, when: datetime, days_before=3.0, days_after=1.0) -> Catalog:
    """Element sets valid around ``when`` from Space-Track GP_HISTORY (needs a free account).

    For each object, the newest element set with epoch <= when is used (else the earliest after).
    Results are cached locally, because Space-Track asks that history is downloaded only once.
    """
    import requests
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    d1 = (when - timedelta(days=days_before)).strftime("%Y-%m-%d")
    d2 = (when + timedelta(days=days_after)).strftime("%Y-%m-%d")
    name = f"spacetrack_hist_{d1}_{d2}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if path.exists():
        recs = json.loads(path.read_text())
    else:
        s = requests.Session()
        r = s.post("https://www.space-track.org/ajaxauth/login", data={"identity": user, "password": password},
                   headers={"User-Agent": USER_AGENT}, timeout=60)
        if r.status_code != 200 or "Failed" in r.text:
            raise RuntimeError("Space-Track login failed")
        q = (f"https://www.space-track.org/basicspacedata/query/class/gp_history/EPOCH/{d1}--{d2}"
             f"/MEAN_MOTION/>6/orderby/NORAD_CAT_ID,EPOCH/format/json")
        r = s.get(q, headers={"User-Agent": USER_AGENT}, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"Space-Track query failed: HTTP {r.status_code}")
        recs = r.json()
        path.write_text(json.dumps(recs))
    tgt = when.replace(tzinfo=None)
    best = {}
    for rec in recs:
        try:
            ep = datetime.fromisoformat(rec["EPOCH"].replace("Z", ""))
        except Exception:
            continue
        nid = int(rec["NORAD_CAT_ID"])
        score = (0 if ep <= tgt else 1, abs((tgt - ep).total_seconds()))
        if nid not in best or score < best[nid][0]:
            best[nid] = (score, rec)
    return _from_records([v[1] for v in best.values()], source=f"spacetrack-history:{d1}..{d2}")


# ------------------------------------------------------------------ synthetic
STARLINK_LIKE_SHELLS = [  # (incl deg, alt km, planes, sats/plane, Walker F)
    (53.0, 480, 72, 22, 17),
    (53.2, 540, 72, 22, 11),
    (70.0, 570, 36, 20, 5),
    (97.6, 560, 6, 58, 1),
    (43.0, 530, 48, 50, 7),
]


def synthetic_constellation(epoch_dt: datetime, shells=None, slot_jitter_km=5.0, seed=0, scale=1.0) -> Catalog:
    """Walker-delta shells initialised with real SGP4 (near-circular orbits).

    ``scale`` < 1 thins every plane (keeps tests fast). Slot jitter breaks exact Walker symmetry, as
    real constellations do (amount unknown; see the feasibility report).
    """
    shells = shells or STARLINK_LIKE_SHELLS
    rng = np.random.default_rng(seed)
    if epoch_dt.tzinfo is None:
        epoch_dt = epoch_dt.replace(tzinfo=timezone.utc)
    ref = datetime(1949, 12, 31, tzinfo=timezone.utc)
    ep_days = (epoch_dt - ref).total_seconds() / 86400.0
    sats, names, ids = [], [], []
    nid = 90000
    mu = 398600.4418
    for si, (inc, alt, P, S, F) in enumerate(shells):
        S_eff = max(2, int(round(S * scale)))
        a = 6378.137 + alt
        n_rad_min = np.sqrt(mu / a**3) * 60.0
        for j in range(P):
            for k in range(S_eff):
                u0 = 2 * np.pi * k / S_eff + 2 * np.pi * F * j / (S_eff * P) + rng.normal(0, slot_jitter_km / a)
                s = Satrec()
                s.sgp4init(WGS72, "i", nid, ep_days, 1e-5, 0.0, 0.0, 1e-4, 0.0, np.radians(inc),
                           u0 % (2 * np.pi), n_rad_min, (2 * np.pi * j / P) % (2 * np.pi))
                sats.append(s)
                names.append(f"SYN-{si}-{j:02d}-{k:02d}")
                ids.append(nid)
                nid += 1
    return Catalog(sats, names, np.array(ids), source="synthetic")
