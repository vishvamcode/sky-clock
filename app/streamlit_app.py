"""sky-clock web app: upload a night photo, get time and place from satellite streaks.

Run:  streamlit run app/streamlit_app.py
Nothing is written to disk except the public satellite-catalogue cache. Photos stay in memory for the
session; plate solving runs locally unless you choose nova.astrometry.net (then an EXIF-free
grayscale copy is uploaded privately).
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyclock import report  # noqa: E402
from skyclock.detect import detect  # noqa: E402
from skyclock.identify import predicted_in_view  # noqa: E402
from skyclock.imageio import exif_hints, load_image  # noqa: E402
from skyclock.observations import build as build_obs  # noqa: E402
from skyclock.pipeline import CATALOG_CHOICES, load_catalog, plate_hint, plate_solve  # noqa: E402
from skyclock.platesolve import local_solver_available  # noqa: E402
from skyclock.solve import MODES, Settings, analyze  # noqa: E402
from skyclock.zenith import UpMarks  # noqa: E402
from skyclock.bodies import BODIES, BodyMark  # noqa: E402
from skyclock import public  # noqa: E402

try:  # optional: click on the photo to mark lines (pip install streamlit-image-coordinates)
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:  # pragma: no cover
    streamlit_image_coordinates = None

st.set_page_config(page_title="sky-clock", page_icon="🛰️", layout="wide")

ETHICS = (
    "**Use responsibly.** sky-clock checks *claims* about night photos (journalism, science, your own "
    "pictures). It must not be used to locate private people from their personal photos. Photos are "
    "processed in memory and not stored."
)


# ------------------------------------------------------------------ cached steps
@st.cache_data(show_spinner=False, max_entries=4)
def _load_and_detect(data: bytes, name: str):
    gray, disp, exif = load_image(data, name)
    det = detect(gray)
    return gray, disp, exif, det


@st.cache_resource(show_spinner=False, max_entries=3)
def _catalog(kind, when_key, file_hash, _file_bytes=None, _user=None, _pw=None):
    when = datetime.fromisoformat(when_key) if when_key else None
    return load_catalog(kind, when, _file_bytes, "upload", spacetrack_user=_user, spacetrack_password=_pw)


@st.cache_data(show_spinner=False, max_entries=2)
def _demo_photo(real_stars: bool):
    from skyclock import simulate as S
    from skyclock.catalog import synthetic_constellation
    from skyclock.pipeline import SYNTHETIC_EPOCH
    from PIL import Image
    import io
    when = datetime(2026, 11, 10, 16, 10, tzinfo=timezone.utc)
    cat = synthetic_constellation(SYNTHETIC_EPOCH)
    img, wcs, truth = S.simulate_photo(cat, when, 59.33, 18.06, exposure_s=10, alt_deg=50, az_deg=200,
                                       cam=S.Camera(fov_x_deg=70), seed=3, real_stars=real_stars,
                                       star_mag_limit=7.5 if real_stars else 6.5)
    b = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 65535).astype(np.uint16)).save(b, format="PNG")
    return b.getvalue(), wcs.to_header().tostring().encode(), truth


def _dt_input(label, default: datetime | None, key):
    c1, c2 = st.columns(2)
    d0 = default or datetime.now(timezone.utc).replace(microsecond=0)
    d = c1.date_input(f"{label}: date (UTC)", d0.date(), key=key + "_d")
    t = c2.text_input(f"{label}: time (UTC, HH:MM:SS)", d0.strftime("%H:%M:%S"), key=key + "_t")
    try:
        hh, mm, ss = (t.strip().split(":") + ["0", "0"])[:3]
        tt = dtime(int(hh), int(mm), int(float(ss)))
    except ValueError:
        st.error(f"Could not read the time '{t}'")
        tt = dtime(0, 0, 0)
    return datetime.combine(d, tt, tzinfo=timezone.utc)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛰️ sky-clock")
    st.caption("Time and place from satellite streaks in a night photo")
    st.info(ETHICS)
    st.subheader("Satellite catalogue")
    short = {"celestrak": "CelesTrak, current (all active)", "celestrak-starlink": "CelesTrak, Starlink only",
             "github-archive": "Past orbits, public archive (no account)",
             "spacetrack": "Space-Track history (account)", "file": "My own TLE/OMM file",
             "synthetic": "Synthetic (demo only)"}
    choices = [c for c in CATALOG_CHOICES if not (public.PUBLIC and c == "spacetrack")]
    cat_kind = st.selectbox("Source", choices, format_func=short.get)
    st.caption(CATALOG_CHOICES[cat_kind])
    st_user = st_pw = None
    cat_file = None
    if cat_kind == "spacetrack":
        st.caption("Credentials stay in this browser session only and are sent only to space-track.org.")
        st_user = st.text_input("Space-Track username")
        st_pw = st.text_input("Space-Track password", type="password")
    if cat_kind == "file":
        up = st.file_uploader("TLE / OMM (JSON or CSV) file", type=["txt", "tle", "json", "csv"])
        cat_file = up.getvalue() if up else None
    st.subheader("Plate solving")
    methods = ["local", "nova", "wcs"]
    default_m = 0 if local_solver_available() else 1
    plate_method = st.radio("Method", methods, index=default_m, format_func={
        "local": "Local astrometry.net (private, needs install / Docker)",
        "nova": "nova.astrometry.net (free API key; uploads an EXIF-free grey copy, private)",
        "wcs": "I have a WCS file for this photo"}.get)
    nova_key = wcs_file = None
    if plate_method == "local" and not local_solver_available():
        st.warning("solve-field not found here. Use the Docker image, or another method.")
    if plate_method == "nova":
        nova_key = st.text_input("nova.astrometry.net API key", type="password")
    if plate_method == "wcs":
        w = st.file_uploader("WCS FITS header (e.g. wcs.fits from astrometry.net)", type=["wcs", "fits", "fit"])
        wcs_file = w.getvalue() if w else None
    st.subheader("Camera")
    tracking = st.checkbox("Camera was on a star-tracking mount", False)
    h_m = st.number_input("Observer height above sea level (m, approx.)", 0.0, 6000.0, 0.0, 50.0)

# ------------------------------------------------------------------ photo
st.header("1. Photo")
c1, c2 = st.columns([3, 2])
with c1:
    up = st.file_uploader("Night photo with satellite streaks (JPEG, PNG, TIFF, FITS; RAW needs rawpy)",
                          type=["jpg", "jpeg", "png", "tif", "tiff", "fits", "fit", "dng", "cr2", "cr3", "nef",
                                "arw"])
with c2:
    st.write("No photo at hand?")
    if st.button("Use a simulated demo photo (Stockholm, 2026-11-10 16:10 UTC)"):
        st.session_state["demo"] = True
    st.caption("The demo uses a synthetic Starlink-like constellation (selected automatically). Try the "
               "'check' mode first, then 'blind'.")

data = name = None
demo_truth = demo_wcs = None
if up is not None:
    data, name = up.getvalue(), up.name
    st.session_state["demo"] = False
elif st.session_state.get("demo"):
    data, demo_wcs, demo_truth = _demo_photo(local_solver_available())
    name = "demo.png"
if data is None:
    st.stop()

with st.spinner("Detecting streaks and stars..."):
    gray, disp, exif, det = _load_and_detect(data, name)
hints = exif_hints(exif)
img_key = hashlib.sha1(data).hexdigest()[:12]

st.subheader("Detected streaks")
st.caption("Untick streaks that are not satellites (aircraft, meteors, lens flares). Circles: real "
           "endpoints; squares: cut by the frame edge (not used for timing).")
rows = [dict(id=s.id, use=True, length_px=round(s.length_px), snr=round(s.snr), clipped=(s.clipped1 or s.clipped2),
             dashed=s.dashed) for s in det.streaks]
if rows:
    edited = st.data_editor(pd.DataFrame(rows), disabled=["id", "length_px", "snr", "clipped", "dashed"],
                            hide_index=True, key="ed_" + img_key)
    use = dict(zip(edited["id"], edited["use"]))
    for s in det.streaks:
        s.enabled = bool(use.get(s.id, True))
else:
    st.warning("No streaks detected.")
st.image(report.annotate(disp, det.streaks), use_container_width=True)
st.caption(f"{len(det.streaks)} streak(s), {len(det.stars)} star(s). " + " ".join(det.notes))
if hints:
    st.caption("EXIF: " + ", ".join(f"{k} = {v}" for k, v in hints.items()))

# ------------------------------------------------------------------ which way is up (optional)
def _marks_ui(img_key):
    """Vertical edges / level horizon marked on the photo -> UpMarks (or None)."""
    vk, hk, ck, bk = "vert_" + img_key, "hor_" + img_key, "click_" + img_key, "bod_" + img_key
    st.session_state.setdefault(vk, [])
    st.session_state.setdefault(hk, [])
    st.session_state.setdefault(bk, [])
    with st.expander("Optional: mark vertical edges, a sea horizon, the Moon or planets", expanded=False):
        st.markdown(
            "If the photo shows **truly vertical** things (a building corner, a lamp post, a door frame) or a **level "
            "sea horizon**, mark them. Together with the stars they give the latitude, and the longitude once the "
            "time is known. With satellites they turn a single streak's line of places into a point. Mark edges as "
            "long as possible; avoid trees and anything that may lean.\n\n"
            "The **Moon or a planet** dates the photo: they move against the stars (the Moon about its own width "
            "per hour, Venus and Mars about a degree a day, Jupiter and Saturn slower). Not sure which planet? "
            "Leave the name as 'any planet'.")
        H, W = gray.shape
        cur = UpMarks(verticals=st.session_state[vk], horizon=st.session_state[hk])
        curb = [BodyMark(r[0], r[1], None if r[2] == "any planet" else r[2]) for r in st.session_state[bk]]
        if streamlit_image_coordinates is not None:
            what = st.radio("A click on the photo adds", ["vertical edge (click both ends)", "horizon point",
                                                          "Moon / planet"],
                            horizontal=True, key="what_" + img_key)
            prev = report.annotate(disp, det.streaks, up_marks=cur, body_marks=curb, max_side=1200)
            click = streamlit_image_coordinates(prev, key="img_" + img_key)
            if click and click.get("unix_time") != st.session_state.get(ck + "_t"):
                st.session_state[ck + "_t"] = click.get("unix_time")
                f = W / float(click.get("width") or prev.size[0])
                x, y = round(click["x"] * f, 1), round(click["y"] * f, 1)
                if what.startswith("horizon"):
                    st.session_state[hk].append([x, y])
                elif what.startswith("Moon"):
                    st.session_state[bk].append([x, y, "any planet"])
                else:
                    pend = st.session_state.get(ck)
                    if pend is None:
                        st.session_state[ck] = [x, y]
                    else:
                        st.session_state[vk].append(pend + [x, y])
                        st.session_state[ck] = None
                st.rerun()
            if st.session_state.get(ck):
                st.caption(f"First end at {st.session_state[ck]}: now click the other end of the same edge.")
        else:
            st.caption("Type pixel coordinates (x to the right, y down, from the top-left corner) from any image "
                       "viewer. Install streamlit-image-coordinates to click on the photo instead.")
        c1, c2 = st.columns(2)
        vdf = pd.DataFrame(st.session_state[vk] or [], columns=["x1", "y1", "x2", "y2"], dtype=float)
        ved = c1.data_editor(vdf, num_rows="dynamic", key=f"ved_{img_key}_{len(vdf)}", hide_index=True)
        hdf = pd.DataFrame(st.session_state[hk] or [], columns=["x", "y"], dtype=float)
        hed = c2.data_editor(hdf, num_rows="dynamic", key=f"hed_{img_key}_{len(hdf)}", hide_index=True)
        st.session_state[vk] = [list(map(float, r)) for r in ved.dropna().to_numpy()]
        st.session_state[hk] = [list(map(float, r)) for r in hed.dropna().to_numpy()]
        st.caption("Moon / planets (pick the name; 'any planet' tries all of them)")
        bdf = pd.DataFrame(st.session_state[bk] or [], columns=["x", "y", "name"])
        bed = st.data_editor(bdf, num_rows="dynamic", key=f"bed_{img_key}_{len(bdf)}", hide_index=True,
                             column_config={"name": st.column_config.SelectboxColumn(
                                 options=["any planet"] + list(BODIES), default="any planet")})
        st.session_state[bk] = [[float(r[0]), float(r[1]), r[2] or "any planet"]
                                for r in bed.dropna(subset=["x", "y"]).to_numpy()]
        auto = st.checkbox("Also look for planets automatically among the brightest objects", False,
                           key="auto_" + img_key)
        c1, c2, c3 = st.columns(3)
        level = c1.checkbox("Camera was level (no tilt sideways)", False, key="lvl_" + img_key)
        hh = c2.number_input("Eye height above the horizon (m)", 0.0, 9000.0, 2.0, key="hh_" + img_key)
        if c3.button("Clear marks", key="clr_" + img_key):
            st.session_state[vk], st.session_state[hk], st.session_state[ck], st.session_state[bk] = [], [], None, []
            st.rerun()
    m = UpMarks(verticals=st.session_state[vk], horizon=st.session_state[hk], horizon_height_m=hh,
                camera_level=level)
    bodies = [BodyMark(r[0], r[1], None if r[2] == "any planet" else r[2]) for r in st.session_state[bk]]
    if m.empty():
        return None, bodies, auto
    if len(m.verticals) + int(m.camera_level) < 2 and len(m.horizon) < 3:
        st.caption("Up marks: one vertical edge alone is not enough; mark at least two (or 3 horizon points).")
    return m, bodies, auto


up_marks, body_marks, auto_bodies = _marks_ui(img_key)

# ------------------------------------------------------------------ question
st.header("2. What do you want to know?")
mode = st.radio("Mode", list(MODES), format_func=lambda m: f"{m}: {MODES[m]}", horizontal=False)
d_time = hints.get("time_utc")
if demo_truth is not None:
    d_time = datetime(2026, 11, 10, 16, 10, tzinfo=timezone.utc)
d_lat = hints.get("lat", 59.33 if demo_truth is not None else 0.0)
d_lon = hints.get("lon", 18.06 if demo_truth is not None else 0.0)
s = Settings(mode=mode, tracking=tracking, h_km=h_m / 1000.0, up_marks=up_marks, body_marks=body_marks,
             auto_bodies=auto_bodies)
use_body_window = False
if mode == "date":
    st.caption("Where to look for the date. A long window is fine: the planets are searched in seconds.")
    c1, c2 = st.columns(2)
    s.date_start = _dt_input("From", datetime(2000, 1, 1, tzinfo=timezone.utc), "ds")
    s.date_end = _dt_input("To", datetime.now(timezone.utc).replace(microsecond=0), "de")
    if st.checkbox("I know the place (makes the Moon far more precise)", False):
        c1, c2 = st.columns(2)
        s.lat = c1.number_input("Latitude (deg)", -90.0, 90.0, float(d_lat), format="%.4f", key="dlat")
        s.lon = c2.number_input("Longitude (deg)", -180.0, 180.0, float(d_lon), format="%.4f", key="dlon")
if mode in ("identify", "check", "place"):
    s.time = _dt_input("Exposure start" + (" (claimed)" if mode == "check" else " (approximate)"), d_time, "t")
    s.sigma_time_s = st.number_input("Time uncertainty, 1 sigma (s)", 0.1, 3600.0,
                                     2.0 if mode == "check" else 60.0)
if mode in ("identify", "check", "time"):
    c1, c2, c3 = st.columns(3)
    s.lat = c1.number_input("Latitude (deg, N positive)", -90.0, 90.0, float(d_lat), format="%.5f")
    s.lon = c2.number_input("Longitude (deg, E positive)", -180.0, 180.0, float(d_lon), format="%.5f")
    s.sigma_loc_km = c3.number_input("Place uncertainty, 1 sigma (km)", 0.01, 500.0,
                                     1.0 if mode == "check" else 10.0)
if mode in ("time", "blind") and body_marks:
    use_body_window = st.checkbox("Take the search window from the Moon/planet date", True)
if mode in ("time", "blind") and not use_body_window:
    st.caption("Search window. Longer windows take longer (about a minute per 6 h with ~10,000 satellites).")
    base = d_time or datetime.now(timezone.utc).replace(microsecond=0)
    s.window_start = _dt_input("Window start", base - timedelta(hours=1), "ws")
    s.window_end = _dt_input("Window end", base + timedelta(hours=1), "we")
    if public.check_window(s.window_start, s.window_end):
        st.error(public.check_window(s.window_start, s.window_end))
        st.stop()
    if s.window_end <= s.window_start:
        st.error("Window end must be after window start.")
        st.stop()
if mode in ("place", "blind"):
    if st.checkbox("Limit the search to a region"):
        c1, c2, c3 = st.columns(3)
        s.region = (c1.number_input("Region centre lat", -90.0, 90.0, float(d_lat)),
                    c2.number_input("Region centre lon", -180.0, 180.0, float(d_lon)),
                    c3.number_input("Radius (km)", 10.0, 20000.0, 2000.0))
c1, c2, c3 = st.columns(3)
known_exp = c1.checkbox("I know the exposure length", value="exposure_s" in hints)
if known_exp:
    s.exposure_s = c2.number_input("Exposure (s)", 0.05, 600.0, float(hints.get("exposure_s", 10.0)))
s.exposure_max_s = c3.number_input("Longest possible exposure (s)", 1.0, 600.0,
                                   max(30.0, float(s.exposure_s or 0) * 1.2))
if not st.checkbox("Observer must be in darkness (Sun below -3 deg)", True):
    s.sun_max_alt_deg = None

consent = True
if mode in ("blind", "place", "time", "date") or public.PUBLIC:
    consent = st.checkbox("I confirm this is my own photo, or a public image whose claimed time/place I am "
                          "verifying, and I will not use the result to locate a private person.")

go = st.button("Analyse", type="primary",
               disabled=not consent or not (any(x.enabled for x in det.streaks) or up_marks is not None
                                            or body_marks or (mode == "date" and auto_bodies)))
if not go and "last" not in st.session_state:
    st.stop()

# ------------------------------------------------------------------ run
if go:
    if demo_truth is not None and cat_kind != "synthetic":
        st.info("Demo photo: using the synthetic catalogue it was made with.")
        cat_kind = "synthetic"
    try:
        with public.analysis_slot():
            with st.spinner("Plate solving..."):
                wbytes = wcs_file if plate_method == "wcs" else None
                method = plate_method
                if method == "local" and not local_solver_available() and demo_wcs is not None:
                    method, wbytes = "wcs", demo_wcs
                plate = plate_solve(det, gray, method, nova_key, wbytes, hint=plate_hint(s),
                                    fov_deg=exif_hints(exif).get("fov_x_deg"))
            if plate is None:
                st.error("Plate solving failed: the stars could not be matched to a catalogue. Try a WCS file.")
                st.stop()
            if use_body_window:
                from dataclasses import replace as _replace
                from skyclock.solve import date_from_bodies, window_from_dates
                with st.spinner("Dating from the Moon/planets..."):
                    dts, _ = date_from_bodies(plate, _replace(s, auto_bodies=False), det)
                if not dts:
                    st.error("The marked Moon/planets give no date in 2000 to now: check the names, or set the window.")
                    st.stop()
                s = window_from_dates(s, dts)
                st.info(f"Search window from the Moon/planet date: {s.window_start:%Y-%m-%d %H:%M} to "
                        f"{s.window_end:%H:%M} UTC.")
            catalog = None
            if mode != "date":
                when = s.time or s.window_start
                with st.spinner("Loading satellite catalogue..."):
                    fh = hashlib.sha1(cat_file).hexdigest() if cat_file else None
                    when_key = when.strftime("%Y-%m-%dT%H") if cat_kind in ("spacetrack", "github-archive") else None
                    catalog = _catalog(cat_kind, when_key, fh, cat_file, st_user, st_pw)
            obs = build_obs(det.streaks, plate, s.epoch())
            bar = st.progress(0.0, "Analysing...")
            res = analyze(obs, plate, catalog, s, progress=lambda f, m: bar.progress(float(min(max(f, 0), 1)), m),
                          det=det)
            bar.empty()
            pred = None
            if res.solutions:
                s0 = res.solutions[0]
                pred = predicted_in_view(catalog, res.epoch, s0.lat, s0.lon, res.epoch.seconds_from(s0.t_mid), plate,
                                         s.h_km, T=s0.exposure_s, tracking=s.tracking)
            st.session_state["last"] = dict(res=res, plate=plate.summary(), pred=pred, key=img_key,
                                            claim=(s.lat, s.lon) if s.lat is not None else None)
    except Exception as e:  # show the reason, keep the app alive
        st.error(f"{type(e).__name__}: {e}")
        st.stop()

last = st.session_state.get("last")
if not last or last["key"] != img_key:
    st.stop()
res = last["res"]

st.header("3. Result")
if res.verdict:
    v = res.verdict
    box = {"consistent": st.success, "tension": st.warning}.get(v["status"], st.error)
    box(f"**{v['status'].upper()}**: {v['text']}")
for n in res.notes:
    st.warning(n)
if res.dates:
    st.subheader("Date from the Moon and planets")
    st.dataframe(pd.DataFrame([dict(rank=i + 1, time_utc=d["time_utc"][:16].replace("T", " "),
                                    plus_minus=report._fmt_s(d["sigma_t_s"]), bodies=", ".join(d["bodies"]))
                               for i, d in enumerate(res.dates[:10])]), hide_index=True)
if res.zenith and "lat" in res.zenith and not res.solutions:
    z = res.zenith
    z1, z2 = st.columns(2)
    z1.metric("Latitude from the marked lines", f"{z['lat']:.3f}")
    if "lon" in z:
        z2.metric("Longitude (at the given time)", f"{z['lon']:.3f}")
        z2.caption(f"+-{z['sigma_east_km']:.0f} km E, +-{z['sigma_north_km']:.0f} km N")
    elif "time_utc" in z:
        z2.metric("Time (at the given longitude)", z["time_utc"][11:19] + " UTC")
        z2.caption(f"+-{z['sigma_t_s'] / 60:.0f} min")
if res.solutions:
    b = res.solutions[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Place (lat, lon)", f"{b.lat:.4f}, {b.lon:.4f}",
              help="Best solution. 1-sigma errors east/north shown below.")
    m1.caption(f"+-{b.sigma_east_km:.2f} km E, +-{b.sigma_north_km:.2f} km N" if b.sigma_east_km is not None
               else "uncertainty not available")
    m2.metric("Exposure start (UTC)", b.t_start.strftime("%H:%M:%S.%f")[:-4])
    m2.caption(b.t_start.strftime("%Y-%m-%d") + (f", +-{b.sigma_t_s:.2f} s" if b.sigma_t_s else ""))
    m3.metric("Exposure", f"{b.exposure_s:.2f} s")
    m4.metric("Streaks explained", f"{len(b.matches)} / {len(b.matches) + len(b.unmatched)}")
    m4.caption(f"chi2/dof {b.chi2 / max(b.dof, 1):.2f}" + (" - DEGENERATE" if b.degenerate else ""))
    if len(res.solutions) > 1:
        best_c = res.solutions[0].chi2
        st.dataframe(pd.DataFrame([dict(rank=i + 1, lat=round(x.lat, 4), lon=round(x.lon, 4),
                                        start_utc=x.t_start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-4],
                                        streaks=len(x.matches), chi2_per_dof=round(x.chi2 / max(x.dof, 1), 2),
                                        delta_chi2=round(x.chi2 - best_c, 1))
                                   for i, x in enumerate(res.solutions)]), hide_index=True)
        st.caption("Alternatives: fewer streaks explained or a larger chi2 mean a worse fit. Close competitors "
                   "are aliases worth checking.")
c1, c2 = st.columns([3, 2])
with c1:
    sol0 = res.solutions[0] if res.solutions else None
    st.image(report.annotate(disp, det.streaks, sol0, last["pred"], up_marks=up_marks,
                             body_marks=body_marks), use_container_width=True,
             caption="Coloured: detected streaks with identifications. Yellow: predicted satellite tracks for "
                     "the best solution.")
with c2:
    m = report.make_map(res, last["claim"])
    try:
        from streamlit_folium import st_folium
        st_folium(m, height=420, use_container_width=True, returned_objects=[])
    except Exception:
        st.components.v1.html(m._repr_html_(), height=440)
    tp = report.time_plot(res)
    if tp:
        st.image(tp, use_container_width=True)
if demo_truth is not None and sol0 is not None:
    from skyclock.geometry import great_circle_distance_km
    tt = datetime.fromisoformat(demo_truth["time_utc"])
    st.info(f"Demo truth: {demo_truth['lat']}, {demo_truth['lon']}, start {tt.strftime('%H:%M:%S')} UTC. "
            f"Error: {great_circle_distance_km(sol0.lat, sol0.lon, demo_truth['lat'], demo_truth['lon']):.2f} km, "
            f"{(sol0.t_start - tt).total_seconds():+.2f} s.")
with st.expander("Full details (identifications, residuals, catalogue, plate solution)"):
    st.markdown(report.summary_markdown(res, last["plate"]))
st.download_button("Download result (JSON)", report.to_json(res, det, last["plate"]), "skyclock_result.json",
                   "application/json")
st.download_button("Download annotated image (PNG)",
                   report.png_bytes(report.annotate(disp, det.streaks, sol0, last["pred"])),
                   "skyclock_annotated.png", "image/png")
with st.expander("How to read this"):
    st.markdown(
        "- **Uncertainties** are 1 sigma from the fit, with a 0.3 s shared catalogue timing bias added to the time.\n"
        "- **chi2/dof** near or below 1 means the streaks fit the predicted satellite tracks as well as expected.\n"
        "- **Alternatives** with the same number of streaks are possible aliases (Starlink shells repeat their "
        "pattern every few minutes); a larger chi2 means a worse fit.\n"
        "- Error model (per-satellite 0.28 s along-track, 0.5 km cross-track, shared 0.3 s) is an estimate for "
        "public catalogue data, not yet calibrated on real photos.\n"
        "- One streak gives a line of possible places, not a point.")
