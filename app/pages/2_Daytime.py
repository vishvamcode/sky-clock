"""sky-clock, daytime: the Sun's altitude from shadows (or the Sun itself), then place or time."""
from __future__ import annotations

import sys
from datetime import datetime, time as dtime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from skyclock import sun as SU  # noqa: E402
from skyclock.imageio import exif_hints, load_image  # noqa: E402
from skyclock import public  # noqa: E402

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:  # pragma: no cover
    streamlit_image_coordinates = None

st.set_page_config(page_title="sky-clock: daytime", layout="wide")
st.title("Daytime: time and place from the Sun")
st.info("Use this only on your own photos, or public images whose claimed time/place you are checking. "
        "Do not use it to locate a private person.")
st.markdown(
    "Mark **two or more upright objects and the tips of their shadows** (posts, people standing, lamp posts on "
    "level ground). The lines from each shadow tip to the top of its object all point at the Sun, so the Sun "
    "does not need to be in the picture. The same objects show which way is up. Together they give the Sun's "
    "altitude, typically to about half a degree with a phone photo.\n\n"
    "* time known: the place lies on a circle (add the camera's compass heading for a point, roughly +-400 km "
    "with a phone compass);\n"
    "* place known: the time, to a few minutes;\n"
    "* two photos a few hours apart from one place: the place, to about 200 km.")

up = st.file_uploader("Daytime photo (JPEG, PNG, TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])
if up is None:
    st.stop()
gray, disp, exif = load_image(up.getvalue(), up.name)
H, W = gray.shape
h = exif_hints(exif)
key = f"{up.name}_{W}x{H}"
pk, sk = "poles_" + key, "sunpt_" + key
st.session_state.setdefault(pk, [])
st.session_state.setdefault(sk, None)

c1, c2, c3 = st.columns(3)
fov = c1.number_input("Horizontal field of view (deg)", 5.0, 150.0, float(h.get("fov_x_deg", 69.0)),
                      help="From EXIF when present. A 26 mm-equivalent phone lens is about 69 deg.")
if "fov_x_deg" not in h:
    c1.caption("No focal length in EXIF: the default is a guess, and the altitude depends on it.")
use_heading = c2.checkbox("I know the compass direction the camera faced", "heading_deg" in h)
heading = None
if use_heading:
    heading = c2.number_input("Camera heading (deg from true north)", 0.0, 360.0, float(h.get("heading_deg", 0.0)))
    if h.get("heading_ref") == "M":
        c2.caption("EXIF says magnetic north: add the local magnetic declination.")
heading_sigma = c3.number_input("Heading uncertainty (deg)", 0.5, 45.0, 8.0)

what = st.radio("A click adds", ["pole: base, then top, then shadow tip (3 clicks)", "the Sun's centre"],
                horizontal=True)
from PIL import Image, ImageDraw  # noqa: E402

im = Image.fromarray(disp)
f = min(1.0, 1200 / max(W, H))
im = im.resize((int(W * f), int(H * f)))
d = ImageDraw.Draw(im)
for p in st.session_state[pk]:
    pts = [(p[i] * f, p[i + 1] * f) for i in range(0, len(p), 2)]
    if len(pts) >= 2:
        d.line(pts[:2], fill="#00e5ff", width=3)
    if len(pts) == 3:
        d.line([pts[2], pts[1]], fill="#ff9f1c", width=2)
    for q in pts:
        d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], outline="#ffffff", width=2)
if st.session_state[sk]:
    x, y = st.session_state[sk]
    d.ellipse([x * f - 10, y * f - 10, x * f + 10, y * f + 10], outline="#ffd60a", width=3)
if streamlit_image_coordinates is not None:
    click = streamlit_image_coordinates(im, key="dimg_" + key)
    if click and click.get("unix_time") != st.session_state.get("dclick_t"):
        st.session_state["dclick_t"] = click.get("unix_time")
        s = W / float(click.get("width") or im.size[0])
        x, y = round(click["x"] * s, 1), round(click["y"] * s, 1)
        if what.startswith("the Sun"):
            st.session_state[sk] = [x, y]
        else:
            P = st.session_state[pk]
            if P and len(P[-1]) < 6:
                P[-1] += [x, y]
            else:
                P.append([x, y])
        st.rerun()
else:
    st.image(im)
    st.caption("Install streamlit-image-coordinates to click; otherwise type the pixel coordinates below.")
cols = ["x_base", "y_base", "x_top", "y_top", "x_tip", "y_tip"]
df = pd.DataFrame([p + [np.nan] * (6 - len(p)) for p in st.session_state[pk]], columns=cols, dtype=float)
ed = st.data_editor(df, num_rows="dynamic", key=f"ped_{key}_{len(df)}_{sum(len(p) for p in st.session_state[pk])}",
                    hide_index=True)
st.session_state[pk] = [[float(v) for v in r if np.isfinite(v)] for r in ed.to_numpy()]
if st.button("Clear marks"):
    st.session_state[pk], st.session_state[sk] = [], None
    st.rerun()

poles = [p for p in st.session_state[pk] if len(p) == 6]
marks = SU.SunMarks(poles=poles, sun=st.session_state[sk], heading_deg=heading, heading_sigma_deg=heading_sigma)
fix = SU.measure_sun(marks, (H, W), fov) if (len(poles) >= 2 or (st.session_state[sk] and len(poles) >= 2)) else None
if fix is None:
    st.caption(f"{len(poles)} complete pole(s) marked; at least 2 are needed.")
    st.stop()
m1, m2 = st.columns(2)
m1.metric("Sun altitude", f"{fix.alt_deg:.2f} deg", f"+-{fix.alt_sigma_deg:.2f}", delta_color="off")
if fix.az_deg is not None:
    m2.metric("Sun azimuth", f"{fix.az_deg:.1f} deg", f"+-{fix.az_sigma_deg:.1f}", delta_color="off")
for n in fix.notes:
    st.caption(n)

if public.PUBLIC and not st.checkbox("I confirm this is my own photo, or a public image whose claimed time/place "
                                     "I am verifying, and I will not use the result to locate a private person."):
    st.stop()
st.header("What do you know?")
mode = st.radio("Known", ["the time: find the place", "the place: find the time"], horizontal=True)
d0 = h.get("time_utc") or datetime.now(timezone.utc).replace(microsecond=0)
c1, c2 = st.columns(2)
day = c1.date_input("Date (UTC)", d0.date())
tt = c2.text_input("Time (UTC, HH:MM:SS)" + (" (approximate is fine)" if mode.startswith("the place") else ""),
                   d0.strftime("%H:%M:%S"))
try:
    hh, mm, ss = (tt.split(":") + ["0", "0"])[:3]
    when = datetime.combine(day, dtime(int(hh), int(mm), int(float(ss))), tzinfo=timezone.utc)
except ValueError:
    st.error("Could not read the time.")
    st.stop()
if mode.startswith("the time"):
    r = SU.place_from_sun([(when, fix)]) if fix.az_deg is not None else None
    import folium
    from streamlit_folium import st_folium
    circ = SU.altitude_circle(fix, when)
    fm = folium.Map(location=[20, 0], zoom_start=2)
    folium.PolyLine(circ, color="#e63946", weight=2, tooltip="Sun at the measured altitude").add_to(fm)
    if r is not None:
        la, lo, cov, _ = r
        st.success(f"Place: {la:.2f}, {lo:.2f} (+-{np.sqrt(cov[0, 0]):.0f} km E, +-{np.sqrt(cov[1, 1]):.0f} km N)")
        folium.Marker([la, lo], tooltip="best place").add_to(fm)
    else:
        st.info("Altitude only: the place is somewhere on the red circle. Add the camera heading for a point.")
    st_folium(fm, height=420, use_container_width=True, returned_objects=[])
else:
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", -90.0, 90.0, float(h.get("lat", 0.0)), format="%.4f")
    lon = c2.number_input("Longitude", -180.0, 180.0, float(h.get("lon", 0.0)), format="%.4f")
    ts = SU.times_for_altitude(fix, lat, lon, when)
    if not ts:
        st.warning("The Sun never reached that altitude there within 12 h of the given time.")
    for o in ts:
        tag = "" if o["az_miss_sigma"] is None else f" (azimuth off by {o['az_miss_sigma']:.1f} sigma)"
        st.write(f"**{o['time_utc']:%Y-%m-%d %H:%M:%S} UTC** +- {o['sigma_t_s'] / 60:.1f} min{tag}")
