"""Outputs: annotated image, map, time plot, JSON and Markdown summaries."""
from __future__ import annotations

import io
import json
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .fit import KM_PER_DEG_LAT

COLORS = ["#2a9df4", "#f4a32a", "#3cc47c", "#e5484d", "#a970ff", "#00b8b8", "#d6a100", "#ff6fae"]


def _font(size):
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate(display_rgb, streaks, solution=None, predicted=None, max_side=2000, up_marks=None):
    """Draw detected streaks (numbered), their identifications, predicted tracks and marked verticals."""
    im = Image.fromarray(display_rgb)
    W, H = im.size
    f = min(1.0, max_side / max(W, H))
    if f < 1:
        im = im.resize((int(W * f), int(H * f)))
    d = ImageDraw.Draw(im)
    font = _font(max(12, int(18 * max(im.size) / 2000)))
    names = {}
    if solution is not None:
        for m in solution.matches:
            names[m["streak"]] = m["name"]
    if predicted:
        for p in predicted:
            xs, ys, fr = np.asarray(p["x"]), np.asarray(p["y"]), np.asarray(p["front"], bool)
            ok = fr & (xs > -0.5 * W) & (xs < 1.5 * W) & (ys > -0.5 * H) & (ys < 1.5 * H)
            for k in range(len(xs) - 1):
                if ok[k] and ok[k + 1]:
                    d.line([(xs[k] * f, ys[k] * f), (xs[k + 1] * f, ys[k + 1] * f)], fill="#ffd60a", width=1)
    lw = max(2, int(3 * max(im.size) / 2000))
    for st in streaks:
        c = COLORS[st.id % len(COLORS)] if st.enabled else "#777777"
        p1 = (st.p1[0] * f, st.p1[1] * f)
        p2 = (st.p2[0] * f, st.p2[1] * f)
        # offset the marker line slightly so the streak itself stays visible
        v = np.subtract(p2, p1)
        n = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-9) * (6 * lw)
        d.line([tuple(p1 + n), tuple(np.add(p2, n))], fill=c, width=lw)
        for p, clipped in ((p1, st.clipped1), (p2, st.clipped2)):
            r = 3 * lw
            if clipped:
                d.rectangle([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=c, width=lw)
            else:
                d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=c, width=lw)
        label = f"#{st.id}" + (f" {names[st.id]}" if st.id in names else "") + (" (dashed)" if st.dashed else "")
        tx, ty = (p1[0] + p2[0]) / 2 + n[0] * 2, (p1[1] + p2[1]) / 2 + n[1] * 2
        bb = d.textbbox((0, 0), label, font=font)
        tx = float(np.clip(tx, 2, im.size[0] - (bb[2] - bb[0]) - 2))
        ty = float(np.clip(ty, 2, im.size[1] - (bb[3] - bb[1]) - 6))
        d.text((tx, ty), label, fill=c, font=font, stroke_width=2, stroke_fill="black")
    if up_marks is not None:
        for i, (x1, y1, x2, y2) in enumerate(up_marks.verticals):
            d.line([(x1 * f, y1 * f), (x2 * f, y2 * f)], fill="#00e5ff", width=lw)
            d.text((x1 * f + 4, y1 * f + 4), f"V{i + 1}", fill="#00e5ff", font=font, stroke_width=2,
                   stroke_fill="black")
        for x, y in up_marks.horizon:
            r = 2 * lw
            d.ellipse([x * f - r, y * f - r, x * f + r, y * f + r], outline="#00e5ff", width=lw)
    return im


def png_bytes(im):
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def ellipse_points(lat, lon, cov_en_km, n_sigma=2.0, n=72):
    """Polygon (lat, lon) of an n-sigma error ellipse from an (east, north) covariance in km."""
    if cov_en_km is None:
        return None
    C = np.asarray(cov_en_km, float)
    if not np.all(np.isfinite(C)):
        return None
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0, None)
    th = np.linspace(0, 2 * np.pi, n)
    pts = (V @ (np.sqrt(w)[:, None] * np.stack([np.cos(th), np.sin(th)]))) * n_sigma
    dlat = pts[1] / KM_PER_DEG_LAT
    dlon = pts[0] / (KM_PER_DEG_LAT * max(np.cos(np.radians(lat)), 0.05))
    return np.stack([lat + dlat, lon + dlon], 1).tolist()


def make_map(result, claim=None, height=480):
    """Folium map with solutions (2-sigma ellipses), the claim, and single-streak loci."""
    import folium
    sols = result.solutions
    if sols:
        center = [sols[0].lat, sols[0].lon]
    elif claim:
        center = list(claim)
    elif result.locus:
        center = result.locus[0]["points"][0]
    else:
        center = [20, 0]
    m = folium.Map(location=center, zoom_start=8 if sols else 4, tiles="OpenStreetMap", height=height)
    for i, s in enumerate(sols):
        col = "#d62828" if i == 0 else "#577590"
        tip = (f"#{i + 1}: {s.lat:.4f}, {s.lon:.4f} | start {s.t_start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]} UTC"
               f" | {len(s.matches)} streaks | chi2/dof {s.chi2 / max(s.dof, 1):.2f}")
        folium.CircleMarker([s.lat, s.lon], radius=6 if i == 0 else 4, color=col, fill=True, tooltip=tip).add_to(m)
        ell = ellipse_points(s.lat, s.lon, s.loc_cov_km)
        if ell and not s.degenerate:
            folium.PolyLine(ell, color=col, weight=2, tooltip=f"#{i + 1} 2-sigma").add_to(m)
    if claim:
        folium.Marker(list(claim), tooltip="claimed / given place", icon=folium.Icon(color="green")).add_to(m)
    if result.locus:
        for tr in result.locus[:300]:
            pts = tr["points"]
            if len(pts) > 1:
                folium.PolyLine(pts, color="#7b2cbf", weight=2, opacity=0.6,
                                tooltip=f"{tr['sat']} @ {tr['time'][:19]}").add_to(m)
    return m


def time_plot(result, max_solutions=1):
    """PNG of the posterior time distribution (Laplace approximation incl. shared catalogue bias)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sols = [s for s in result.solutions[:max_solutions] if s.samples]
    if not sols:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=130)
    ref = sols[0].t_start
    for i, s in enumerate(sols):
        T = s.exposure_s
        t = np.array(s.samples["t"])
        t_start = t - T / 2
        base = result.epoch.seconds_from(ref)
        ax.hist(t_start - base, bins=40, alpha=0.6 if i else 0.85, label=f"#{i + 1}", color=COLORS[i % len(COLORS)])
    ax.set_xlabel(f"exposure start, seconds after {ref.strftime('%H:%M:%S')} UTC")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    if len(sols) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png")
    plt.close(fig)
    return b.getvalue()


def summary_markdown(result, plate_summary=None):
    L = [f"**Mode:** {result.mode}"]
    if result.verdict:
        v = result.verdict
        L.append(f"**Verdict: {v['status'].upper()}** - {v['text']}")
        if "p_value" in v:
            L.append(f"p = {v['p_value']:.3g} (likelihood ratio, {v['dof']} dof); best fit is "
                     f"{v['best_minus_claim_km']:.2f} km and {v['best_start_minus_claim_s']:+.2f} s from the claim "
                     f"(tolerances {v['tolerances']['loc_km']} km, {v['tolerances']['time_s']} s, 1 sigma).")
    for i, s in enumerate(result.solutions):
        L.append("")
        hdr = "Best solution" if i == 0 else f"Alternative #{i + 1}"
        L.append(f"**{hdr}** ({len(s.matches)} streak(s) used{', DEGENERATE' if s.degenerate else ''})")
        se = f"+-{s.sigma_east_km:.2f} km E, +-{s.sigma_north_km:.2f} km N (1 sigma)" if s.sigma_east_km is not None else ""
        L.append(f"- Place: {s.lat:.5f}, {s.lon:.5f} {se}")
        st = f"+-{s.sigma_t_s:.2f} s" if s.sigma_t_s is not None else ""
        L.append(f"- Exposure start: {s.t_start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]} UTC {st}")
        sT = f" +-{s.sigma_exposure_s:.2f}" if s.sigma_exposure_s is not None else ""
        L.append(f"- Exposure length: {s.exposure_s:.2f}{sT} s; chi2/dof = {s.chi2:.1f}/{s.dof}")
        for mt in s.matches:
            L.append(f"  - streak #{mt['streak']}: {mt['name']} (NORAD {mt['norad']}), range {mt['range_km']:.0f} km, "
                     f"residual {mt['rms_px']:.2f} px")
        if s.unmatched:
            L.append(f"  - unexplained streaks: {', '.join('#' + str(u) for u in s.unmatched)}")
        for n in s.notes:
            L.append(f"  - {n}")
    z = getattr(result, "zenith", None)
    if z:
        L.append("")
        L.append(f"**Zenith from the marked lines:** RA {z['radec_deg'][0]:.2f}, Dec {z['radec_deg'][1]:.2f} deg "
                 f"(+-{z['sigma_deg']:.2f} deg, {z['n_marks']} marks)")
    for n in result.notes:
        L.append(f"> {n}")
    if result.catalog_info:
        c = result.catalog_info
        L.append("")
        age = c.get("median_age_days")
        L.append(f"Catalogue: {c['source']}, {c['n']} objects" +
                 (f", median element age {age:+.2f} days relative to the photo" if age is not None else ""))
    if plate_summary:
        L.append(f"Plate solution: {plate_summary.get('solver')}, centre RA {plate_summary['ra_center']:.3f} "
                 f"Dec {plate_summary['dec_center']:+.3f}, scale {plate_summary['pixel_scale_arcmin']:.3f}'/px" +
                 (f", star RMS {plate_summary['rms_px']:.2f} px" if plate_summary.get("rms_px") else ""))
    return "\n".join(L)


def to_json(result, detection=None, plate_summary=None, extra=None):
    d = result.to_dict()
    if detection is not None:
        d["streaks"] = [dict(id=s.id, p1=s.p1, p2=s.p2, clipped=[s.clipped1, s.clipped2], dashed=s.dashed,
                             enabled=s.enabled, length_px=s.length_px, snr=s.snr) for s in detection.streaks]
        d["n_stars"] = len(detection.stars)
    if plate_summary:
        d["plate"] = plate_summary
    d["generated_utc"] = datetime.utcnow().isoformat() + "Z"
    d["error_model"] = dict(per_sat_dt_s=0.28, shared_dt_s=0.30, cross_track_km=0.5,
                            note="Public-GP error model (ESTIMATED); calibrate on photos with ground truth.")
    if extra:
        d.update(extra)
    return json.dumps(d, indent=1, default=str)
