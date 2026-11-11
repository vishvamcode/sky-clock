"""Command line: ``skyclock analyze | simulate | detect | fetch | benchmark``."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

from .pipeline import CATALOG_CHOICES, load_catalog, parse_time, run_all
from .solve import MODES, Settings

ETHICS = ("sky-clock is for checking claims about images (journalism, science, your own photos). "
          "Do not use it to locate private people from their personal photos.")


def _settings(a):
    region = None
    if a.region:
        la, lo, r = (float(v) for v in a.region.split(","))
        region = (la, lo, r)
    return Settings(mode=a.mode, time=parse_time(a.time), lat=a.lat, lon=a.lon, sigma_time_s=a.sigma_time,
                    sigma_loc_km=a.sigma_loc, window_start=parse_time(a.window[0]) if a.window else None,
                    window_end=parse_time(a.window[1]) if a.window else None, exposure_s=a.exposure,
                    exposure_max_s=a.exposure_max, tracking=a.tracking, h_km=a.height / 1000.0,
                    sun_max_alt_deg=None if a.no_sun_check else -3.0, region=region,
                    max_solutions=a.max_solutions, up_marks=_up_marks(a))


def _up_marks(a):
    from .zenith import UpMarks
    v = [[float(x) for x in e.split(",")] for e in (a.vertical or [])]
    if any(len(e) != 4 for e in v):
        sys.exit("--vertical needs X1,Y1,X2,Y2 (pixels)")
    hz = [[float(x) for x in p.split(",")] for p in a.horizon.split(";")] if a.horizon else []
    m = UpMarks(verticals=v, horizon=hz, horizon_height_m=a.horizon_height, camera_level=a.level)
    return None if m.empty() else m


def cmd_analyze(a):
    from . import report
    print(ETHICS, file=sys.stderr)
    if a.mode == "blind" and not a.i_understand:
        sys.exit("blind mode locates the photographer: add --i-understand to confirm you have the right to do this "
                 "(your own photo, or a public image whose claim you are checking)")
    img = Path(a.image)
    s = _settings(a)
    if a.use_exif:
        from .imageio import exif_hints, load_image
        _, _, exif = load_image(img.read_bytes(), img.name)
        h = exif_hints(exif)
        s.time = s.time or h.get("time_utc")
        if s.lat is None and "lat" in h:
            s.lat, s.lon = h["lat"], h["lon"]
        s.exposure_s = s.exposure_s or h.get("exposure_s")
        print(f"EXIF hints used: { {k: str(v) for k, v in h.items()} }", file=sys.stderr)
    when = s.time or s.window_start
    cat_bytes = Path(a.catalog_file).read_bytes() if a.catalog_file else None
    catalog = load_catalog(a.catalog, when, cat_bytes, a.catalog_file or "file")
    print(f"catalogue: {catalog.source}, {len(catalog)} LEO objects", file=sys.stderr)
    wcs_bytes = Path(a.wcs).read_bytes() if a.wcs else None
    method = "wcs" if a.wcs else a.plate
    edits = {int(i): False for i in a.disable.split(",")} if a.disable else None

    def prog(f, msg):
        print(f"  [{f * 100:5.1f}%] {msg}", file=sys.stderr)

    out = run_all(img.read_bytes(), img.name, s, catalog, method, None, wcs_bytes, edits, prog)
    res = out["result"]
    ps = out["plate"].summary()
    md = report.summary_markdown(res, ps)
    print(md)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "result.json").write_text(report.to_json(res, out["detection"], ps))
    (od / "summary.md").write_text(md + "\n")
    pred = None
    if res.solutions:
        from .identify import predicted_in_view
        s0 = res.solutions[0]
        pred = predicted_in_view(catalog, res.epoch, s0.lat, s0.lon, res.epoch.seconds_from(s0.t_mid), out["plate"],
                                 s.h_km, T=s0.exposure_s, tracking=s.tracking)
    im = report.annotate(out["display"], out["detection"].streaks, res.solutions[0] if res.solutions else None, pred)
    im.save(od / "annotated.png")
    report.make_map(res, (s.lat, s.lon) if s.lat is not None else None).save(str(od / "map.html"))
    tp = report.time_plot(res)
    if tp:
        (od / "time.png").write_bytes(tp)
    print(f"\nwrote {od}/result.json, summary.md, annotated.png, map.html", file=sys.stderr)


def cmd_simulate(a):
    from . import simulate as S
    from .catalog import synthetic_constellation
    from .pipeline import SYNTHETIC_EPOCH
    when = parse_time(a.time)
    cat = synthetic_constellation(SYNTHETIC_EPOCH)
    cam = S.Camera(width=a.width, height=a.height, fov_x_deg=a.fov)
    img, wcs, truth = S.simulate_photo(cat, when, a.lat, a.lon, exposure_s=a.exposure, alt_deg=a.alt, az_deg=a.az,
                                       cam=cam, seed=a.seed, real_stars=a.real_stars,
                                       star_mag_limit=7.5 if a.real_stars else 6.5,
                                       dashed_gap=(a.dashed, a.dashed / 4) if a.dashed else None)
    paths = S.save_outputs(a.out, img, wcs, truth)
    print(f"{len(truth['streaks'])} satellites in frame; wrote " + ", ".join(paths))


def cmd_detect(a):
    from .detect import detect
    from .imageio import load_image
    from . import report
    img = Path(a.image)
    gray, disp, _ = load_image(img.read_bytes(), img.name)
    det = detect(gray)
    for st in det.streaks:
        print(f"#{st.id}: ({st.p1[0]:.1f},{st.p1[1]:.1f}) -> ({st.p2[0]:.1f},{st.p2[1]:.1f}) len {st.length_px:.0f}px "
              f"snr {st.snr:.0f} clipped {st.clipped1},{st.clipped2} dashed {st.dashed}")
    print(f"{len(det.stars)} stars; notes: {det.notes}")
    if a.out:
        report.annotate(disp, det.streaks).save(a.out)


def cmd_fetch(a):
    when = parse_time(a.time) if a.time else None
    c = load_catalog(a.catalog, when)
    print(f"{c.source}: {len(c)} LEO objects cached")


def cmd_benchmark(a):
    from . import validate as V
    folder = Path(a.folder)
    if a.simulate:
        n = V.make_simulated_set(folder, a.simulate, seed=a.seed, dashed=a.dashed,
                                 catalog_errors=not a.exact_catalog)
        print(f"made {n} simulated photos with truth files in {folder}", file=sys.stderr)
    modes = tuple(m.strip() for m in a.modes.split(","))
    rows, summary, calib, report_path = V.score_folder(folder, modes, progress=lambda m: print("  " + m, file=sys.stderr))
    print(report_path.read_text())
    if a.apply:
        if not calib.get("enough") and not a.force:
            sys.exit("fewer than 8 uniquely solved photos per mode: not applying (use --force to apply anyway)")
        print(f"wrote {V.apply_calibration(calib)}; the solver uses it from the next run", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(prog="skyclock", description="Time and place from satellite streaks in a night photo")
    sub = p.add_subparsers(dest="cmd", required=True)

    bm = sub.add_parser("benchmark", help="score photos with known time and place (FOLDER/*.truth.json)")
    bm.add_argument("folder")
    bm.add_argument("--modes", default="check,time,place", help="comma list of check,time,place,blind")
    bm.add_argument("--apply", action="store_true", help="write the suggested error model to skyclock/calibration.json")
    bm.add_argument("--force", action="store_true", help="apply even with fewer than 8 photos")
    bm.add_argument("--simulate", type=int, default=0, help="first make this many simulated photos in FOLDER")
    bm.add_argument("--dashed", action="store_true", help="simulated photos: dashed streaks (phone night mode)")
    bm.add_argument("--exact-catalog", action="store_true",
                    help="simulated photos: draw satellites exactly where the catalogue says (no catalogue errors)")
    bm.add_argument("--seed", type=int, default=0)
    bm.set_defaults(func=cmd_benchmark)

    an = sub.add_parser("analyze", help="analyse a photo")
    an.add_argument("image")
    an.add_argument("--mode", choices=list(MODES), default="identify",
                    help="; ".join(f"{k}: {v}" for k, v in MODES.items()))
    an.add_argument("--time", help="claimed/approximate exposure start, ISO 8601 UTC (e.g. 2026-11-10T16:10:00Z)")
    an.add_argument("--lat", type=float)
    an.add_argument("--lon", type=float)
    an.add_argument("--height", type=float, default=0.0, help="observer height above the ellipsoid, metres")
    an.add_argument("--sigma-time", type=float, default=60.0, help="1-sigma uncertainty of --time, s")
    an.add_argument("--sigma-loc", type=float, default=10.0, help="1-sigma uncertainty of --lat/--lon, km")
    an.add_argument("--window", nargs=2, metavar=("START", "END"), help="search window (time, blind modes)")
    an.add_argument("--region", help="limit place/blind search: LAT,LON,RADIUS_KM")
    an.add_argument("--exposure", type=float, help="exposure length, s (fitted if omitted)")
    an.add_argument("--exposure-max", type=float, default=30.0)
    an.add_argument("--tracking", action="store_true", help="camera was on a star-tracking mount")
    an.add_argument("--use-exif", action="store_true", help="take time/place/exposure from EXIF where present")
    an.add_argument("--catalog", choices=list(CATALOG_CHOICES), default="celestrak",
                    help="; ".join(f"{k}: {v}" for k, v in CATALOG_CHOICES.items()))
    an.add_argument("--catalog-file", help="TLE/OMM file for --catalog file")
    an.add_argument("--plate", choices=["local", "nova"], default="local",
                    help="plate solver (nova needs ASTROMETRY_NET_API_KEY in the environment)")
    an.add_argument("--wcs", help="use this WCS (FITS header) instead of plate solving")
    an.add_argument("--disable", help="comma-separated streak ids to ignore (e.g. aircraft)")
    an.add_argument("--no-sun-check", action="store_true", help="allow solutions where the Sun is up")
    an.add_argument("--max-solutions", type=int, default=5)
    an.add_argument("--i-understand", action="store_true", help="required for blind mode (see ethics note)")
    an.add_argument("--vertical", action="append", metavar="X1,Y1,X2,Y2",
                    help="a truly vertical edge in the photo (building corner, pole), in pixels; repeat for each")
    an.add_argument("--horizon", metavar="X,Y;X,Y;...", help="3+ points on a level sea/plain horizon, pixels")
    an.add_argument("--horizon-height", type=float, default=2.0, help="eye height above that horizon, metres")
    an.add_argument("--level", action="store_true", help="camera was level (no roll): weak extra constraint")
    an.add_argument("--out", default="skyclock_out")
    an.set_defaults(func=cmd_analyze)

    si = sub.add_parser("simulate", help="make a synthetic test photo with known truth")
    si.add_argument("--out", default="demo")
    si.add_argument("--time", default="2026-11-10T16:10:00Z")
    si.add_argument("--lat", type=float, default=59.33)
    si.add_argument("--lon", type=float, default=18.06)
    si.add_argument("--alt", type=float, default=50.0)
    si.add_argument("--az", type=float, default=200.0)
    si.add_argument("--fov", type=float, default=70.0)
    si.add_argument("--width", type=int, default=3000)
    si.add_argument("--height", type=int, default=2000)
    si.add_argument("--exposure", type=float, default=10.0)
    si.add_argument("--dashed", type=float, default=0.0, help="simulate a stacked night mode with this sub-exposure")
    si.add_argument("--real-stars", action="store_true", help="Tycho-2 stars (needs astrometry.net index files)")
    si.add_argument("--seed", type=int, default=3)
    si.set_defaults(func=cmd_simulate)

    de = sub.add_parser("detect", help="only detect streaks and stars")
    de.add_argument("image")
    de.add_argument("--out", help="annotated PNG")
    de.set_defaults(func=cmd_detect)

    fe = sub.add_parser("fetch", help="download and cache a catalogue")
    fe.add_argument("--catalog", choices=list(CATALOG_CHOICES), default="celestrak")
    fe.add_argument("--time")
    fe.set_defaults(func=cmd_fetch)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
