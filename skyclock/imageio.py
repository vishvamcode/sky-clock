"""Image loading (JPEG/PNG/TIFF/FITS, optional camera RAW via rawpy) and EXIF reading."""
from __future__ import annotations

import io
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image

RAW_EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf"}
Image.MAX_IMAGE_PIXELS = 400_000_000


def load_image(src, name: str | None = None):
    """Return (gray float32 HxW, display uint8 HxWx3, exif dict).

    ``src`` may be a path or raw bytes. For 8-bit JPEGs gray is the sum of RGB (gamma is not undone;
    that only rescales streak brightness, not geometry).
    """
    if isinstance(src, (str, Path)):
        name = name or str(src)
        data = Path(src).read_bytes()
    else:
        data = bytes(src)
    ext = Path(name or "x.jpg").suffix.lower()
    exif = {}
    if ext in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        arr = np.asarray(fits.getdata(io.BytesIO(data)), dtype=np.float32)
        gray = arr.sum(axis=0) if arr.ndim == 3 else arr
    elif ext in RAW_EXT:
        import rawpy  # optional dependency: pip install rawpy
        with rawpy.imread(io.BytesIO(data)) as raw:
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
        gray = rgb.astype(np.float32).sum(axis=2)
    else:
        im = Image.open(io.BytesIO(data))
        exif = read_exif(im)
        if im.mode in ("I;16", "I;16B", "I", "F"):
            gray = np.asarray(im, dtype=np.float32)
        else:
            gray = np.asarray(im.convert("RGB"), dtype=np.float32).sum(axis=2)
    return gray, to_display(gray), exif


def to_display(gray, lo=1.0, hi=99.8):
    a, b = np.percentile(gray, [lo, hi])
    x = np.clip((gray - a) / max(b - a, 1e-9), 0, 1) ** 0.6
    u8 = (x * 255).astype(np.uint8)
    return np.stack([u8] * 3, axis=-1)


def read_exif(im) -> dict:
    out = {}
    try:
        raw = im.getexif()
        for k, v in raw.items():
            out[ExifTags.TAGS.get(k, str(k))] = v
        for ifd in (0x8769,):
            try:
                for k, v in raw.get_ifd(ifd).items():
                    out[ExifTags.TAGS.get(k, str(k))] = v
            except Exception:
                pass
        try:
            gps = raw.get_ifd(0x8825)
            if gps:
                out["GPSInfo"] = {ExifTags.GPSTAGS.get(k, str(k)): v for k, v in gps.items()}
        except Exception:
            pass
    except Exception:
        pass
    keep = ("DateTimeOriginal", "DateTime", "OffsetTimeOriginal", "ExposureTime", "FocalLength",
            "FocalLengthIn35mmFilm", "Model", "Make", "ISOSpeedRatings", "GPSInfo")
    res = {}
    for k in keep:
        if k in out:
            v = out[k]
            res[k] = {kk: _to_plain(vv) for kk, vv in v.items()} if isinstance(v, dict) else _to_plain(v)
    return res


def _to_plain(v):
    if isinstance(v, tuple):
        return [_to_plain(x) for x in v]
    try:
        return float(v) if not isinstance(v, (str, bytes)) else (v.decode(errors="ignore") if isinstance(v, bytes) else v)
    except Exception:
        return str(v)


def _num(x) -> float:
    """Parse an EXIF number given as float, '1/30', or '(1, 30)' without evaluating code."""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    m = re.fullmatch(r"\(?\s*(-?\d+(?:\.\d+)?)\s*[,/]\s*(-?\d+(?:\.\d+)?)\s*\)?", s)
    if m:
        return float(Fraction(m.group(1)) / Fraction(m.group(2)))
    return float(s)


def exif_hints(exif: dict) -> dict:
    """Best-effort claimed time / place / exposure from EXIF. The solver never uses these unless
    you explicitly choose to check the EXIF claim."""
    from datetime import datetime, timedelta, timezone
    hints = {}
    dt = exif.get("DateTimeOriginal") or exif.get("DateTime")
    if isinstance(dt, str):
        try:
            d = datetime.strptime(dt.strip(), "%Y:%m:%d %H:%M:%S")
            off = exif.get("OffsetTimeOriginal")
            if isinstance(off, str) and re.fullmatch(r"[+-]\d\d:\d\d", off.strip()):
                sign = 1 if off.strip()[0] == "+" else -1
                d = d - sign * timedelta(hours=int(off[1:3]), minutes=int(off[4:6]))
                hints["time_utc"] = d.replace(tzinfo=timezone.utc)
            else:
                hints["time_local_naive"] = d
        except ValueError:
            pass
    try:
        if exif.get("ExposureTime") is not None:
            hints["exposure_s"] = _num(exif["ExposureTime"])
    except (ValueError, ZeroDivisionError):
        pass
    try:
        f35 = exif.get("FocalLengthIn35mmFilm")
        if f35 is not None and _num(f35) > 0:
            import math
            hints["fov_x_deg"] = math.degrees(2 * math.atan(18.0 / _num(f35)))
    except (ValueError, ZeroDivisionError):
        pass
    gps = exif.get("GPSInfo") or {}
    try:
        def dms(val):
            parts = val if isinstance(val, list) else re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", str(val))
            p = [_num(x) for x in parts]
            return p[0] + p[1] / 60 + p[2] / 3600
        lat = dms(gps["GPSLatitude"]) * (1 if str(gps.get("GPSLatitudeRef", "N")).startswith("N") else -1)
        lon = dms(gps["GPSLongitude"]) * (1 if str(gps.get("GPSLongitudeRef", "E")).startswith("E") else -1)
        hints["lat"], hints["lon"] = lat, lon
    except (KeyError, IndexError, ValueError, ZeroDivisionError):
        pass
    return hints
