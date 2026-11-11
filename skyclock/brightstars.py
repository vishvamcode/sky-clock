"""Bright catalogue stars, for telling a planet from a star the plate solver did not match.

Two sources, used together:

* a built-in table of 115 named bright stars, most of the sky's stars brighter than magnitude 2.5 (Hipparcos
  J2000 positions, as tabulated in PyEphem's ``stars.py``). Tycho-2 is incomplete for the brightest stars, and
  the astrometry.net index files hold only a subset of it: Sirius, Canopus, Vega, Polaris and a dozen more of
  the 2.5-magnitude stars are missing there, and they are the ones most easily mistaken for a planet;
* the astrometry.net Tycho-2 index files, when installed, queried with ``query-starkd`` down to
  magnitude 4 (Tycho V_T).

Proper motion is ignored: the fastest of these stars moves well under 0.01 deg in 25 years.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess

import numpy as np

from .geometry import radec_to_unit, unit_to_radec

# (name, RA deg, Dec deg, V mag), J2000
BRIGHT = [
    ("Sirius", 101.2872, -16.7161, -1.44),
    ("Canopus", 95.9880, -52.6957, -0.62),
    ("Arcturus", 213.9153, 19.1824, -0.05),
    ("Rigil Kentaurus", 219.9021, -60.8340, -0.01),
    ("Vega", 279.2347, 38.7837, 0.03),
    ("Capella", 79.1723, 45.9980, 0.08),
    ("Rigel", 78.6345, -8.2016, 0.18),
    ("Procyon", 114.8255, 5.2250, 0.40),
    ("Achernar", 24.4285, -57.2368, 0.45),
    ("Betelgeuse", 88.7929, 7.4071, 0.45),
    ("Agena", 210.9559, -60.3730, 0.61),
    ("Hadar", 210.9559, -60.3730, 0.61),
    ("Altair", 297.6958, 8.8683, 0.76),
    ("Acrux", 186.6496, -63.0991, 0.77),
    ("Aldebaran", 68.9802, 16.5093, 0.87),
    ("Spica", 201.2982, -11.1613, 0.98),
    ("Antares", 247.3519, -26.4320, 1.06),
    ("Pollux", 116.3290, 28.0262, 1.16),
    ("Fomalhaut", 344.4127, -29.6222, 1.17),
    ("Formalhaut", 344.4127, -29.6222, 1.17),
    ("Deneb", 310.3580, 45.2803, 1.25),
    ("Mimosa", 191.9303, -59.6888, 1.25),
    ("Regulus", 152.0930, 11.9672, 1.36),
    ("Adara", 104.6565, -28.9721, 1.50),
    ("Adhara", 104.6565, -28.9721, 1.50),
    ("Castor", 113.6494, 31.8883, 1.58),
    ("Gacrux", 187.7915, -57.1132, 1.59),
    ("Shaula", 263.4022, -37.1038, 1.62),
    ("Bellatrix", 81.2828, 6.3497, 1.64),
    ("Elnath", 81.5730, 28.6075, 1.65),
    ("Miaplacidus", 138.2999, -69.7172, 1.67),
    ("Alnilam", 84.0534, -1.2019, 1.69),
    ("Alnair", 332.0583, -46.9610, 1.73),
    ("Alnitak", 85.1897, -1.9426, 1.74),
    ("Alioth", 193.5073, 55.9598, 1.76),
    ("Kaus Australis", 276.0430, -34.3846, 1.79),
    ("Mirfak", 51.0807, 49.8612, 1.79),
    ("Dubhe", 165.9320, 61.7510, 1.81),
    ("Wezen", 107.0979, -26.3932, 1.83),
    ("Alcaid", 206.8852, 49.3133, 1.85),
    ("Alkaid", 206.8852, 49.3133, 1.85),
    ("Avior", 125.6285, -59.5095, 1.86),
    ("Menkalinan", 89.8822, 44.9474, 1.90),
    ("Atria", 252.1662, -69.0277, 1.91),
    ("Alhena", 99.4279, 16.3993, 1.93),
    ("Peacock", 306.4119, -56.7351, 1.94),
    ("Polaris", 37.9545, 89.2641, 1.97),
    ("Mirzam", 95.6749, -17.9559, 1.98),
    ("Alphard", 141.8968, -8.6586, 1.99),
    ("Algieba", 154.9931, 19.8415, 2.01),
    ("Hamal", 31.7934, 23.4624, 2.01),
    ("Diphda", 10.8974, -17.9866, 2.04),
    ("Nunki", 283.8164, -26.2967, 2.05),
    ("Menkent", 211.6706, -36.3700, 2.06),
    ("Alpheratz", 2.0969, 29.0904, 2.07),
    ("Kochab", 222.6764, 74.1555, 2.07),
    ("Mirach", 17.4330, 35.6206, 2.07),
    ("Saiph", 86.9391, -9.6696, 2.07),
    ("Sirrah", 2.0969, 29.0904, 2.07),
    ("Rasalhague", 263.7336, 12.5600, 2.08),
    ("Algol", 47.0422, 40.9556, 2.09),
    ("Almach", 30.9748, 42.3297, 2.10),
    ("Denebola", 177.2649, 14.5721, 2.14),
    ("Naos", 120.8960, -40.0031, 2.21),
    ("Alphecca", 233.6720, 26.7147, 2.22),
    ("Mizar", 200.9814, 54.9254, 2.23),
    ("Sadr", 305.5571, 40.2567, 2.23),
    ("Suhail", 136.9990, -43.4326, 2.23),
    ("Eltanin", 269.1515, 51.4889, 2.24),
    ("Etamin", 269.1515, 51.4889, 2.24),
    ("Schedar", 10.1268, 56.5373, 2.24),
    ("Mintaka", 83.0017, -0.2991, 2.25),
    ("Caph", 2.2945, 59.1498, 2.28),
    ("Merak", 165.4603, 56.3824, 2.34),
    ("Izar", 221.2467, 27.0742, 2.35),
    ("Enif", 326.0465, 9.8750, 2.38),
    ("Ankaa", 6.5710, -42.3060, 2.40),
    ("Phecda", 178.4577, 53.6948, 2.41),
    ("Sabik", 257.5945, -15.7249, 2.43),
    ("Scheat", 345.9436, 28.0828, 2.44),
    ("Alderamin", 319.6449, 62.5856, 2.45),
    ("Markab", 346.1902, 15.2053, 2.49),
    ("Menkar", 45.5699, 4.0897, 2.54),
    ("Arneb", 83.1826, -17.8223, 2.58),
    ("Gienah", 183.9515, -17.5419, 2.58),
    ("Gienah Corvi", 183.9515, -17.5419, 2.58),
    ("Unukalhai", 236.0670, 6.4256, 2.63),
    ("Tarazed", 296.5649, 10.6133, 2.72),
    ("Zubenelgenubi", 222.7196, -16.0418, 2.75),
    ("Cebalrai", 265.8681, 4.5673, 2.76),
    ("Rasalgethi", 258.6619, 14.3903, 2.78),
    ("Nihal", 82.0613, -20.7594, 2.81),
    ("Algenib", 3.3090, 15.1836, 2.83),
    ("Alcyone", 56.8712, 24.1051, 2.85),
    ("Vindemiatrix", 195.5442, 10.9592, 2.85),
    ("Acamar", 44.5653, -40.3047, 2.88),
    ("Sadalmelik", 331.4460, -0.3199, 2.95),
    ("Zaurak", 59.5074, -13.5085, 2.97),
    ("Minkar", 182.5312, -22.6198, 3.02),
    ("Albereo", 292.6803, 27.9597, 3.05),
    ("Albireo", 292.6803, 27.9597, 3.05),
    ("Alfirk", 322.1650, 70.5607, 3.23),
    ("Sulafat", 284.7359, 32.6896, 3.25),
    ("Megrez", 183.8565, 57.0326, 3.32),
    ("Sheliak", 282.5200, 33.3627, 3.52),
    ("Atlas", 57.2906, 24.0534, 3.62),
    ("Thuban", 211.0973, 64.3759, 3.67),
    ("Alshain", 298.8283, 6.4068, 3.71),
    ("Electra", 56.2189, 24.1133, 3.72),
    ("Maia", 56.4567, 24.3677, 3.87),
    ("Arkab Prior", 290.6596, -44.4590, 3.96),
    ("Rukbat", 290.9716, -40.6159, 3.96),
    ("Alcor", 201.3064, 54.9880, 3.99),
    ("Merope", 56.5816, 23.9484, 4.14),
    ("Arkab Posterior", 290.8047, -44.7998, 4.27),
    ("Taygeta", 56.3021, 24.4673, 4.30),
]

INDEX_DIRS = ("/usr/share/astrometry", "/usr/local/astrometry/data", "/usr/local/share/astrometry",
              "/opt/homebrew/share/astrometry", "/opt/homebrew/Cellar/astrometry-net")


def _index_file():
    """The widest-scale Tycho-2 index file (it holds every Tycho-2 star), or None."""
    dirs = [os.environ["SKYCLOCK_INDEX_DIR"]] if os.environ.get("SKYCLOCK_INDEX_DIR") else []
    for d in dirs + list(INDEX_DIRS):
        hits = sorted(glob.glob(os.path.join(d, "**", "index-tycho2-1[0-9]*.fits"), recursive=True))
        if hits:
            return hits[0]
    return None


def index_available():
    return shutil.which("query-starkd") is not None and _index_file() is not None


def _query_index(ra, dec, radius_deg, max_mag):
    exe, idx = shutil.which("query-starkd"), _index_file()
    if exe is None or idx is None:
        return np.zeros((0, 3))
    try:
        out = subprocess.run([exe, "-T", "-r", f"{ra:.6f}", "-d", f"{dec:.6f}", "-R", f"{radius_deg:.4f}", idx],
                             capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return np.zeros((0, 3))
    rows = []
    for line in out.splitlines():
        p = line.split(",")
        if len(p) != 3:
            continue
        try:
            r, d, m = (float(v) for v in p)
        except ValueError:
            continue  # header lines
        if m <= max_mag:
            rows.append((r, d, m))
    return np.array(rows).reshape(-1, 3)


def catalogue_stars(ra, dec, radius_deg, max_mag=4.0):
    """(n, 3) array of RA, Dec (deg) and magnitude of the stars within ``radius_deg`` of (ra, dec): the
    built-in bright list plus, when installed, the Tycho-2 index down to ``max_mag``."""
    c = radec_to_unit(np.array([ra]), np.array([dec]))[0]
    b = np.array([r[1:] for r in BRIGHT])
    u = radec_to_unit(b[:, 0], b[:, 1])
    keep = b[np.degrees(np.arccos(np.clip(u @ c, -1, 1))) <= radius_deg]
    keep = keep[keep[:, 2] <= max_mag]
    return np.vstack([keep, _query_index(ra, dec, radius_deg, max_mag)])


def near_star_mask(plate, xy, max_mag=4.0, tol_px=None):
    """For each pixel position in ``xy`` (n, 2): True when a catalogue star brighter than ``max_mag`` lies
    within ``tol_px`` (default: max(4 px, 3 x plate RMS)). Positions are compared on the sky."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if not len(xy):
        return np.zeros(0, bool)
    H, W = plate.shape
    corners = plate.pix2unit(np.array([0, W, 0, W, W / 2]), np.array([0, 0, H, H, H / 2]))
    centre = corners[-1]
    radius = float(np.degrees(np.max(np.arccos(np.clip(corners[:-1] @ centre, -1, 1))))) + 1.0
    ra, dec = unit_to_radec(centre[None, :])
    stars = catalogue_stars(float(np.ravel(ra)[0]), float(np.ravel(dec)[0]), min(radius, 90.0), max_mag)
    if not len(stars):
        return np.zeros(len(xy), bool)
    rms = plate.rms_px if getattr(plate, "rms_px", None) is not None else 1.0
    tol = tol_px if tol_px is not None else max(4.0, 3.0 * float(rms))
    su = radec_to_unit(stars[:, 0], stars[:, 1])
    pu = plate.pix2unit(xy[:, 0], xy[:, 1])
    scale = np.array([plate.pixel_scale_rad(x, y) for x, y in xy])
    sep = np.arccos(np.clip(pu @ su.T, -1, 1)).min(axis=1)
    return sep < tol * scale
