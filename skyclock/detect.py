"""Star and satellite-streak detection with sub-pixel line fits.

Pipeline: robust background -> smoothed significance mask -> connected components ->
compact = stars, elongated = streak fragments -> merge collinear fragments (dashed streaks,
gaps from stars/noise) -> refine each streak on the unsmoothed residual with cross-track
profiles every few pixels -> sub-pixel centre line, endpoints, clipped/dashed flags.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")


@dataclass
class Streak:
    id: int
    p1: list  # [x, y] endpoint A, 0-based pixels (x right, y down)
    p2: list  # [x, y] endpoint B
    clipped1: bool  # endpoint A touches the frame edge (not an exposure boundary)
    clipped2: bool
    dashed: bool  # gaps along the streak (stacked frames / blinking) -> endpoints untrusted for timing
    samples: list  # [[x, y], ...] measured centre-line points
    sigma_px: float  # per-sample cross-track uncertainty
    width_px: float
    length_px: float
    snr: float
    enabled: bool = True

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Streak(**d)


@dataclass
class Star:
    x: float
    y: float
    flux: float


@dataclass
class Detection:
    streaks: list
    stars: list
    shape: tuple
    bg_sigma: float
    notes: list = field(default_factory=list)


def background(gray, block=64):
    H, W = gray.shape
    ny, nx = int(np.ceil(H / block)), int(np.ceil(W / block))
    pad = np.pad(gray, ((0, ny * block - H), (0, nx * block - W)), mode="edge")
    blocks = pad.reshape(ny, block, nx, block).transpose(0, 2, 1, 3).reshape(ny, nx, -1)
    med = np.median(blocks, axis=2)
    med = ndimage.median_filter(med, size=3, mode="nearest")
    zy, zx = H / ny, W / nx
    yy = (np.arange(H) + 0.5) / zy - 0.5
    xx = (np.arange(W) + 0.5) / zx - 0.5
    Y, X = np.meshgrid(np.clip(yy, 0, ny - 1), np.clip(xx, 0, nx - 1), indexing="ij")
    return ndimage.map_coordinates(med, [Y, X], order=1, mode="nearest").astype(np.float32)


def robust_sigma(x, n=200000, seed=0):
    x = np.asarray(x).ravel()
    if x.size > n:
        x = x[np.random.default_rng(seed).integers(0, x.size, n)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-9


def detect(gray, min_length_px=None, k_mask=2.5, star_k=6.0, max_stars=400, edge_margin=None):
    gray = np.asarray(gray, dtype=np.float32)
    H, W = gray.shape
    diag = float(np.hypot(H, W))
    min_length_px = min_length_px or max(30.0, 0.02 * diag)
    frag_len = max(10.0, 0.004 * diag)
    res = gray - background(gray)
    sig = robust_sigma(res)
    sm = ndimage.gaussian_filter(res, 1.2)
    sig_sm = robust_sigma(sm)
    mask = sm > k_mask * sig_sm
    lab, n = ndimage.label(mask)
    notes = []
    if n == 0:
        return Detection([], [], (H, W), float(sig), ["no significant pixels"])
    objs = ndimage.find_objects(lab)
    stars, frags = [], []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = lab[sl] == i
        area = int(sub.sum())
        if area < 4:
            continue
        ys, xs = np.nonzero(sub)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        w = np.clip(sm[ys, xs], 0, None) + 1e-6
        cx, cy = np.average(xs, weights=w), np.average(ys, weights=w)
        if area < 8:
            l1 = l2 = 1.0
        else:
            cov = np.cov(np.vstack([xs, ys]), aweights=w)
            ev, evec = np.linalg.eigh(cov)
            l2, l1 = max(ev[0], 1e-3), max(ev[1], 1e-3)
        L, wid = 4 * np.sqrt(l1), 4 * np.sqrt(l2)
        peak = float(sm[ys, xs].max())
        if L >= frag_len and L / wid >= 3.0:
            u = evec[:, 1] if area >= 8 else np.array([1.0, 0.0])
            t = (xs - cx) * u[0] + (ys - cy) * u[1]
            frags.append(dict(c=np.array([cx, cy]), u=u / np.linalg.norm(u), tmin=t.min(), tmax=t.max(),
                              L=L, w=wid, flux=float(w.sum())))
        elif L / wid < 2.0 and peak > star_k * sig_sm and area < 2000:
            stars.append(Star(float(cx), float(cy), float(w.sum())))
    stars.sort(key=lambda s: -s.flux)
    stars = stars[:max_stars]

    clusters = _merge_fragments(frags, diag)
    streaks = []
    edge_margin = edge_margin
    for cl in clusters:
        st = _refine(res, sig, cl, edge_margin)
        if st is None:
            continue
        if st["length"] < min_length_px or st["snr"] < 6:
            continue
        streaks.append(st)
    # drop near-duplicates (same line found twice)
    streaks.sort(key=lambda s: -s["length"])
    kept = []
    for s in streaks:
        if any(_same_line(s, k) for k in kept):
            continue
        kept.append(s)
    out = []
    for i, s in enumerate(kept):
        out.append(Streak(i, s["p1"], s["p2"], s["c1"], s["c2"], s["dashed"], s["samples"], s["sigma"],
                          s["width"], s["length"], s["snr"]))
    if not stars:
        notes.append("no stars found: plate solving will fail")
    return Detection(out, stars, (H, W), float(sig), notes)


def _same_line(a, b):
    ua = np.subtract(a["p2"], a["p1"]); ua = ua / np.linalg.norm(ua)
    ub = np.subtract(b["p2"], b["p1"]); ub = ub / np.linalg.norm(ub)
    if abs(ua @ ub) < np.cos(np.radians(3)):
        return False
    nb = np.array([-ub[1], ub[0]])
    mid = (np.array(a["p1"]) + np.array(a["p2"])) / 2
    return abs((mid - b["p1"]) @ nb) < 4.0 and abs((mid - b["p1"]) @ ub) < b["length"] + 5


def _merge_fragments(frags, diag):
    n = len(frags)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = frags[i], frags[j]
            if abs(a["u"] @ b["u"]) < np.cos(np.radians(3.0)):
                continue
            na = np.array([-a["u"][1], a["u"][0]])
            nb = np.array([-b["u"][1], b["u"][0]])
            tol = 2.5 + 0.5 * max(a["w"], b["w"])
            if abs((b["c"] - a["c"]) @ na) > tol or abs((a["c"] - b["c"]) @ nb) > tol:
                continue
            # gap along the line
            tb = (b["c"] - a["c"]) @ a["u"]
            lo_b, hi_b = tb + b["tmin"] * np.sign(a["u"] @ b["u"]), tb + b["tmax"] * np.sign(a["u"] @ b["u"])
            lo_b, hi_b = min(lo_b, hi_b), max(lo_b, hi_b)
            gap = max(lo_b - a["tmax"], a["tmin"] - hi_b, 0)
            if gap <= max(40.0, 0.08 * diag):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(frags[i])
    clusters = []
    for g in groups.values():
        if sum(f["L"] for f in g) < 15:
            continue
        pts, wts = [], []
        for f in g:
            for t in np.linspace(f["tmin"], f["tmax"], 8):
                pts.append(f["c"] + t * f["u"]); wts.append(f["flux"] / 8)
        pts = np.array(pts); wts = np.array(wts)
        c = np.average(pts, axis=0, weights=wts)
        cov = np.cov((pts - c).T, aweights=wts)
        ev, evec = np.linalg.eigh(cov)
        u = evec[:, 1]
        t = (pts - c) @ u
        clusters.append(dict(c=c, u=u / np.linalg.norm(u), tmin=t.min(), tmax=t.max(),
                             w=float(np.median([f["w"] for f in g]))))
    return clusters


def _profiles(res, c, u, ts, hw, nsub=3, step=2.0):
    nrm = np.array([-u[1], u[0]])
    offs = np.arange(-hw, hw + 1, 1.0)
    sub = np.linspace(-step / 2, step / 2, nsub)
    T, O, S = np.meshgrid(ts, offs, sub, indexing="ij")
    X = c[0] + (T + S) * u[0] + O * nrm[0]
    Y = c[1] + (T + S) * u[1] + O * nrm[1]
    vals = ndimage.map_coordinates(res, [Y, X], order=1, mode="constant", cval=np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return offs, np.nanmean(vals, axis=2)  # (nt, noff)


def _refine(res, sig, cl, edge_margin):
    H, W = res.shape
    c, u = cl["c"].astype(float), cl["u"].astype(float)
    if u[0] < 0 or (u[0] == 0 and u[1] < 0):
        u = -u
    hw = int(max(6, np.ceil(3 * cl["w"])))
    step = 2.0
    tmin, tmax = cl["tmin"] - 25, cl["tmax"] + 25
    core_hw = max(2.0, 1.5 * cl["w"])
    for _ in range(3):
        ts = np.arange(tmin, tmax + step, step)
        offs, prof = _profiles(res, c, u, ts, hw, step=step)
        valid = np.isfinite(prof).all(axis=1)
        if valid.sum() < 5:
            return None
        tails = np.concatenate([prof[:, :3], prof[:, -3:]], axis=1)
        bg = np.nanmedian(tails, axis=1)
        p = prof - bg[:, None]
        core = np.abs(offs) <= core_hw
        S = np.nansum(np.where(core[None, :], p, 0), axis=1)
        n_core = core.sum()
        sigS = sig * np.sqrt(n_core / 3.0)
        on = valid & (S > 3.0 * sigS)
        # edge-like (horizon/tree) profiles: tails on the two sides differ strongly
        left, right = np.nanmedian(prof[:, :3], axis=1), np.nanmedian(prof[:, -3:], axis=1)
        edge_like = np.abs(left - right) > 0.6 * np.maximum(S / max(n_core, 1), 1e-9)
        on &= ~edge_like
        if on.sum() < 5:
            return None
        pos = np.clip(p, 0, None) * core[None, :]
        cen = (pos * offs[None, :]).sum(axis=1) / np.maximum(pos.sum(axis=1), 1e-9)
        # streak profile width from the median profile of 'on' stations (robust to stars)
        mp = np.nanmedian(p[on], axis=0)
        mp = np.clip(mp - np.median(mp[np.abs(offs) > 0.6 * hw]), 0, None)
        if mp.sum() > 0:
            pk = offs[np.argmax(mp)]
            sel_w = np.abs(offs - pk) <= max(3.0, core_hw)
            sd = np.sqrt(np.sum(mp[sel_w] * (offs[sel_w] - pk) ** 2) / max(mp[sel_w].sum(), 1e-9))
            core_hw = float(np.clip(2.5 * sd, 2.0, hw - 2))
        tt, cc = ts[on], cen[on]
        # robust line fit: offset = a + b*t
        good = np.ones_like(tt, bool)
        for _k in range(3):
            A = np.vstack([np.ones(good.sum()), tt[good]]).T
            coef, *_ = np.linalg.lstsq(A, cc[good], rcond=None)
            r = cc - (coef[0] + coef[1] * tt)
            s = 1.4826 * np.median(np.abs(r[good])) + 1e-3
            good = np.abs(r) < 3.5 * s + 0.3
        nrm = np.array([-u[1], u[0]])
        ang = np.arctan(coef[1])
        c = c + coef[0] * nrm
        u = np.cos(ang) * u + np.sin(ang) * nrm
        u /= np.linalg.norm(u)
        # solid runs (>= 4 stations with gaps <= 2), then chain runs across gaps <= 20 stations
        # (bridges stars and short dropouts without letting isolated noise stations extend the ends)
        idx = np.where(on)[0]
        runs, start = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 2:
                runs.append((start, a)); start = b
        runs.append((start, idx[-1]))
        solid = [r for r in runs if r[1] - r[0] >= 3] or [max(runs, key=lambda ab: ab[1] - ab[0])]
        chains = [list(solid[0])]
        for r0, r1 in solid[1:]:
            if r0 - chains[-1][1] <= 20:
                chains[-1][1] = r1
            else:
                chains.append([r0, r1])
        i0, i1 = max(chains, key=lambda ab: ab[1] - ab[0])
        tmin, tmax = ts[i0] - 25, ts[i1] + 25
    # final products
    seg = slice(i0, i1 + 1)
    on_seg = on[seg]
    gaps = np.diff(np.where(on_seg)[0])
    dashed = bool(np.sum(gaps > 2) >= 2)
    Ssm = ndimage.uniform_filter1d(np.where(valid, S, 0.0), 3)
    plateau = np.median(Ssm[seg][on_seg]) if on_seg.any() else 0
    half = 0.5 * plateau

    def edge_pos(i, direction):
        j = i
        while 0 <= j + direction < len(ts) and Ssm[j + direction] > half:
            j += direction
        k = j + direction
        if 0 <= k < len(ts) and Ssm[j] != Ssm[k]:
            f = (Ssm[j] - half) / (Ssm[j] - Ssm[k])
            return ts[j] + direction * step * np.clip(f, 0, 1)
        return ts[j]

    ta, tb = edge_pos(i0, -1), edge_pos(i1, +1)
    p1 = c + ta * u
    p2 = c + tb * u
    em = edge_margin if edge_margin is not None else hw + 3.0

    def clipped(pp):
        return bool(pp[0] < em or pp[1] < em or pp[0] > W - 1 - em or pp[1] > H - 1 - em)

    nrm = np.array([-u[1], u[0]])
    sel = np.where(on)[0]
    sel = sel[(sel >= i0) & (sel <= i1)]
    # re-measure centroids relative to the final line
    offs, prof = _profiles(res, c, u, ts[sel], hw, step=step)
    bg = np.nanmedian(np.concatenate([prof[:, :3], prof[:, -3:]], axis=1), axis=1)
    p = np.clip(prof - bg[:, None], 0, None)
    core = np.abs(offs) <= core_hw
    p = p * core[None, :]
    cen = (p * offs[None, :]).sum(axis=1) / np.maximum(p.sum(axis=1), 1e-9)
    tsel = ts[sel]
    A = np.vstack([np.ones(len(tsel)), tsel]).T
    coef2, *_ = np.linalg.lstsq(A, cen, rcond=None)
    r = cen - A @ coef2
    mad = 1.4826 * np.median(np.abs(r)) + 1e-3
    keep = np.abs(r) < 4 * mad + 0.2  # drop samples contaminated by stars or crossings
    pts = c[None, :] + tsel[keep][:, None] * u[None, :] + cen[keep][:, None] * nrm[None, :]
    sigma = float(max(0.15, 1.4826 * np.median(np.abs(r[keep]))))
    mp = np.median(p[keep], axis=0) if keep.any() else np.zeros_like(offs)
    width = float(np.sqrt(max(np.sum(mp * offs ** 2) / max(mp.sum(), 1e-9), 0.25)) * 2.355)
    if len(pts) > 200:
        pts = pts[np.linspace(0, len(pts) - 1, 200).astype(int)]
    snr = float(S[sel].sum() / (sigS * np.sqrt(max(len(sel), 1))))
    return dict(p1=[float(p1[0]), float(p1[1])], p2=[float(p2[0]), float(p2[1])], c1=clipped(p1), c2=clipped(p2),
                dashed=dashed, samples=pts.tolist(), sigma=sigma, width=width, length=float(tb - ta), snr=snr)
