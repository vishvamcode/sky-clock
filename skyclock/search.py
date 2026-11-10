"""Candidate generation when the time and/or the place are unknown.

Geometry used
-------------
A streak's great circle (pole n) and the observer o define a plane that contains the satellite's
trajectory during the exposure. Hence, in the Earth-fixed frame at mid-exposure time t:

  (motion)   n(t) . v_sat(t) ~ 0        -- depends only on the satellite and t, not on o
  (plane)    n(t) . o = n(t) . r_sat(t)  -- the observer lies on a plane through the satellite

Step 1 finds, for every streak and catalogue object, the instants where the motion condition holds
(roots of a slowly varying function, sampled every ``step`` seconds). Each root comes with a time
window whose width follows from the uncertainty of the streak's orientation.

Step 2 turns every root into an *observer segment*: the ground points from which the satellite, at
that time, appears on the observed arc (rays traced back from the satellite along the arc), keeping
only points where the satellite is above the horizon, within range, and (optionally) where the Sun is
down. A single streak therefore constrains the observer to a set of short curves.

Step 3 hashes the segment points into ~100 km cells; two streaks whose segments share a cell at
overlapping times give a candidate. The candidate is refined by intersecting the two planes (a line)
with the Earth's surface and choosing the time that puts both satellites on their observed arcs.

Step 4 clusters candidates in (place, time); a cluster supported by many streak pairs is a strong
candidate. Final ranking is done by the full fit in ``solve.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (EARTH_ROT, Epoch, line_ellipsoid, ray_ellipsoid, sun_gcrs, sunlit, unit)

MU = 398600.4418
OMEGA = np.array([0.0, 0.0, EARTH_ROT])
CELL_KM = 100.0


@dataclass
class Roots:
    sat: np.ndarray
    t: np.ndarray
    w: np.ndarray
    r: np.ndarray
    v: np.ndarray

    def __len__(self):
        return len(self.sat)

    def take(self, m):
        return Roots(self.sat[m], self.t[m], self.w[m], self.r[m], self.v[m])


def motion_tolerance(ob):
    return 3.0 * ob.orient_sigma + 2e-3


def motion_roots(obs_list, catalog, epoch: Epoch, t_lo, t_hi, step=10.0, chunk_t=90, w_max=60.0,
                 require_sunlit=True, progress=None):
    """Roots of n_i(t).v_hat_k(t) for every streak i and catalogue object k in [t_lo, t_hi]."""
    times = np.arange(t_lo, t_hi + step, step)
    acc = [dict(sat=[], t=[], w=[], r=[], v=[]) for _ in obs_list]
    tols = [motion_tolerance(ob) for ob in obs_list]
    n_chunks = int(np.ceil((len(times) - 1) / chunk_t))
    for ci, c0 in enumerate(range(0, len(times) - 1, chunk_t)):
        tc = times[c0:c0 + chunk_t + 1]
        r, v, ok = catalog.propagate(epoch, tc)
        sp = np.linalg.norm(v, axis=2)
        vh = v / np.where(sp > 0, sp, 1)[..., None]
        sun_i = epoch.gcrs_to_itrs_vecs(sun_gcrs(epoch, tc), tc)
        for i, ob in enumerate(obs_list):
            n_t = epoch.gcrs_to_itrs_vecs(ob.pole[None, :], tc)
            g = np.einsum("ntk,tk->nt", vh, n_t)
            g = np.where(ok, g, np.nan)
            with np.errstate(invalid="ignore"):
                sc = g[:, :-1] * g[:, 1:] < 0
            ni, ti = np.nonzero(sc)
            if len(ni) == 0:
                continue
            g0, g1 = g[ni, ti], g[ni, ti + 1]
            f = g0 / (g0 - g1)
            tr = tc[ti] + f * step
            slope = (g1 - g0) / step
            w = np.minimum(tols[i] / np.maximum(np.abs(slope), 1e-12), w_max)
            rr = r[ni, ti] + f[:, None] * (r[ni, ti + 1] - r[ni, ti])
            vv = v[ni, ti] + f[:, None] * (v[ni, ti + 1] - v[ni, ti])
            keep = np.ones(len(ni), bool)
            if require_sunlit:
                keep = sunlit(rr, sun_i[ti], margin_km=60)
            a = acc[i]
            a["sat"].append(ni[keep]); a["t"].append(tr[keep]); a["w"].append(w[keep])
            a["r"].append(rr[keep]); a["v"].append(vv[keep])
        if progress:
            progress((ci + 1) / n_chunks, f"motion search {ci + 1}/{n_chunks}")
    out = []
    for a in acc:
        if a["sat"]:
            out.append(Roots(np.concatenate(a["sat"]), np.concatenate(a["t"]), np.concatenate(a["w"]),
                             np.concatenate(a["r"]), np.concatenate(a["v"])))
        else:
            z = np.zeros((0, 3))
            out.append(Roots(np.zeros(0, int), np.zeros(0), np.zeros(0), z, z))
    return out


def propagate_local(r0, v0, dt):
    """Second-order Taylor step in the rotating frame (gravity, Coriolis, centrifugal).

    Error < ~0.5 km for |dt| <= 60 s, adequate for candidate generation (the final fit uses SGP4)."""
    dt = np.asarray(dt, float)[..., None]
    rn = np.linalg.norm(r0, axis=-1, keepdims=True)
    a = -MU * r0 / rn**3 - 2 * np.cross(OMEGA, v0) - np.cross(OMEGA, np.cross(OMEGA, r0))
    return r0 + v0 * dt + 0.5 * a * dt**2


def _arc_bounds(ob, omega, w, T_max):
    """Allowed arc coordinate of the satellite at a time ``w`` s away from mid-exposure."""
    a = ob.half_len
    t1, t2 = ob.trusted_ends
    base = 0.1 * a + np.radians(0.3)
    lo = -base - (0.0 if t1 else a + omega * T_max / 2) - omega * w
    hi = base + (0.0 if t2 else a + omega * T_max / 2) + omega * w
    return lo, hi


def _sun_alt_ok(epoch, t, pts, sun_max_alt_deg):
    if sun_max_alt_deg is None:
        return np.ones(pts.shape[:-1], bool)
    sun_i = epoch.gcrs_to_itrs_vecs(sun_gcrs(epoch, t), t)
    up = unit(pts)
    return np.sum(sun_i * up, axis=-1) < np.sin(np.radians(sun_max_alt_deg))


def segments(ob, roots: Roots, epoch: Epoch, T_max=30.0, h_km=0.0, el_min_deg=8.0, max_range_km=3000.0,
             sun_max_alt_deg=-3.0, ds_deg=0.5, s_max_deg=40.0, chunk=6000, region=None):
    """Observer segments of each root. Returns (points (M,3), root index (M,))."""
    s = np.radians(np.arange(-s_max_deg, s_max_deg + ds_deg, ds_deg))
    dg = ob.arc_point(s)  # (S,3) GCRS
    sin_el = np.sin(np.radians(el_min_deg))
    P, R = [], []
    for c0 in range(0, len(roots), chunk):
        rt = roots.take(slice(c0, c0 + chunk))
        n = len(rt)
        if n == 0:
            continue
        d = epoch.gcrs_to_itrs_vecs(dg[None, :, :], rt.t[:, None])  # (n,S,3)
        org = np.broadcast_to(rt.r[:, None, :], d.shape)
        pts, rho, hit = ray_ellipsoid(org, -d, h_km)
        up = unit(pts)
        el_ok = np.sum(d * up, -1) > sin_el
        v = rt.v[:, None, :]
        vperp = np.linalg.norm(v - np.sum(v * d, -1, keepdims=True) * d, axis=-1)
        omega = vperp / np.maximum(rho, 1.0)
        lo, hi = _arc_bounds(ob, omega, rt.w[:, None], T_max)
        keep = hit & el_ok & (rho < max_range_km) & (s[None, :] >= lo) & (s[None, :] <= hi)
        sun_i = epoch.gcrs_to_itrs_vecs(sun_gcrs(epoch, rt.t), rt.t)  # (n,3)
        if sun_max_alt_deg is not None:
            keep &= np.einsum("nsk,nk->ns", up, sun_i) < np.sin(np.radians(sun_max_alt_deg))
        if region is not None:
            c, rad = region
            keep &= np.linalg.norm(pts - c, axis=-1) < rad
        ii, jj = np.nonzero(keep)
        P.append(pts[ii, jj])
        R.append(ii + c0)
    if not P:
        return np.zeros((0, 3)), np.zeros(0, int)
    return np.concatenate(P), np.concatenate(R)


def cell_keys(pts, cell=CELL_KM):
    q = np.floor(pts / cell).astype(np.int64) + 128
    return (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]


def _unique_pairs(keys, roots_idx):
    kr = np.unique(np.stack([keys, roots_idx], 1), axis=0)
    return kr[:, 0], kr[:, 1]


def join_cells(ka, ra, kb, rb, max_pairs=3_000_000):
    """All (root_a, root_b) that share a cell. Sort-merge join."""
    oa, ob_ = np.argsort(ka, kind="stable"), np.argsort(kb, kind="stable")
    ka, ra, kb, rb = ka[oa], ra[oa], kb[ob_], rb[ob_]
    common = np.intersect1d(ka, kb)
    if len(common) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    la, ha = np.searchsorted(ka, common, "left"), np.searchsorted(ka, common, "right")
    lb, hb = np.searchsorted(kb, common, "left"), np.searchsorted(kb, common, "right")
    na, nb = ha - la, hb - lb
    tot = int(np.sum(na * nb))
    if tot > max_pairs:
        # keep the densest information: sub-sample cells uniformly
        keep = np.random.default_rng(0).random(len(common)) < max_pairs / tot
        la, ha, lb, hb, na, nb = la[keep], ha[keep], lb[keep], hb[keep], na[keep], nb[keep]
    A, B = [], []
    for a0, a1, b0, b1 in zip(la, ha, lb, hb):
        aa = ra[a0:a1]
        bb = rb[b0:b1]
        A.append(np.repeat(aa, len(bb)))
        B.append(np.tile(bb, len(aa)))
    A, B = np.concatenate(A), np.concatenate(B)
    u = np.unique(np.stack([A, B], 1), axis=0)
    return u[:, 0], u[:, 1]


def refine_pairs(oa, ob, Ra: Roots, Rb: Roots, ia, ib, epoch: Epoch, T_max=30.0, h_km=0.0, el_min_deg=8.0,
                 max_range_km=3000.0, sun_max_alt_deg=-3.0, K=21, chunk=20000):
    """For root pairs (ia[k] of streak a, ib[k] of streak b) find the time and place where both
    planes meet on the ground and both satellites sit on their arcs. Returns dict of arrays."""
    out = dict(ia=[], ib=[], t=[], o=[], score=[])
    sin_el = np.sin(np.radians(el_min_deg))
    for c0 in range(0, len(ia), chunk):
        A = Ra.take(ia[c0:c0 + chunk])
        B = Rb.take(ib[c0:c0 + chunk])
        lo = np.maximum(A.t - A.w, B.t - B.w) - 1.0
        hi = np.minimum(A.t + A.w, B.t + B.w) + 1.0
        bad = lo > hi
        mid = 0.5 * (A.t + B.t)
        lo = np.where(bad, mid - 1, lo)
        hi = np.where(bad, mid + 1, hi)
        tau = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, K)[None, :]  # (P,K)
        ra = propagate_local(A.r[:, None, :], A.v[:, None, :], tau - A.t[:, None])
        rb = propagate_local(B.r[:, None, :], B.v[:, None, :], tau - B.t[:, None])
        na = epoch.gcrs_to_itrs_vecs(oa.pole[None, None, :], tau)
        nb = epoch.gcrs_to_itrs_vecs(ob.pole[None, None, :], tau)
        ca = np.sum(na * ra, -1)
        cb = np.sum(nb * rb, -1)
        ell = np.cross(na, nb)
        l2 = np.sum(ell * ell, -1)
        l2s = np.where(l2 > 1e-10, l2, 1e-10)
        p0 = (ca[..., None] * np.cross(nb, ell) + cb[..., None] * np.cross(ell, na)) / l2s[..., None]
        pts, hit = line_ellipsoid(p0, ell, h_km)  # (P,K,2,3)
        hit &= l2 > 1e-8
        best_score = np.full(len(A.t), np.inf)
        best_t = np.zeros(len(A.t))
        best_o = np.zeros((len(A.t), 3))
        for side in range(2):
            o = pts[:, :, side, :]
            up = unit(o)
            score = np.zeros(o.shape[:2])
            ok = hit.copy()
            for ob_, rr, R_ in ((oa, ra, A), (ob, rb, B)):
                rel = rr - o
                rho = np.linalg.norm(rel, axis=-1)
                d = rel / rho[..., None]
                ok &= (np.sum(d * up, -1) > sin_el) & (rho < max_range_km)
                dg = epoch.itrs_to_gcrs_vecs(d, tau)
                s = np.arctan2(dg @ ob_.tangent, dg @ ob_.mid)
                v = R_.v[:, None, :]
                vperp = np.linalg.norm(v - np.sum(v * d, -1, keepdims=True) * d, axis=-1)
                om = vperp / np.maximum(rho, 1.0)
                l_, h_ = _arc_bounds(ob_, om, 0.0, T_max)
                tol = np.radians(0.3) + om * 1.5
                viol = np.maximum(0, np.maximum(s - h_, l_ - s)) / tol
                score += viol**2
            if sun_max_alt_deg is not None:
                ok &= _sun_alt_ok(epoch, tau, o, sun_max_alt_deg)
            score = np.where(ok, score, np.inf)
            j = np.argmin(score, axis=1)
            sc = score[np.arange(len(j)), j]
            better = sc < best_score
            best_score = np.where(better, sc, best_score)
            best_t = np.where(better, tau[np.arange(len(j)), j], best_t)
            best_o = np.where(better[:, None], o[np.arange(len(j)), j], best_o)
        good = best_score < 2.0
        out["ia"].append(ia[c0:c0 + chunk][good]); out["ib"].append(ib[c0:c0 + chunk][good])
        out["t"].append(best_t[good]); out["o"].append(best_o[good]); out["score"].append(best_score[good])
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


@dataclass
class Cluster:
    o: np.ndarray
    t: float
    assignment: dict  # obs index -> catalogue index
    support: int  # number of streak-pair candidates
    n_streaks: int


def cluster_candidates(cands, n_obs, space_km=60.0, time_s=12.0, max_clusters=40):
    """cands: list of (i, k_i, j, k_j, t, o). Connected components in scaled (o, t) space."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    if not cands:
        return []
    X = np.array([np.concatenate([c[5] / space_km, [c[4] / time_s]]) for c in cands])
    tree = cKDTree(X)
    pairs = tree.query_pairs(1.0, output_type="ndarray")
    n = len(cands)
    if len(pairs):
        G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, lab = connected_components(G, directed=False)
    else:
        lab = np.arange(n)
    clusters = []
    for L in np.unique(lab):
        mem = np.where(lab == L)[0]
        votes = {}
        for m in mem:
            i, ki, j, kj = cands[m][:4]
            votes.setdefault(i, {}).setdefault(ki, 0)
            votes[i][ki] += 1
            votes.setdefault(j, {}).setdefault(kj, 0)
            votes[j][kj] += 1
        assign = {i: max(v, key=v.get) for i, v in votes.items()}
        cons = [m for m in mem if assign[cands[m][0]] == cands[m][1] and assign[cands[m][2]] == cands[m][3]]
        cons = cons or list(mem)
        o = np.median(np.array([cands[m][5] for m in cons]), axis=0)
        t = float(np.median([cands[m][4] for m in cons]))
        clusters.append(Cluster(o, t, assign, len(mem), len(assign)))
    clusters.sort(key=lambda c: (-c.n_streaks, -c.support))
    return clusters[:max_clusters]


def blind_candidates(obs_list, catalog, epoch, t_lo, t_hi, T_max=30.0, h_km=0.0, el_min_deg=8.0,
                     max_range_km=3000.0, sun_max_alt_deg=-3.0, require_sunlit=True, region=None,
                     progress=None, max_pairs_per_streak_pair=3_000_000):
    """Full blind search. Returns (clusters, roots, segments) - segments are kept for single-streak loci."""
    def prog(f, msg):
        if progress:
            progress(f, msg)

    roots = motion_roots(obs_list, catalog, epoch, t_lo - T_max / 2, t_hi + T_max / 2,
                         require_sunlit=require_sunlit, progress=lambda f, m: prog(0.6 * f, m))
    segs = []
    for i, (ob, R) in enumerate(zip(obs_list, roots)):
        P, ri = segments(ob, R, epoch, T_max, h_km, el_min_deg, max_range_km, sun_max_alt_deg, region=region)
        segs.append((P, ri))
        prog(0.6 + 0.2 * (i + 1) / len(obs_list), f"observer segments for streak {i + 1}/{len(obs_list)}")
    cands = []
    pairs = [(i, j) for i in range(len(obs_list)) for j in range(i + 1, len(obs_list))]
    for pi, (i, j) in enumerate(pairs):
        Pi, ri = segs[i]
        Pj, rj = segs[j]
        if len(Pi) == 0 or len(Pj) == 0:
            continue
        ka, ra = _unique_pairs(cell_keys(Pi), ri)
        kb, rb = _unique_pairs(cell_keys(Pj), rj)
        A, B = join_cells(ka, ra, kb, rb, max_pairs_per_streak_pair)
        if len(A) == 0:
            continue
        Ri, Rj = roots[i], roots[j]
        dt = np.abs(Ri.t[A] - Rj.t[B])
        m = dt <= Ri.w[A] + Rj.w[B] + 2.0
        m &= Ri.sat[A] != Rj.sat[B]
        A, B = A[m], B[m]
        if len(A) == 0:
            continue
        ref = refine_pairs(obs_list[i], obs_list[j], Ri, Rj, A, B, epoch, T_max, h_km, el_min_deg,
                           max_range_km, sun_max_alt_deg)
        for a, b, t, o in zip(ref["ia"], ref["ib"], ref["t"], ref["o"]):
            cands.append((i, int(Ri.sat[int(a)]), j, int(Rj.sat[int(b)]), float(t), o))
        prog(0.8 + 0.2 * (pi + 1) / len(pairs), f"pairing streaks {pi + 1}/{len(pairs)}")
    return cluster_candidates(cands, len(obs_list)), roots, segs


def time_candidates(obs_list, catalog, epoch, o, t_lo, t_hi, T_max=30.0, sigma_loc_km=5.0,
                    require_sunlit=True, el_min_deg=5.0, progress=None, K=25):
    """Observer known: per streak, satellites whose motion root puts them on the observed arc as seen
    from ``o``. Returns clusters (in time) like blind_candidates."""
    roots = motion_roots(obs_list, catalog, epoch, t_lo - T_max / 2, t_hi + T_max / 2,
                         require_sunlit=require_sunlit, progress=progress)
    up = unit(o)
    per = []
    for i, (ob, R) in enumerate(zip(obs_list, roots)):
        if len(R) == 0:
            per.append([])
            continue
        # quick prefilter: satellite above horizon and within range at the root
        rel = R.r - o
        rho = np.linalg.norm(rel, axis=1)
        pre = (rel @ up / rho > np.sin(np.radians(el_min_deg))) & (rho < 3500)
        Rk = R.take(pre)
        if len(Rk) == 0:
            per.append([])
            continue
        tau = Rk.t[:, None] + Rk.w[:, None] * np.linspace(-1, 1, K)[None, :]
        rr = propagate_local(Rk.r[:, None, :], Rk.v[:, None, :], tau - Rk.t[:, None])
        rel = rr - o
        rho = np.linalg.norm(rel, axis=-1)
        d = rel / rho[..., None]
        dg = epoch.itrs_to_gcrs_vecs(d, tau)
        cross = dg @ ob.pole
        s = np.arctan2(dg @ ob.tangent, dg @ ob.mid)
        v = Rk.v[:, None, :]
        vperp = np.linalg.norm(v - np.sum(v * d, -1, keepdims=True) * d, axis=-1)
        om = vperp / np.maximum(rho, 1.0)
        l_, h_ = _arc_bounds(ob, om, 0.0, T_max)
        tol_s = np.radians(0.3) + om * 1.5
        tol_x = 3 * np.sqrt((sigma_loc_km / rho) ** 2 + (0.5 / rho) ** 2 + ob.sigma_offset_rad ** 2) + 3e-4
        score = (cross / tol_x) ** 2 + (np.maximum(0, np.maximum(s - h_, l_ - s)) / tol_s) ** 2
        j = np.argmin(score, axis=1)
        sc = score[np.arange(len(j)), j]
        good = sc < 2.0
        per.append([(i, int(Rk.sat[g]), float(tau[g, j[g]])) for g in np.where(good)[0]])
    # cluster in time
    flat = sorted([c for p in per for c in p], key=lambda c: c[2])
    clusters = []
    for c in flat:
        placed = False
        for cl in clusters:
            if abs(cl["t"] - c[2]) < 12.0:
                cl["members"].append(c)
                cl["t"] = float(np.median([m[2] for m in cl["members"]]))
                placed = True
                break
        if not placed:
            clusters.append(dict(t=c[2], members=[c]))
    out = []
    for cl in clusters:
        votes = {}
        for i, k, _ in cl["members"]:
            votes.setdefault(i, {}).setdefault(k, 0)
            votes[i][k] += 1
        assign = {i: max(v, key=v.get) for i, v in votes.items()}
        out.append(Cluster(np.asarray(o, float), cl["t"], assign, len(cl["members"]), len(assign)))
    out.sort(key=lambda c: (-c.n_streaks, -c.support))
    return out, roots
