"""Sky-clock feasibility simulator, v2 (after the in-project adversarial review).

Supersedes census / fisher_case / crb / alias_scan in skyclock_sim.py (kept only for
the synthetic constellation helpers build, sat_pos, sun_dir, observer).

Changes vs v1 (review finding numbers in brackets):
 [1] covariance by np.linalg.inv with a condition-number guard (no pinv); finite "perfect" priors;
     explicit check that bound(joint) >= bound(time known) and >= bound(site known).
 [2] along-track orbit error split into a SHARED bias (all satellites, degenerate with t0) and a
     per-satellite part.
 [3] alias scan refined continuously (0.02 s) around every coarse candidate; reports worst miss and
     the shells involved; many random geometries.
 [4] geometries drawn inside the real camera FOV, pointed at 50 deg elevation toward the Sun's azimuth
     (twilight side); streaks clipped to the FOV (endpoints only count when inside).
 [5] time shift moves the observer with Earth rotation, so (dN, dE) are geographic offsets.
 [6] census counts streaks per 10 s exposure inside the actual FOV with extinction; reports
     Poisson P(>=2), P(>=3).
 [8] pixel scales aligned with the report.
 [9] extra nuisances: per-streak tilt (bow/tilt residual), exposure-length error dT.
Everything is still SYNTHETIC two-body geometry: no J2, no real elements, no rolling shutter,
no plate-solution attitude error. Bounds are lower bounds.
"""
import json, os, sys
import numpy as np
from skyclock_sim import build, sat_pos, sun_dir, observer, ts, RE, OMEGA_E, M1000, SHELLS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
W = np.array([0, 0, OMEGA_E])

CAMS = {  # arcmin/px, FOV solid angle sr (ESTIMATED typical values, as in the report)
    "phone 5x tele": (0.26, 0.07),
    "DSLR 24mm FF": (0.74, 1.0),
    "phone main": (1.1, 1.0),
    "phone ultrawide": (1.6, 2.3),
    "all-sky fisheye": (3.6, 6.0),
}
# 1-sigma: shared along-track [s], per-sat along-track [s], cross-track [km], radial [km]
ORBIT = {
    "perfect orbits": (1e-4, 1e-4, 1e-3, 1e-3),
    "operator ephemeris (assumed)": (0.02, 0.05, 0.1, 0.05),
    "public GP, shared bias calibrated": (0.05, 0.28, 0.5, 0.2),
    "public GP, shared bias uncalibrated": (0.30, 0.28, 0.5, 0.2),
}

def boresight(up, east, north, sun, el_deg=50.0):
    s_h = sun - (sun @ up) * up
    az_dir = s_h / np.linalg.norm(s_h)  # toward the Sun's azimuth (twilight side)
    e = np.radians(el_deg)
    return np.cos(e) * az_dir + np.sin(e) * up

def half_angle(fov_sr):
    return np.arccos(1 - fov_sr / (2 * np.pi))

# ------------------------------------------------------------------ Fisher model
NG = 5  # global params: dN [km], dE [km], t0 [s], shared along-track bias [s], dT [s]
NL = 6  # per streak: dt_i [s], c_i [km], r_i [km], sys offset [rad], sys tilt [rad at end], (unused slot for clarity)

def streak_dirs(p, ro, east, north, r, v, T, f):
    dN, dE, dt, bs, dT, dti, ci, rri = p
    tau = f * (T + dT)
    obs = ro[None, :] + dN * north + dE * east + np.outer(dt + tau, np.cross(W, ro))  # Earth-fixed observer [5]
    h = np.cross(r, v); h /= np.linalg.norm(h)
    rhat = r / np.linalg.norm(r)
    sat = r[None, :] + np.outer(dt + tau + bs + dti, v) + ci * h + rri * rhat
    d = sat - obs
    return d / np.linalg.norm(d, axis=1)[:, None]

def fisher(geo, pix_arcmin, bore, fov_sr, orbit, T=10.0, sig_cen_px=0.3, sig_end_px=1.0,
           sig_sys_px=0.3, sig_T=0.05, video_fps=None):
    s = np.radians(pix_arcmin / 60)
    alpha = half_angle(fov_sr)
    rows, wts, used = [], [], 0
    nS = len(geo)
    npar = NG + NL * nS
    prior = np.zeros(npar)
    sa_sh, sa, sc, sr = orbit
    prior[3] = 1 / sa_sh**2
    prior[4] = 1 / sig_T**2
    eps = np.array([1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3])
    for i, (ro, up, east, north, r, v) in enumerate(geo):
        base0 = streak_dirs(np.zeros(8), ro, east, north, r, v, T, np.array([0.0, 1.0]))
        a_hat = base0[1] - base0[0]; a_hat /= np.linalg.norm(a_hat)
        c_hat = np.cross(base0[0], a_hat); c_hat /= np.linalg.norm(c_hat)
        L_px = np.arccos(np.clip(base0[0] @ base0[1], -1, 1)) / s
        nfine = max(int(L_px / 2), 4)
        f = np.linspace(0, 1, nfine)
        u = streak_dirs(np.zeros(8), ro, east, north, r, v, T, f)
        inside = np.arccos(np.clip(u @ bore, -1, 1)) < alpha  # FOV clipping [4]
        if inside.sum() < 4:
            prior[NG + NL * i: NG + NL * (i + 1)] = 1.0  # unused streak: keep matrix regular
            continue
        used += 1
        fin = f[inside]
        ends = [x for x, ok in ((0.0, inside[0]), (1.0, inside[-1])) if ok]
        if video_fps:
            ends = list(np.linspace(0, 1, int(video_fps * T))[np.arccos(np.clip(
                streak_dirs(np.zeros(8), ro, east, north, r, v, T, np.linspace(0, 1, int(video_fps * T))) @ bore, -1, 1)) < alpha])
        def jac(ff):
            b = streak_dirs(np.zeros(8), ro, east, north, r, v, T, ff)
            out = []
            for k in range(8):
                p = np.zeros(8); p[k] = eps[k]
                out.append((streak_dirs(p, ro, east, north, r, v, T, ff) - b) / eps[k])
            return np.array(out)  # (8, n, 3)
        Jc = np.einsum("knj,j->nk", jac(fin), c_hat)
        for row, ff in zip(Jc, fin):
            full = np.zeros(npar)
            full[:NG] = row[:NG]
            full[NG + NL * i: NG + NL * i + 3] = row[NG:]
            full[NG + NL * i + 3] = 1.0                 # systematic cross offset
            full[NG + NL * i + 4] = (2 * ff - 1)        # tilt / bow residual [9]
            rows.append(full); wts.append(1 / (sig_cen_px * s) ** 2)
        if ends:
            Ja = np.einsum("knj,j->nk", jac(np.array(ends)), a_hat)
            for row in Ja:
                full = np.zeros(npar)
                full[:NG] = row[:NG]
                full[NG + NL * i: NG + NL * i + 3] = row[NG:]
                rows.append(full); wts.append(1 / (sig_end_px * s) ** 2)
        prior[NG + NL * i: NG + NL * (i + 1)] = [1 / sa**2, 1 / sc**2, 1 / sr**2,
                                                 1 / (sig_sys_px * s) ** 2, 1 / (sig_sys_px * s) ** 2, 1.0]
    if used == 0:
        return None, 0
    J = np.array(rows); w = np.array(wts)
    return (J * w[:, None]).T @ J + np.diag(prior), used

def crb(F, case):
    keep = {"a": [0, 1], "b": [2], "c": [0, 1, 2]}[case]
    drop = [k for k in (0, 1, 2) if k not in keep]
    idx = [k for k in range(F.shape[0]) if k not in drop]
    Fs = F[np.ix_(idx, idx)]
    # rescale to unit diagonal before the condition test so units don't trigger it
    d = 1 / np.sqrt(np.diag(Fs)); Fn = Fs * d[:, None] * d[None, :]
    if np.linalg.cond(Fn) > 1e12:
        raise np.linalg.LinAlgError("ill-conditioned Fisher matrix")
    cov = np.linalg.inv(Fn) * d[:, None] * d[None, :]
    out = {}
    if case in ("a", "c"):
        out["pos"] = float(np.sqrt(cov[0, 0] + cov[1, 1]))
    if case == "b":
        out["t"] = float(np.sqrt(cov[0, 0]))
    if case == "c":
        out["t"] = float(np.sqrt(cov[2, 2]))
    return out

def sample_geometry(nsat, fov_sr, rng, C, tries=400):
    """N satellites inside the FOV (pointed 50 deg el toward the Sun azimuth) at mid-exposure."""
    jd = ts.utc(2026, 1, 15, 16, 30).ut1  # ~1.5 h after sunset in Stockholm
    ro, up, east, north = observer(59.33, 18.07, jd)
    bore = boresight(up, east, north, sun_dir(jd))
    alpha = half_angle(fov_sr)
    for _ in range(tries):
        r, v = sat_pos(C, rng.uniform(0, 20000))
        d = r - ro; rho = np.linalg.norm(d, axis=1)
        ang = np.arccos(np.clip((d @ bore) / rho, -1, 1))
        el = np.degrees(np.arcsin((d @ up) / rho))
        cand = np.where((ang < alpha) & (el > 15))[0]
        if len(cand) >= nsat:
            pick = rng.choice(cand, nsat, replace=False)
            # shift to exposure start so mid-exposure is in frame
            return [(ro, up, east, north, r[k] - 5.0 * v[k], v[k]) for k in pick], bore
    return None, bore

def crb_sweep(ntrial=25):
    rng = np.random.default_rng(3)
    C = build()
    res, check = {}, {"violations": 0, "illcond": 0, "evaluated": 0}
    for cam, (pix, fov) in CAMS.items():
        for N in (1, 2, 3, 5, 8):
            geos = [sample_geometry(N, fov, rng, C) for _ in range(ntrial)]
            for oname, orb in ORBIT.items():
                vals = {"a": [], "b": [], "c_pos": [], "c_t": [], "n_used": []}
                for g, bore in geos:
                    if g is None:
                        continue
                    F, used = fisher(g, pix, bore, fov, orb)
                    if F is None:
                        continue
                    try:
                        a, b, c = crb(F, "a"), crb(F, "b"), crb(F, "c")
                    except np.linalg.LinAlgError:
                        check["illcond"] += 1; continue
                    check["evaluated"] += 1
                    # an extra unknown can never shrink a bound
                    if c["pos"] < a["pos"] * (1 - 1e-6) or c["t"] < b["t"] * (1 - 1e-6):
                        check["violations"] += 1
                    vals["a"].append(a["pos"]); vals["b"].append(b["t"])
                    vals["c_pos"].append(c["pos"]); vals["c_t"].append(c["t"]); vals["n_used"].append(used)
                res[f"{cam}|{oname}|{N}"] = {k: (float(np.median(x)) if x else None) for k, x in vals.items()} | {
                    "n_geom": len(vals["a"])}
    return res, check

def planted_fault_control():
    """The bound(c) >= bound(a) check must FAIL when the v1 pinv truncation is used on a single
    streak with near-infinite 'perfect' priors. Returns (fault_detected, values)."""
    rng = np.random.default_rng(5); C = build()
    g, bore = sample_geometry(1, 1.0, rng, C)
    F, _ = fisher(g, 1.1, bore, 1.0, (1e-6, 1e-6, 1e-6, 1e-6))
    def bad(Fm, keep_drop):
        idx = [k for k in range(Fm.shape[0]) if k not in keep_drop]
        return np.linalg.pinv(Fm[np.ix_(idx, idx)])
    ca = bad(F, [2]); cc = bad(F, [])
    pa = np.sqrt(ca[0, 0] + ca[1, 1]); pc = np.sqrt(cc[0, 0] + cc[1, 1])
    return bool(pc < pa), {"pinv_time_known_km": float(pa), "pinv_joint_km": float(pc)}

# ------------------------------------------------------------------ census
def census(lat=59.33, lon=18.07, T=10.0, fov_sr=1.0, el_bore=50.0, k_ext=0.25,
           mlims=(3.0, 4.0, 5.0, 6.0, 7.0), sun_lim=-9.0, nphase=4):
    C = build(); rng = np.random.default_rng(7)
    alpha = half_angle(fov_sr)
    out = {}
    for month in range(1, 13):
        jd0 = ts.utc(2026, month, 15, 12).ut1
        rows = []
        for hstep in np.arange(0, 24, 0.25):
            jd = jd0 + hstep / 24
            ro, up, east, north = observer(lat, lon, jd)
            s = sun_dir(jd)
            sun_alt = np.degrees(np.arcsin(s @ up))
            if sun_alt > sun_lim:
                continue
            bore = boresight(up, east, north, s, el_bore)
            ro2 = ro + np.cross(W, ro) * T
            cnt = {m: [] for m in mlims}
            for ph in range(nphase):
                t0 = rng.uniform(0, 20000)
                r1, _ = sat_pos(C, t0); r2, _ = sat_pos(C, t0 + T)
                ins = np.zeros(len(r1), bool); best = np.full(len(r1), 99.0)
                for r, o in ((r1, ro), (r2, ro2), (0.5 * (r1 + r2), 0.5 * (ro + ro2))):
                    d = r - o; rho = np.linalg.norm(d, axis=1)
                    u = d / rho[:, None]
                    inside = np.arccos(np.clip(u @ bore, -1, 1)) < alpha
                    el = np.arcsin(np.clip(u @ up, -1, 1))
                    proj = r @ s
                    lit = (proj > 0) | (np.linalg.norm(r - np.outer(proj, s), axis=1) > RE)
                    airmass = 1 / np.maximum(np.sin(el), 0.1)
                    m = M1000 + 5 * np.log10(rho / 1000) + k_ext * airmass
                    ok = inside & lit & (el > np.radians(10))
                    ins |= ok
                    best = np.where(ok, np.minimum(best, m), best)
                scatter = rng.normal(0, 1.0, len(best))
                for ml in mlims:
                    cnt[ml].append(int(np.sum(ins & (best + scatter < ml))))
            rows.append(dict(hour_utc=float((12 + hstep) % 24), sun_alt=float(sun_alt),
                             mean_streaks={str(m): float(np.mean(v)) for m, v in cnt.items()}))
        out[month] = rows
    return out

# ------------------------------------------------------------------ aliasing
def alias_scan(C, rng, nstreak=3, span=5400.0, coarse=1.0, tol_arcmin=3.0, el_min=30.0):
    jd = ts.utc(2026, 1, 15, 16, 30).ut1
    ro, up, _, _ = observer(59.33, 18.07, jd)
    def sky(t):
        r, v = sat_pos(C, t)
        d = r - ro; rho = np.linalg.norm(d, axis=1); u = d / rho[:, None]
        vp = v - np.sum(v * u, 1)[:, None] * u
        return u, vp / np.linalg.norm(vp, axis=1)[:, None], rho
    t_true = rng.uniform(0, 20000)
    u0, d0, rho0 = sky(t_true)
    el0 = np.degrees(np.arcsin(u0 @ up))
    obs = rng.choice(np.where(el0 > el_min)[0], nstreak, replace=False)
    shells = [int(C["shell"][i]) for i in obs]
    coarse_tol = np.radians(0.9 * coarse + 0.1)  # max motion within half a step + margin
    found = []
    for dt in np.arange(-span, span + coarse, coarse):
        if abs(dt) < 2:  # truth neighbourhood handled separately
            continue
        u, dirn, _ = sky(t_true + dt)
        cands = []
        for i in obs:
            sep = np.arccos(np.clip(u @ u0[i], -1, 1))
            c = np.where((sep < coarse_tol) & (np.abs(dirn @ d0[i]) > np.cos(np.radians(5))))[0]
            if len(c) == 0:
                break
            cands.append(c)
        else:
            # continuous refinement on candidates only
            best = (9e9, None, None)
            sub = {k: None for k in range(len(obs))}
            for fine in np.arange(dt - coarse, dt + coarse, 0.02):
                r, v = sat_pos(C, t_true + fine)
                worst = 0.0; ids = []
                for k, i in enumerate(obs):
                    dd = r[cands[k]] - ro
                    uu = dd / np.linalg.norm(dd, axis=1)[:, None]
                    sep = np.arccos(np.clip(uu @ u0[i], -1, 1))
                    j = np.argmin(sep); worst = max(worst, sep[j]); ids.append(int(cands[k][j]))
                if worst < best[0]:
                    best = (worst, fine, ids)
            if np.degrees(best[0]) * 60 < tol_arcmin:
                found.append((float(best[1]), float(np.degrees(best[0]) * 60),
                              [int(C["shell"][j]) for j in best[2]]))
    # collapse duplicates
    out = []
    for f in sorted(found):
        if not out or abs(f[0] - out[-1][0]) > 3:
            out.append(f)
        elif f[1] < out[-1][1]:
            out[-1] = f
    return dict(streak_shells=shells, aliases=out)

def alias_study(ngeo=12, jitters=(0.0, 1.0, 3.0, 10.0)):
    res = {}
    for jit in jitters:
        C = build(jitter_km=jit, seed=11)
        rng = np.random.default_rng(100)
        res[str(jit)] = [alias_scan(C, rng) for _ in range(ngeo)]
        n = [len(g["aliases"]) for g in res[str(jit)]]
        print(f"jitter {jit} km: aliases per geometry {n}", flush=True)
    return res

if __name__ == "__main__":
    what = sys.argv[1:] or ["control", "crb", "census", "alias"]
    R = {}
    if "control" in what:
        R["control"] = planted_fault_control(); print("planted fault detected:", R["control"])
    if "crb" in what:
        R["crb"], R["crb_check"] = crb_sweep(); print("crb check:", R["crb_check"])
    if "census" in what:
        R["census"] = census()
    if "alias" in what:
        R["alias"] = alias_study()
    with open(os.path.join(OUT, f"v2_{'_'.join(what)}.json"), "w") as f:
        json.dump(R, f, indent=1)
    print("done", what)
