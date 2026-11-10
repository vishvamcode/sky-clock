"""Sky-clock feasibility simulator (v1). NOTE: census, fisher_case, crb and alias_scan here are SUPERSEDED by skyclock_v2.py (see review log in report.md); only the constellation helpers are still used.

Everything here is SYNTHETIC: a Walker-style stand-in for the Starlink constellation
(shell parameters are approximate, not fitted to real element sets, because CelesTrak
was unreachable from the sandbox). Results are ESTIMATES of scaling, not predictions
for any particular night.

Run:  python3 skyclock_sim.py   (writes figures + results.json next to ../figs)
Requires numpy, scipy, matplotlib, skyfield (only for timescale / WGS84 / GMST).
"""
import json, os
import numpy as np
from skyfield.api import load, wgs84

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
RES = {}
ts = load.timescale()
RE = 6378.137
MU = 398600.4418
OMEGA_E = 7.2921159e-5  # rad/s

# ---------------------------------------------------------------- constellation
# (inclination deg, altitude km, planes, sats/plane, Walker F) -- ESTIMATED stand-in
SHELLS = [
    (53.0, 480, 72, 22, 17),   # Gen1 shell-1 after the 2026 lowering 550 -> ~480 km
    (53.2, 540, 72, 22, 11),
    (70.0, 570, 36, 20, 5),
    (97.6, 560, 6, 58, 1),
    (43.0, 530, 48, 50, 7),    # Gen2 "V2 Mini" style shells
    (53.0, 525, 48, 50, 9),
    (33.0, 535, 28, 60, 3),
]

def build(jitter_km=0.0, seed=1):
    rng = np.random.default_rng(seed)
    el = []
    for si, (inc, alt, P, S, F) in enumerate(SHELLS):
        a = RE + alt
        n = np.sqrt(MU / a**3)
        for j in range(P):
            for k in range(S):
                u0 = 2*np.pi*k/S + 2*np.pi*F*j/(S*P)
                u0 += rng.normal(0, jitter_km / a) if jitter_km > 0 else 0.0
                el.append((si, np.radians(inc), 2*np.pi*j/P, u0, a, n))
    el = np.array(el)
    return dict(shell=el[:, 0].astype(int), inc=el[:, 1], raan=el[:, 2], u0=el[:, 3], a=el[:, 4], n=el[:, 5])

def sat_pos(C, t_s, du=None):
    """Inertial positions (km) and velocities (km/s) at t_s seconds after epoch (circular two-body)."""
    u = C["u0"] + C["n"] * t_s + (0 if du is None else du)
    ci, si_ = np.cos(C["inc"]), np.sin(C["inc"])
    cO, sO = np.cos(C["raan"]), np.sin(C["raan"])
    cu, su = np.cos(u), np.sin(u)
    x = cO*cu - sO*su*ci; y = sO*cu + cO*su*ci; z = su*si_
    vx = -cO*su - sO*cu*ci; vy = -sO*su + cO*cu*ci; vz = cu*si_
    r = np.stack([x, y, z], -1) * C["a"][:, None]
    v = np.stack([vx, vy, vz], -1) * (C["a"]*C["n"])[:, None]
    return r, v

def sun_dir(jd):
    """Low-precision Sun unit vector (equatorial, ~0.01 deg; Astronomical Almanac)."""
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474*n) % 360); g = np.radians((357.528 + 0.9856003*n) % 360)
    lam = L + np.radians(1.915)*np.sin(g) + np.radians(0.020)*np.sin(2*g)
    eps = np.radians(23.439 - 4e-7*n)
    return np.array([np.cos(lam), np.cos(eps)*np.sin(lam), np.sin(eps)*np.sin(lam)])

def observer(lat, lon, jd_ut1, h_km=0.0):
    """Observer position (km) in an Earth-centred frame rotated by GMST (simple, no precession)."""
    t = ts.ut1_jd(jd_ut1)
    gmst = np.radians(t.gmst * 15.0)
    p = wgs84.latlon(lat, lon, h_km*1000).itrs_xyz.km
    c, s = np.cos(gmst), np.sin(gmst)
    r = np.array([c*p[0]-s*p[1], s*p[0]+c*p[1], p[2]])
    up = r / np.linalg.norm(r)
    east = np.cross([0, 0, 1], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return r, up, east, north

# ------------------------------------------------------------ brightness model
M1000 = 7.87      # VERIFIED (abstract-only): Mallama et al. 2023, arXiv:2306.06657, V2 Mini mean at 1000 km
M_SCATTER = 1.0   # ESTIMATED spread of individual magnitudes (phase angle, attitude)

def mag(rho_km, rng):
    return M1000 + 5*np.log10(rho_km/1000.0) + rng.normal(0, M_SCATTER, rho_km.shape)

# -------------------------------------------------------- visibility census
def census(lat=59.33, lon=18.07, year=2026, sun_lim=-9.0, el_min=25.0, mlims=(4.0, 5.0, 6.0, 7.0)):
    C = build()
    rng = np.random.default_rng(7)
    out = {}
    for month in range(1, 13):
        jd0 = ts.utc(year, month, 15, 12).ut1  # local noon-ish UTC (Stockholm UTC+1)
        rows = []
        for hstep in np.arange(0, 24, 0.25):
            jd = jd0 + hstep/24
            t_s = (jd - jd0) * 86400 + (month-1)*0  # constellation phase irrelevant for stats
            ro, up, _, _ = observer(lat, lon, jd)
            s = sun_dir(jd)
            sun_alt = np.degrees(np.arcsin(np.dot(s, up)))
            if sun_alt > sun_lim:
                continue
            r, _ = sat_pos(C, t_s + month*86400*3.3)
            d = r - ro; rho = np.linalg.norm(d, axis=1)
            el = np.degrees(np.arcsin((d @ up) / rho))
            proj = r @ s
            lit = (proj > 0) | (np.linalg.norm(r - np.outer(proj, s), axis=1) > RE)
            vis = (el > el_min) & lit
            m = mag(rho, rng)
            # area above el_min in sr
            omega = 2*np.pi*(1 - np.sin(np.radians(el_min)))
            dens = {ml: float(np.sum(vis & (m < ml))) / omega for ml in mlims}
            rows.append(dict(hour_utc=float((12 + hstep) % 24), sun_alt=float(sun_alt),
                             lit_above=int(vis.sum()), dens_per_sr=dens))
        out[month] = rows
    return out

# ------------------------------------------------------------ Fisher / CRB
CAMS = {  # pixel scale arcmin/px (ESTIMATED from typical focal lengths/sensors), FOV sr
    "phone ultrawide": (1.8, 2.3),
    "phone main": (1.1, 1.0),
    "phone 5x tele": (0.22, 0.07),
    "DSLR 24mm FF": (0.74, 1.0),
    "all-sky fisheye": (3.6, 6.28),
}
ORBIT = {  # 1-sigma per-satellite biases: along-track [s], cross-track [km], radial [km]
    "perfect orbits": (0.0, 0.0, 0.0),
    "operator ephemeris (assumed)": (0.05, 0.1, 0.05),   # ESTIMATED
    "public GP/TLE, fresh": (0.4, 0.5, 0.2),             # along: Halferty 2022 (0.3 s mean, 0.28 sd); cross: ESTIMATED
}

def streak_meas(ro, r_sat, v_sat, T, n_cross, t_off=0.0):
    """Return unit vectors at n_cross sample times plus the two endpoints."""
    tt = np.linspace(0, T, n_cross) + t_off
    P = r_sat[None, :] + np.outer(tt, v_sat)  # straight-line over a few s is fine (<0.1 px curvature check below)
    d = P - ro[None, :]
    return d / np.linalg.norm(d, axis=1)[:, None]

def fisher_case(geo, pix_arcmin, T=10.0, sig_cen_px=0.3, sig_end_px=1.0, sig_sys_px=0.3,
                orbit=(0.4, 0.5, 0.2), video_fps=None):
    """Fisher matrix for theta=(north km, east km, t0 s) with per-streak nuisances.

    geo: list of (ro, up, east, north, r_sat, v_sat).  Measurements:
      - cross-track offsets of many samples along the streak (sigma = sig_cen per PSF-length sample)
      - along-track position of the two endpoints (sigma = sig_end), or every video frame
      - per-streak systematic cross-track offset (lens distortion residual), sigma_sys
      - per-satellite orbit bias: along-track time dt_i, cross-track c_i, radial r_i
    """
    s = np.radians(pix_arcmin/60.0)  # rad/px
    rows_J, rows_w = [], []
    nS = len(geo)
    npar = 3 + 4*nS  # theta + (dt_i, c_i, rr_i, sys_i)
    prior = np.zeros(npar)
    for i, (ro, up, east, north, r, v) in enumerate(geo):
        rho = np.linalg.norm(r - ro)
        omega = np.linalg.norm(np.cross(r - ro, v - np.cross([0, 0, OMEGA_E], ro))) / rho**2  # rad/s
        L_px = omega*T/s
        n_cross = max(int(L_px/2), 2)
        vrel = v - np.cross([0, 0, OMEGA_E], ro)
        h = np.cross(r, v); h /= np.linalg.norm(h)
        rhat = r/np.linalg.norm(r)
        def model(p, idx_times):
            dN, dE, dt, dti, ci, rri = p
            ro2 = ro + dN*north + dE*east
            r2 = r + v*(dt+dti) + ci*h + rri*rhat
            return streak_meas(ro2, r2, vrel, T, idx_times)
        # local streak frame
        u0 = model(np.zeros(6), 2)
        a_hat = u0[1]-u0[0]; a_hat /= np.linalg.norm(a_hat)
        c_hat = np.cross(u0[0], a_hat); c_hat /= np.linalg.norm(c_hat)
        eps = np.array([1e-3, 1e-3, 1e-4, 1e-4, 1e-3, 1e-3])
        def jac(nt):
            base = model(np.zeros(6), nt)
            J = []
            for k in range(6):
                p = np.zeros(6); p[k] = eps[k]
                J.append((model(p, nt) - base)/eps[k])
            return np.array(J)  # (6, nt, 3)
        Jc = jac(n_cross)                       # cross-track samples
        Jc = np.einsum("knj,j->nk", Jc, c_hat)  # (nt,6)
        w_c = 1/(sig_cen_px*s)**2
        if video_fps:
            nf = max(int(video_fps*T), 2)
            Ja = np.einsum("knj,j->nk", jac(nf), a_hat); w_a = 1/(sig_end_px*s)**2
        else:
            Ja = np.einsum("knj,j->nk", jac(2), a_hat); w_a = 1/(sig_end_px*s)**2
        for J, w in ((Jc, w_c), (Ja, w_a)):
            for row in J:
                full = np.zeros(npar)
                full[0:3] = row[0:3]
                full[3+4*i:3+4*i+3] = row[3:6]
                if J is Jc:
                    full[3+4*i+3] = 1.0   # systematic cross offset in radians
                rows_J.append(full); rows_w.append(w)
        sa, sc, sr = orbit
        prior[3+4*i:3+4*i+4] = [1/max(sa, 1e-6)**2, 1/max(sc, 1e-6)**2, 1/max(sr, 1e-6)**2, 1/(sig_sys_px*s)**2]
    J = np.array(rows_J); w = np.array(rows_w)
    F = (J*w[:, None]).T @ J + np.diag(prior)
    return F

def crb(F, case):
    keep = {"a": [0, 1], "b": [2], "c": [0, 1, 2]}[case]
    drop = [k for k in (0, 1, 2) if k not in keep]
    idx = [k for k in range(F.shape[0]) if k not in drop]
    Fs = F[np.ix_(idx, idx)]
    cov = np.linalg.pinv(Fs)
    sub = cov[:len(keep), :len(keep)]
    if case == "b":
        return dict(sig_t=float(np.sqrt(sub[0, 0])))
    pos = float(np.sqrt(np.trace(sub[:2, :2]))) if case != "b" else None
    d = dict(sig_pos_km=pos)
    if case == "c":
        d["sig_t"] = float(np.sqrt(sub[2, 2]))
    return d

def random_geometries(nsat, fov_sr, rng, lat=59.33, lon=18.07, el_min=30):
    """Pick nsat satellites from the synthetic constellation inside a cone of solid angle fov_sr around zenith-ish."""
    C = build()
    jd = ts.utc(2026, 1, 15, 17).ut1
    for _ in range(200):
        ro, up, east, north = observer(lat, lon, jd)
        r, v = sat_pos(C, rng.uniform(0, 6000))
        d = r - ro; rho = np.linalg.norm(d, axis=1)
        el = np.degrees(np.arcsin((d@up)/rho))
        half = np.arccos(1 - fov_sr/(2*np.pi))  # cone half-angle with that solid angle
        bore = np.cos(np.radians(25))*north*0 + up
        ang = np.arccos(np.clip((d@bore)/rho, -1, 1))
        cand = np.where((el > el_min) & (ang < max(half, np.radians(45))))[0]
        if len(cand) < nsat:  # wide-field fallback: anywhere above 25 deg elevation
            cand = np.where(el > 25)[0]
        if len(cand) >= nsat:
            pick = rng.choice(cand, nsat, replace=False)
            return [(ro, up, east, north, r[i], v[i]) for i in pick]
    return None

def crb_sweep():
    rng = np.random.default_rng(3)
    res = {}
    for cam, (pix, fov) in CAMS.items():
        for oname, orb in ORBIT.items():
            for N in (1, 2, 3, 5, 8):
                vals = {"a": [], "b": [], "c_pos": [], "c_t": []}
                for trial in range(25):
                    g = random_geometries(N, max(fov, 1.0), rng)
                    if g is None: continue
                    F = fisher_case(g, pix, orbit=orb)
                    vals["a"].append(crb(F, "a")["sig_pos_km"])
                    vals["b"].append(crb(F, "b")["sig_t"])
                    c = crb(F, "c"); vals["c_pos"].append(c["sig_pos_km"]); vals["c_t"].append(c["sig_t"])
                res[f"{cam}|{oname}|{N}"] = {k: float(np.median(v)) for k, v in vals.items()}
    return res

# ---------------------------------------------------------- Walker aliasing
def alias_scan(jitter_km, nstreak=3, seed=11, span_s=3*3600, step=1.0, tol_arcmin=3.0):
    """Inertially-fixed observer: scan time shifts and count shifts at which EVERY observed streak
    (position + direction) is matched by SOME satellite within tol. Shifts != 0 are aliases.

    Observer longitude co-moves (lambda' = lambda - omega_E*dt) so that the observer's inertial position
    is unchanged; stars are therefore identical, and only the constellation pattern can disambiguate."""
    rng = np.random.default_rng(seed)
    C = build(jitter_km=jitter_km, seed=seed)
    jd = ts.utc(2026, 1, 15, 17).ut1
    ro, up, _, _ = observer(59.33, 18.07, jd)
    def sky(t):
        r, v = sat_pos(C, t)
        d = r - ro; rho = np.linalg.norm(d, axis=1)
        u = d/rho[:, None]
        el = np.degrees(np.arcsin(u@up))
        vperp = v - (np.sum(v*u, 1))[:, None]*u
        dirn = vperp/np.linalg.norm(vperp, axis=1)[:, None]
        return u, dirn, el
    u0, d0, el0 = sky(0.0)
    obs = rng.choice(np.where(el0 > 35)[0], nstreak, replace=False)
    tol = np.radians(tol_arcmin/60)
    hits = []
    for dt in np.arange(-span_s, span_s+step, step):
        u, dirn, el = sky(dt)
        ok = True; worst = 0.0
        for i in obs:
            sep = np.arccos(np.clip(u@u0[i], -1, 1))
            j = np.argmin(sep)
            if sep[j] > tol or abs(dirn[j]@d0[i]) < np.cos(np.radians(2)):
                ok = False; break
            worst = max(worst, sep[j])
        if ok:
            hits.append((float(dt), float(np.degrees(worst)*60)))
    return hits

if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["census", "crb", "alias"]
    if "census" in what:
        RES["census"] = census()
    if "crb" in what:
        RES["crb"] = crb_sweep()
    if "alias" in what:
        RES["alias"] = {}
        for jit in (0.0, 0.3, 3.0, 30.0):
            hits = alias_scan(jit, step=2.0)
            RES["alias"][str(jit)] = hits
            print("jitter", jit, "km -> matching shifts:", [h for h in hits][:20], "count", len(hits))
    with open(os.path.join(OUT, f"results_{'_'.join(what)}.json"), "w") as f:
        json.dump(RES, f, indent=1)
    print("done", what)
