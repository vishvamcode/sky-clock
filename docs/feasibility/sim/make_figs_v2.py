"""Figures for the corrected (v2) simulation. Reads ../figs/v2_*.json."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4 (adjacent-pair validated)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6})
R = {}
for f in glob.glob(os.path.join(D, "v2_*.json")):
    R.update(json.load(open(f)))

if "census" in R:
    cen = R["census"]; hours = np.arange(0, 24, 0.25)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (ml, title) in zip(axes, [("3.0", "Streaks brighter than mag 3\n(≈ phone, Stockholm; ESTIMATED limit)"),
                                       ("4.0", "Brighter than mag 4\n(≈ phone, dark site; ESTIMATED)"),
                                       ("6.0", "Brighter than mag 6\n(≈ DSLR fast wide lens, dark site; ESTIMATED)")]):
        Z = np.full((12, len(hours)), np.nan)
        for m, rows in cen.items():
            for r in rows:
                Z[int(m) - 1, int(round(r["hour_utc"] / 0.25)) % len(hours)] = r["mean_streaks"][ml]
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1.0,
                       extent=[0, 24, 0.5, 12.5], interpolation="nearest")
        ax.set_title(title, fontsize=10, color=INK); ax.set_xlabel("UTC hour (local = UTC+1 in winter)")
        ax.set_xticks(range(0, 25, 4)); ax.grid(False)
    axes[0].set_yticks(range(1, 13)); axes[0].set_yticklabels("JFMAMJJASOND"); axes[0].set_ylabel("Month (15th)")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01); cb.set_label("mean Starlink-like streaks per 10 s frame")
    fig.text(0.42, -0.06, "Synthetic constellation; 59.3°N; 1-sr field at 50° elevation toward the Sun's azimuth; Sun < −9°; "
             "extinction 0.25 mag/airmass. Blank = no usable darkness.", ha="center", fontsize=9, color=INK2)
    fig.savefig(os.path.join(D, "fig1_streaks_per_photo.png"), dpi=150, bbox_inches="tight")

if "crb" in R:
    crb = R["crb"]; Ns = [2, 3, 5, 8]
    names = ["public GP, shared bias uncalibrated", "public GP, shared bias calibrated",
             "operator ephemeris (assumed)", "perfect orbits"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, col in zip(names, S):
        pos = [crb[f"phone main|{name}|{n}"]["c_pos"] for n in Ns]
        tt = [crb[f"phone main|{name}|{n}"]["c_t"] for n in Ns]
        a1.plot(Ns, pos, "-o", color=col, lw=2, ms=6, label=name)
        a2.plot(Ns, tt, "-o", color=col, lw=2, ms=6, label=name)
    for ax, yl, goal in [(a1, "1σ horizontal position error [km]", 2.0), (a2, "1σ time error [s]", 1.0)]:
        ax.set_yscale("log"); ax.set_xlabel("streaks in the photo (joint solve; 1 streak is degenerate)")
        ax.set_ylabel(yl); ax.set_xticks(Ns)
        ax.axhline(goal, color=INK2, lw=1, ls="--"); ax.text(2, goal * 1.12, f"v4 target ({goal:g})", fontsize=8, color=INK2)
    a1.set_title("Location (time unknown)", fontsize=10); a2.set_title("Time (location unknown)", fontsize=10)
    a2.legend(frameon=False, fontsize=8, loc="center right")
    fig.text(0.5, -0.06, "Cramér–Rao lower bound; phone main 1.1′/px; 10 s; streaks inside a 1-sr field at 50° elevation, clipped "
             "to the frame; median over sampled geometries. NUMERICAL, synthetic.", ha="center", fontsize=9, color=INK2)
    fig.savefig(os.path.join(D, "fig2_crb_vs_streaks.png"), dpi=150, bbox_inches="tight")

    cams = ["phone 5x tele", "DSLR 24mm FF", "phone main", "phone ultrawide", "all-sky fisheye"]
    pix = [0.26, 0.74, 1.1, 1.6, 3.6]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, col in zip(names, S):
        xs, ys = [], []
        for c, p in zip(cams, pix):
            v = crb.get(f"{c}|{name}|3", {}).get("c_pos")
            if v is not None:
                xs.append(p); ys.append(v)
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=6, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(pix); ax.set_xticklabels([f"{p}\n{c.split()[-1]}" for p, c in zip(pix, cams)], fontsize=8)
    ax.set_xlabel("pixel scale [arcmin/px]  (tele: too few geometries with 3 streaks in field)")
    ax.set_ylabel("1σ position error, 3 streaks, joint [km]"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("With real orbit errors, pixel scale matters little", fontsize=10)
    fig.savefig(os.path.join(D, "fig3_pixel_scale.png"), dpi=150, bbox_inches="tight")

if False and "alias" in R:  # all-zero result; reported as a table instead
    al = R["alias"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for k, (jit, col) in enumerate(zip(["0.0", "1.0", "3.0", "10.0"], S)):
        if jit not in al: continue
        n = [len(g["aliases"]) for g in al[jit]]
        x = np.array(n) + np.random.default_rng(k).uniform(-0.15, 0.15, len(n))
        ax.scatter(x, [k] * len(n), s=60, color=col, edgecolor=SURF, linewidth=2, zorder=3)
        ax.text(max(n) + 0.6 if n else 0.5, k, f"median {np.median(n):.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(4)); ax.set_yticklabels(["perfect slots", "1 km scatter", "3 km scatter", "10 km scatter"])
    ax.set_xlabel("false (time, longitude) solutions within ±90 min matching all 3 streaks to < 3′")
    ax.set_title("Aliases per 3-streak geometry (each dot = one random geometry)", fontsize=10)
    fig.savefig(os.path.join(D, "fig4_aliases.png"), dpi=150, bbox_inches="tight")
print("figs written:", sorted(k for k in R))
