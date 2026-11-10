# sky-clock

**Find when and where a night photo was taken from the satellite streaks in it.**

Upload a night-sky photo that shows satellite trails (Starlink and friends). sky-clock finds the
streaks and stars, works out exactly where the camera pointed (plate solving), and compares the streaks
with where every catalogued satellite was. From that it can:

| Mode | You give | You get |
|---|---|---|
| `identify` | rough time and place | which satellite made each streak, refined time and place |
| `check` | a claimed time and place (caption, EXIF) | **consistent / tension / inconsistent**, with a p-value |
| `time` | the place | the exposure start time (UTC) |
| `place` | the time | the observer's position |
| `blind` | a search window (e.g. one night) | time and place, plus ranked alternatives |

On simulated photos (see *Accuracy*), time comes out to about 0.3 s and the place to about 0.5 km
(1 sigma) from 5 to 8 streaks. **Real photos have not been tested yet**, and the error model is an
estimate (see *Validation status*).

> **Ethics.** sky-clock is for checking claims about images (journalism, OSINT on public material,
> science, your own photos). Do not use it to locate private people from their personal photos. The app
> keeps photos in memory only, and blind/place/time searches ask you to confirm the use first.

---

## Quick start

### Option A: Docker (everything included, local plate solving)

```bash
git clone https://github.com/vishvamcode/sky-clock.git
cd sky-clock
docker compose up --build        # first build downloads ~400 MB of star index files
# open http://localhost:8501
```

### Option B: Python (macOS / Linux / Windows)

```bash
git clone https://github.com/vishvamcode/sky-clock.git
cd sky-clock
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
streamlit run app/streamlit_app.py
```

Without Docker you need a plate solver. You can:

* install astrometry.net locally (Ubuntu/Debian: `sudo apt install astrometry.net astrometry-data-tycho2-10-19`;
  macOS: `brew install astrometry-net` plus the 4100-series index files), or
* use the free **nova.astrometry.net** web solver with an API key (see *Accounts*), or
* upload a WCS file you already have.

### Try it without a photo

Click **"Use a simulated demo photo"** in the app. It renders a 70-degree view over Stockholm on
2026-11-10 16:10 UTC, with real Tycho-2 stars and a synthetic Starlink-like constellation, so you can
run every mode and see the error against the known truth.

---

## Using the app

1. **Photo.** Upload JPEG, PNG, TIFF or FITS (RAW needs `pip install rawpy`). Prefer the original file:
   no crop, no rotation, no heavy noise reduction. Longer exposures (5 to 30 s), wide lenses and a fixed
   tripod work best.
2. **Streaks.** Check the detected streaks. Untick aircraft (blinking dotted lines), meteors (short,
   tapered) and lens flares. Circles mark real endpoints; squares mark ends cut by the frame edge, which
   are not used for timing. Dashed streaks (phone night modes stack many short frames) are used for
   direction only.
3. **Mode.** Pick what you know (see the table). Uncertainties are 1 sigma: "time +-60 s" means you
   are fairly sure within a minute. If EXIF has a time, place or exposure, the app shows it and fills it
   in, but the EXIF clock is only used if you leave it in.
4. **Catalogue** (sidebar). For photos from the last ~2 days, *CelesTrak, current* is fine. For older
   photos use *Space-Track history* (free account), because orbits drift by kilometres per day.
5. **Analyse.** You get the place, the exposure start (UTC), the exposure length, which satellite made
   each streak, the fit quality, a map with the 2-sigma ellipse, alternatives (aliases), and downloads
   (JSON and annotated PNG).

### Reading the result

* **chi2/dof** close to or below 1 means the streaks match the predicted tracks as well as the error model
  expects. Much above 1 means something is off: a misidentified streak, a stale catalogue, a distorted or
  cropped image.
* **Alternatives.** Starlink shells repeat their pattern every few minutes, so a solution a few minutes
  and a few hundred km away can explain some streaks too. The best solution should explain more streaks
  or have a much smaller chi2. Streaks from different shells (different directions of motion) break these
  aliases.
* **One streak** only gives a set of lines on the map, not a point, unless the place or the time is
  known.
* **Check verdict.** A likelihood-ratio test of your claim with your stated tolerances: *consistent*
  (p >= 0.01), *tension* (1e-4 <= p < 0.01), *inconsistent* (p < 1e-4). *No match* means no catalogued
  satellite fits at the claimed time and place: the claim is wrong, or the streaks are not catalogued
  satellites, or the catalogue is stale.

---

## Command line

```bash
# make a test photo with known truth (real stars if astrometry.net is installed)
skyclock simulate --out demo --real-stars

# check a claim
skyclock analyze demo.png --catalog synthetic --mode check \
    --time 2026-11-10T16:10:00Z --lat 59.33 --lon 18.06 --sigma-time 2 --sigma-loc 1

# find the time for a known place, within one night
skyclock analyze photo.jpg --mode time --lat 59.33 --lon 18.06 \
    --window 2026-11-10T15:00:00Z 2026-11-11T05:00:00Z

# blind search (requires confirming legitimate use)
skyclock analyze photo.jpg --mode blind --window 2026-11-10T15:00Z 2026-11-10T21:00Z --i-understand

# other useful flags: --use-exif, --exposure 10, --tracking, --disable 3,5, --region 59,18,1500,
#   --catalog spacetrack | celestrak-starlink | file --catalog-file my.tle, --plate nova, --wcs my.wcs
```

Outputs go to `skyclock_out/`: `summary.md`, `result.json`, `annotated.png`, `map.html`, `time.png`.

---

## Accounts (all free, all optional)

sky-clock never creates accounts for you. Keys and passwords are read from environment variables or typed
into the app for that session only; they are never written to files or logs.

1. **CelesTrak**: no account. Used by default for current orbits.
2. **Space-Track.org** (for photos older than ~2 days):
   1. Go to <https://www.space-track.org/auth/createAccount> and register (free; you accept their user
      agreement yourself).
   2. Wait for the confirmation e-mail and log in once in the browser.
   3. Give sky-clock the credentials either in the app sidebar or as environment variables
      `SPACETRACK_USER` and `SPACETRACK_PASSWORD` (e.g. in a local `.env` file next to
      `docker-compose.yml`; `.env` is git-ignored).
   4. Space-Track data may not be redistributed: sky-clock caches it locally (`~/.cache/skyclock`) and
      never commits or uploads it.
3. **nova.astrometry.net** (only if you cannot install astrometry.net locally):
   1. Sign in at <https://nova.astrometry.net/signin> (e.g. with a Google/GitHub account).
   2. Open <https://nova.astrometry.net/api_help> and copy your API key.
   3. Paste it in the app sidebar, or set `ASTROMETRY_NET_API_KEY`.
   4. sky-clock uploads an 8-bit grey copy without EXIF, marked private. If that is still too much for a
      sensitive image, use the local solver (Docker).

---

## How it works

1. **Detection** (`skyclock/detect.py`): robust background, significance mask, components split into
   stars (compact) and streak fragments (elongated); collinear fragments are merged across stars and
   gaps; each streak is refined with cross-track profiles every 2 px (sub-pixel centre line, half-maximum
   endpoints, flags for frame-clipped and dashed streaks).
2. **Plate solving** (`skyclock/platesolve.py`): our own star list goes to astrometry.net, so the WCS
   matches our pixel convention exactly; RMS from the matched stars feeds the error model.
3. **Measurements** (`skyclock/observations.py`): streak samples become sky directions; annual
   aberration is removed so they can be compared with satellite positions; each streak gets a great
   circle, an arc and uncertainties.
4. **Model** (`skyclock/fit.py`): SGP4 positions (Earth-fixed frame), the camera's Earth-fixed
   orientation (fixed tripod; tracking mounts supported), and per-streak nuisance parameters for
   catalogue errors (0.28 s along-track, 0.5 km cross-track) plus a shared 0.3 s timing bias.
   Cross-track residuals pin down *where* you are; the streak endpoints (start and end of the exposure)
   pin down *when*.
5. **Search** (`skyclock/search.py`): a streak's great circle and the observer define a plane that must
   contain the satellite's velocity. Roots of that condition over the search window give candidate
   (satellite, time) pairs; each becomes a short curve of possible observer positions on the ground; curves
   from different streaks that cross at the same time are candidate solutions, which are clustered and then
   fully fitted and ranked.
6. **Verdicts** (`skyclock/solve.py`): the check mode compares the best fit with a fit constrained to the
   claim (likelihood ratio, 3 degrees of freedom).

Background, error budget, novelty search and the original feasibility study: `docs/feasibility/report.md`.

---

## Accuracy and validation status

**What has been tested** (`pytest`, 21 tests, all passing at the time of writing):

* Geometry: WGS84 round trip, aberration inverse, Earth rotation rate, and our SGP4 Earth-fixed positions
  against Skyfield's independent implementation (agree to < 10 m).
* Detection: endpoint accuracy on controlled streaks, clipped-edge flagging, detections on simulated scenes.
* Plate solving: astrometry.net on a simulated photo with real Tycho-2 stars, error < 0.5 px.
* Solver, on a simulated Stockholm photo (5 to 8 streaks): identify, time, place and blind modes recover
  the truth within 2 km and 1 s (typically 0.1 km and 0.1 s); the check mode accepts the true claim and
  rejects claims that are 30 s, 55 km, or 3 s (with 0.3 s tolerance) off; one streak gives a set of lines,
  not a point, for place mode, but the correct time for time mode.
* The tests were checked against five planted bugs (wrong aberration sign, tripod modelled as tracking,
  verdict forced to "consistent", Earth rotation rate +1 %, endpoints ignored); each one made at least
  one test fail.

**What has not been tested:** real photos. The simulator and the solver share the propagation code, so
the simulations test internal consistency and statistics, not agreement with the real sky. The
catalogue error model (0.28 s, 0.5 km, shared 0.3 s) is an estimate for public GP data and needs
calibration.

**Validation protocol for real photos** (cheap, and worth doing before trusting the numbers):

1. Take 10 to 20 photos yourself on a tripod (wide lens, 5 to 20 s, ISO 1600 to 6400) in the hour after
   dusk, with a phone showing GPS time (e.g. time.is) photographed before and after, and note the
   place from GPS.
2. Run `check` with the truth and `blind` with a window of a few hours.
3. Record time error, distance error, chi2/dof and the error bars. If errors are often more than 3
   sigma, widen the error model in `skyclock/fit.py` (`SIG_DT`, `SIG_CROSS_KM`, `SIG_SHARED_DT`).
4. Repeat with Space-Track history elements, and with phone night-mode photos (dashed streaks).

---

## Limitations

* Needs at least ~8 detectable stars in the frame for plate solving; heavy light pollution, clouds and
  very long lenses (< 5 degree fields) may fail. Fisheye lenses need a strong SIP fit and may be worse.
* Cropped, rotated, or warped images break the "pixel -> sky" mapping only if the plate solve fails; a
  plain crop is fine.
* Uncatalogued objects (classified satellites, new launches before cataloguing), aircraft and meteors
  will not match; untick them.
* Phone night modes stack frames: endpoints are not used for timing, so time comes only from what the
  dashes give.
* Starlink manoeuvres make old elements wrong by km and seconds: use elements from within a day of the
  photo (Space-Track history).
* The blind search over a whole night with ~10,000 LEO objects takes minutes on a laptop; narrow the window
  or region when you can.
* Observer height is fixed (default sea level); a mountain top at 2 km shifts the place by roughly that
  much along the line of sight.

## Project layout

```
skyclock/      library: catalog, geometry, imageio, detect, platesolve, observations,
               fit, identify, search, solve, report, pipeline, simulate, cli
app/           Streamlit web app
tests/         pytest suite
docs/feasibility/   original feasibility report, figures and simulation scripts
```

## Licence

MIT. Satellite data: CelesTrak (public), Space-Track (redistribution restricted; never committed here).
Star catalogue: Tycho-2 via astrometry.net index files.
