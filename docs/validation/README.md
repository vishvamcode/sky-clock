# Field night: photos with a known time and place

This produces the benchmark set that tells us how accurate sky-clock really is. One clear evening
gives about 20 photos, enough for a first calibration.

## Before you go (10 minutes)

- [ ] Clear sky forecast, ideally no Moon. The best window is **from about 1 h to 2.5 h after sunset**, when
      satellites are still sunlit but the sky is dark. In Stockholm in November that is about 16:30 to 18:00.
- [ ] Camera on a tripod, the widest lens you have (14 to 24 mm on full frame is ideal), manual focus at
      infinity, RAW + JPEG, noise reduction off, no in-camera crop.
- [ ] Camera clock set roughly right. It does not need to be exact, because the reference shots below
      measure its offset.
- [ ] Phone with <https://time.is> open, which shows the time to about 0.1 s. Check that it says the clock
      is exact.
- [ ] Phone GPS on. Note your position from a maps app, to about 10 m.

## At the site

1. **Clock reference, start.** Photograph the phone screen showing time.is with the camera, at a fast
   shutter speed (1/100 s or faster). The difference between the shown time and the camera's EXIF time is
   the camera clock offset. Write down the shown time as well, in case the screen is blurred.
2. **Shoot 15 to 25 frames** over about an hour:
   - exposure 10 to 25 s, ISO 1600 to 6400, aperture wide open;
   - point at different parts of the sky (east, south, west, overhead), at least 30° above the horizon;
   - keep a few frames with trees or roofs at the edge, since real photos have them;
   - optional: 3 to 5 frames in phone night mode, held on the tripod. They test dashed streaks.
3. **Clock reference, end.** Repeat step 1. If the two offsets differ by more than 0.5 s, note it.
4. Note the place: latitude, longitude (decimal degrees) and height above sea level.

## Afterwards

1. Copy the photos (original files, not exported copies) into one folder.
2. For each photo, create a truth file next to it, named like the photo with `.truth.json` instead of the
   extension. `IMG_0412.CR3` needs `IMG_0412.truth.json`. Use `truth.template.json` here as a starting
   point:
   - `time_utc` = EXIF time + clock offset, converted to **UTC**. Stockholm in winter is UTC+1, so
     17:05:12 local is 16:05:12Z. This is the start of the exposure.
   - `exposure_s` = the shutter time.
   - `catalog`: use `celestrak` if you run the benchmark within 2 days. For older photos use
     `spacetrack` (free account) or, if the date is covered, `github-archive`.
3. Run:

   ```bash
   skyclock benchmark path/to/folder                 # check, time and place modes
   skyclock benchmark path/to/folder --modes blind   # slower: no time or place given
   ```

   This writes `results.csv`, `report.md` and `summary.json` into the folder.
4. If at least 8 photos per mode solve, `skyclock benchmark path/to/folder --apply` writes the measured
   error model to `skyclock/calibration.json`, which the solver uses from then on.

## What to share

The benchmark results contain only numbers. Share the photos and truth files only if you are happy for
your observing site to be public. Keep a private site's files out of the repository.
