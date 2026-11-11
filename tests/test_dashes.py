"""Stacked night-mode photos: a satellite leaves a dashed streak, one dash per sub-exposure. Regular dashes
are timed (observations.dash_timing) and their ends trusted; irregular ones are not."""
from datetime import timedelta

import numpy as np
import pytest

from skyclock import observations as O, simulate as S, solve as SV
from skyclock.detect import detect

from conftest import LAT, LON, WHEN
from test_detect import _line_image


@pytest.fixture(scope="module")
def dashed_line():
    # 10 frames, each covering 0.8 of a 0.1 cycle along the line: dash k spans [k/10, k/10 + 0.08]
    p1, p2 = (300.0, 200.0), (900.0, 520.0)
    det = detect(_line_image(p1, p2, gap=0.8))
    assert len(det.streaks) == 1
    return det.streaks[0]


def test_dashes_are_measured_and_timed(dashed_line):
    s = dashed_line
    assert s.dashed and len(s.dashes) == 10
    f, xy = O.dash_timing(s.dashes, s.p1, s.p2)
    # exposure [0, 0.98] of the line; dash k centred at k/10 + 0.04, i.e. (k - 4.5) / 9.8 of it from the middle
    truth = (np.arange(10) - 4.5) / 9.8
    if np.hypot(*np.subtract(s.p1, (300.0, 200.0))) > 50:
        truth = truth[::-1] * -1  # endpoints reported the other way round: same fractions, reversed order
    assert np.max(np.abs(f - truth)) < 0.003, f - truth


@pytest.mark.parametrize("damage", ["dropped frame", "blinking", "star in a gap", "too few"])
def test_irregular_dashes_are_not_timed(dashed_line, damage):
    """Failure controls: each break of the stacked-exposure pattern must switch the timing off."""
    d = [list(map(list, x)) for x in dashed_line.dashes]
    if damage == "dropped frame":
        d = d[:4] + d[5:]
    elif damage == "blinking":  # every other dash much shorter, centres unchanged (only the duty check sees it)
        for k in range(1, len(d), 2):
            a, b = np.array(d[k][0]), np.array(d[k][1])
            d[k] = [list(a + 0.25 * (b - a)), list(a + 0.75 * (b - a))]
    elif damage == "star in a gap":  # dashes 3 and 4 merged into one
        d = d[:3] + [[d[3][0], d[4][1]]] + d[5:]
    else:
        d = d[:2]
    assert O.dash_timing(d, dashed_line.p1, dashed_line.p2) is None


@pytest.fixture(scope="module")
def dashed_scene(constellation):
    from skyclock import platesolve as PS
    img, wcs, truth = S.simulate_photo(constellation, WHEN, LAT, LON, exposure_s=10, alt_deg=50, az_deg=200,
                                       cam=S.Camera(fov_x_deg=70), seed=3, dashed_gap=(10 / 6, 10 / 24))
    return dict(det=detect(img), plate=PS.from_wcs_object(wcs, img.shape), truth=truth)


def _settings(**kw):
    return SV.Settings(mode="time", lat=LAT, lon=LON, sigma_loc_km=1, window_start=WHEN - timedelta(minutes=30),
                       window_end=WHEN + timedelta(minutes=30), **kw)


def test_dashed_photo_gives_time_and_exposure(dashed_scene, constellation):
    s = _settings()
    obs = O.build(dashed_scene["det"].streaks, dashed_scene["plate"], s.epoch())
    assert sum(o.regular_dashes for o in obs) >= 2
    sol = SV.analyze(obs, dashed_scene["plate"], constellation, s).solutions[0]
    assert abs((sol.t_start - WHEN).total_seconds()) < 1.0
    assert abs(sol.exposure_s - 10.0) < 0.5, sol.exposure_s


def test_dash_timing_constrains_the_exposure(dashed_scene, constellation):
    """At the true time and place, a 10 % too long exposure must fit clearly worse when the dashes are timed;
    with the timing switched off (the old behaviour: dashed ends untrusted) it barely matters, since an
    untrusted end only stops the exposure from being too short to cover the visible streak."""
    from skyclock.fit import Fitter
    s = _settings()
    sol = SV.analyze(O.build(dashed_scene["det"].streaks, dashed_scene["plate"], s.epoch()), dashed_scene["plate"],
                     constellation, s).solutions[0]
    m = next(m for m in sol.matches)
    k = int(np.where(constellation.norad == m["norad"])[0][0])

    def dchi2(timed, ends=True):
        obs = O.build(dashed_scene["det"].streaks, dashed_scene["plate"], s.epoch())
        ob = next(o for o in obs if o.id == m["streak"])
        if not timed:
            ob.dash_f = ob.dash_g = None
        if not ends:
            ob.clipped = (True, True)
        f = Fitter([ob], [k], constellation, s.epoch(), scale_rad=dashed_scene["plate"].pixel_scale_rad())
        x = np.array([LAT, LON, s.epoch().seconds_from(WHEN + timedelta(seconds=5)), 10.0, 0.0, 0.0])
        c0 = np.sum(f.residuals(x) ** 2)
        x[3] = 11.0
        return np.sum(f.residuals(x) ** 2) - c0, ob

    d_on, ob = dchi2(True)
    assert ob.regular_dashes and len(ob.dash_f) >= 1
    d_off, _ = dchi2(False)
    d_inner, _ = dchi2(True, ends=False)  # the interior dashes alone, both ends hidden
    assert d_on > 25 and d_off < 0.2 * d_on and d_inner > 9, (d_on, d_off, d_inner)
