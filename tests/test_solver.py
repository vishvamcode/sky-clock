"""End-to-end recovery on a simulated photo (the solver and simulator share propagation code, so these
tests check internal consistency and statistics, not agreement with the real sky)."""
from datetime import timedelta

import numpy as np
import pytest

from skyclock import observations as O, solve as SV
from skyclock.geometry import great_circle_distance_km

from conftest import LAT, LON, WHEN


def _run(scene, constellation, **kw):
    s = SV.Settings(**kw)
    obs = O.build(scene["det"].streaks, scene["plate"], s.epoch())
    return SV.analyze(obs, scene["plate"], constellation, s)


def _err(sol):
    return great_circle_distance_km(sol.lat, sol.lon, LAT, LON), (sol.t_start - WHEN).total_seconds()


def test_identify_recovers_truth(scene, constellation):
    r = _run(scene, constellation, mode="identify", time=WHEN + timedelta(seconds=20), lat=LAT + 0.05,
             lon=LON - 0.1, sigma_time_s=30, sigma_loc_km=10)
    sol = r.solutions[0]
    km, s = _err(sol)
    assert km < 2.0 and abs(s) < 1.0
    names = {t["name"] for t in scene["truth"]["streaks"]}
    assert all(m["name"] in names for m in sol.matches)
    assert len(sol.matches) >= 4


def test_check_accepts_true_claim(scene, constellation):
    r = _run(scene, constellation, mode="check", time=WHEN, lat=LAT, lon=LON, sigma_time_s=2, sigma_loc_km=1)
    assert r.verdict["status"] == "consistent"


@pytest.mark.parametrize("dt,dlat", [(30.0, 0.0), (0.0, 0.5), (3.0, 0.0)])
def test_check_rejects_false_claim(scene, constellation, dt, dlat):
    """Failure control: a claim 30 s, 55 km, or 3 s (10 sigma at 0.3 s tolerance) away must be rejected."""
    tol_t = 0.3 if dt == 3.0 else 2.0
    r = _run(scene, constellation, mode="check", time=WHEN + timedelta(seconds=dt), lat=LAT + dlat, lon=LON,
             sigma_time_s=tol_t, sigma_loc_km=1)
    assert r.verdict["status"] in ("inconsistent", "no-match")


def test_time_mode(scene, constellation):
    r = _run(scene, constellation, mode="time", lat=LAT, lon=LON, sigma_loc_km=1,
             window_start=WHEN - timedelta(minutes=30), window_end=WHEN + timedelta(minutes=30))
    km, s = _err(r.solutions[0])
    assert abs(s) < 1.0


def test_place_mode(scene, constellation):
    r = _run(scene, constellation, mode="place", time=WHEN, sigma_time_s=2)
    km, s = _err(r.solutions[0])
    assert km < 2.0


def test_blind_mode(scene, constellation):
    r = _run(scene, constellation, mode="blind", window_start=WHEN - timedelta(minutes=30),
             window_end=WHEN + timedelta(minutes=30))
    km, s = _err(r.solutions[0])
    assert km < 2.0 and abs(s) < 1.0
    # the best solution must beat its aliases on fit quality or streak count
    if len(r.solutions) > 1:
        a, b = r.solutions[0], r.solutions[1]
        assert len(a.matches) > len(b.matches) or a.chi2 < b.chi2


def test_single_streak_is_degenerate_for_place(scene, constellation):
    for st in scene["det"].streaks:
        st.enabled = st.id == 0
    try:
        r = _run(scene, constellation, mode="place", time=WHEN, sigma_time_s=2)
        assert not r.solutions and r.locus  # a set of lines, not a point
        t = _run(scene, constellation, mode="time", lat=LAT, lon=LON, sigma_loc_km=1,
                 window_start=WHEN - timedelta(minutes=10), window_end=WHEN + timedelta(minutes=10))
        assert abs(_err(t.solutions[0])[1]) < 1.0  # but the time is found when the place is known
    finally:
        for st in scene["det"].streaks:
            st.enabled = True


def test_uncertainty_is_calibrated_roughly(scene, constellation):
    r = _run(scene, constellation, mode="identify", time=WHEN, lat=LAT, lon=LON, sigma_time_s=30, sigma_loc_km=10)
    sol = r.solutions[0]
    km, s = _err(sol)
    sig_km = np.hypot(sol.sigma_east_km, sol.sigma_north_km)
    assert km < 4 * sig_km + 0.2
    assert abs(s) < 4 * sol.sigma_t_s
