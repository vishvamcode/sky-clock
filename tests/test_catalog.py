import numpy as np

from skyclock import catalog as C

ISS = """ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537
"""


def test_parse_tle():
    c = C.load_text(ISS)
    assert len(c) == 1 and c.names[0] == "ISS (ZARYA)" and c.norad[0] == 25544
    assert abs(c.mean_motion[0] - 15.72125391) < 1e-6


def test_parse_omm_json_and_merge():
    rec = dict(OBJECT_NAME="TEST", OBJECT_ID="2020-001A", EPOCH="2026-11-01T00:00:00.000000", MEAN_MOTION=15.05,
               ECCENTRICITY=0.0001, INCLINATION=53.0, RA_OF_ASC_NODE=10.0, ARG_OF_PERICENTER=90.0, MEAN_ANOMALY=0.0,
               EPHEMERIS_TYPE=0, CLASSIFICATION_TYPE="U", NORAD_CAT_ID=99999, ELEMENT_SET_NO=999, REV_AT_EPOCH=1,
               BSTAR=0.0001, MEAN_MOTION_DOT=0.0, MEAN_MOTION_DDOT=0.0)
    import json
    c = C.load_text(json.dumps([rec]))
    assert len(c) == 1 and c.norad[0] == 99999
    m = C.merge(C.load_text(ISS), c)
    assert sorted(m.norad.tolist()) == [25544, 99999]


def test_leo_filter(constellation):
    assert len(constellation.leo_only(11)) == len(constellation)
    assert np.all(constellation.mean_motion > 14)


def test_github_archive_coverage():
    from datetime import datetime, timezone
    from skyclock.catalog import github_archive_for
    assert github_archive_for(datetime(2023, 8, 18, 2, 7, tzinfo=timezone.utc))["repo"] == "OndroNR/tle-archive"
    assert github_archive_for(datetime(2026, 11, 1, tzinfo=timezone.utc))["repo"] == "crb0229/overhead"
    assert github_archive_for(datetime(2025, 1, 1, tzinfo=timezone.utc)) is None
