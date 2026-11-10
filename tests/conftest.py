import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyclock import catalog as C, simulate as S  # noqa: E402

WHEN = datetime(2026, 11, 10, 16, 10, tzinfo=timezone.utc)
LAT, LON = 59.33, 18.06


@pytest.fixture(scope="session")
def constellation():
    return C.synthetic_constellation(WHEN - timedelta(hours=6))


@pytest.fixture(scope="session")
def scene(constellation):
    """Simulated fixed-tripod photo over Stockholm with its true WCS and detections."""
    from skyclock import detect as D, platesolve as PS
    img, wcs, truth = S.simulate_photo(constellation, WHEN, LAT, LON, exposure_s=10, alt_deg=50, az_deg=200,
                                       cam=S.Camera(fov_x_deg=70), seed=3)
    det = D.detect(img)
    plate = PS.from_wcs_object(wcs, img.shape)
    return dict(img=img, wcs=wcs, truth=truth, det=det, plate=plate)
