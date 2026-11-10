import shutil

import numpy as np
import pytest

from conftest import LAT, LON, WHEN

needs_astrometry = pytest.mark.skipif(shutil.which("solve-field") is None or shutil.which("query-starkd") is None,
                                      reason="astrometry.net not installed")


@needs_astrometry
def test_local_plate_solve_real_stars(constellation):
    from skyclock import detect as D, platesolve as PS, simulate as S
    from skyclock.geometry import radec_to_unit
    img, wcs, _ = S.simulate_photo(constellation, WHEN, LAT, LON, exposure_s=10, alt_deg=50, az_deg=200,
                                   cam=S.Camera(fov_x_deg=70), seed=3, real_stars=True, star_mag_limit=7.5)
    det = D.detect(img)
    p = PS.solve_local(det.stars, img.shape)
    assert p is not None
    H, W = img.shape
    xs, ys = np.array([W / 2, 300, W - 300]), np.array([H / 2, 300, H - 300])
    ra, dec = wcs.all_pix2world(xs, ys, 0)
    err = np.degrees(np.arccos(np.clip(np.sum(p.pix2unit(xs, ys) * radec_to_unit(ra, dec), 1), -1, 1))) * 3600
    assert np.all(err < 0.5 * np.degrees(p.pixel_scale_rad()) * 3600)  # better than half a pixel
