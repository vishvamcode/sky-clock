import json

import numpy as np

from skyclock import validate as V


def test_calibrate_recovers_known_scale():
    """Errors drawn with twice the reported sigma must give a time scale near 2 and a place scale near 2."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        rows.append(V.Row(f"p{i}", "time", True, time_err_s=float(rng.normal(0, 2.0)), sigma_t_s=1.0))
        e, n = rng.normal(0, 2.0, 2)
        rows.append(V.Row(f"p{i}", "place", True, loc_z2=float(e * e + n * n)))
    c = V.calibrate(rows)
    assert c["enough"]
    assert abs(c["time_scale"] - 2.0) < 0.2
    assert abs(c["place_scale"] - 2.0) < 0.2


def test_benchmark_on_simulated_photos(tmp_path):
    n = V.make_simulated_set(tmp_path, 2, seed=3, catalog_errors=False)
    assert n == 2
    rows, summary, calib, report = V.score_folder(tmp_path, modes=("check", "time"))
    assert (tmp_path / "results.csv").exists() and report.exists()
    t = [r for r in rows if r.mode == "time"]
    assert len(t) == 2 and all(r.ok for r in t)
    assert all(abs(r.time_err_s) < 1.0 for r in t)  # exact catalogue: errors are tiny
    assert summary["check"]["consistent"] == 2
    truth = json.loads(next(tmp_path.glob("*.truth.json")).read_text())
    assert truth["catalog"] == "synthetic"
