import numpy as np

from skyclock.detect import detect


def _line_image(p1, p2, H=800, W=1200, flux=400.0, seed=0, gap=None):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 100.0)
    n = 2000
    t = np.linspace(0, 1, n)
    if gap is not None:
        t = t[(t * 10 % 1) < gap]
    xs = p1[0] + t * (p2[0] - p1[0])
    ys = p1[1] + t * (p2[1] - p1[1])
    yy, xx = np.mgrid[0:H, 0:W]
    for x, y in zip(xs[::4], ys[::4]):
        sl = (slice(max(0, int(y) - 5), int(y) + 6), slice(max(0, int(x) - 5), int(x) + 6))
        img[sl] += flux * 4 / n * 50 * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * 1.2**2))
    for _ in range(60):  # stars
        x, y = rng.uniform(20, W - 20), rng.uniform(20, H - 20)
        sl = (slice(int(y) - 6, int(y) + 7), slice(int(x) - 6, int(x) + 7))
        img[sl] += 300 * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * 1.2**2))
    return rng.poisson(img).astype(np.float32)


def test_single_streak_endpoints():
    p1, p2 = (300.0, 200.0), (700.0, 420.0)
    det = detect(_line_image(p1, p2))
    assert len(det.streaks) == 1
    s = det.streaks[0]
    ends = sorted([tuple(s.p1), tuple(s.p2)])
    assert np.hypot(*np.subtract(ends[0], p1)) < 2.5
    assert np.hypot(*np.subtract(ends[1], p2)) < 2.5
    assert not s.clipped1 and not s.clipped2
    assert len(det.stars) > 30


def test_streak_touching_edge_is_flagged_clipped():
    det = detect(_line_image((-50.0, 300.0), (500.0, 500.0)))
    assert len(det.streaks) == 1
    s = det.streaks[0]
    assert s.clipped1 or s.clipped2


def test_detection_on_simulated_scene(scene):
    truth = scene["truth"]["streaks"]
    det = scene["det"]
    assert len(det.streaks) >= len(truth) - 1
    for st in det.streaks:  # each detection lies on a true track
        best = min(np.hypot(*np.subtract(st.p1, t["start_px"])) + np.hypot(*np.subtract(st.p2, t["end_px"])) for t in truth)
        best_r = min(np.hypot(*np.subtract(st.p2, t["start_px"])) + np.hypot(*np.subtract(st.p1, t["end_px"])) for t in truth)
        assert min(best, best_r) < 15
