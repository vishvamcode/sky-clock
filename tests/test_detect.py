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


def _foreground_scene(seed=1):
    """Sky with a gradient, a dark 'tree line' at the bottom full of bright leaf gaps and short edges,
    and one streak that runs from open sky into the trees."""
    rng = np.random.default_rng(seed)
    H, W = 800, 1200
    yy, xx = np.mgrid[0:H, 0:W]
    img = 100.0 + 60.0 * yy / H  # light-polluted sky, brighter towards the horizon
    tree_top = 520 + 40 * np.sin(xx / 90.0)
    tree = yy > tree_top
    img = np.where(tree, 12.0, img)
    for _ in range(400):  # leaf gaps: bright dots and short bright lines inside the trees
        x, y = rng.uniform(0, W), rng.uniform(560, H)
        if rng.random() < 0.5:
            L, a = rng.uniform(15, 60), rng.uniform(0, np.pi)
            for t in np.linspace(0, L, int(L)):
                xi, yi = int(x + t * np.cos(a)), int(y + t * np.sin(a))
                if 0 <= xi < W and 0 <= yi < H:
                    img[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2] += 120
        else:
            img[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3] += 250
    sky_stars = []
    for _ in range(60):
        x, y = rng.uniform(20, W - 20), rng.uniform(20, 420)
        sl = (slice(int(y) - 6, int(y) + 7), slice(int(x) - 6, int(x) + 7))
        img[sl] += 300 * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * 1.2**2))
        sky_stars.append((x, y))
    p1, p2 = np.array([300.0, 250.0]), np.array([800.0, 700.0])  # ends inside the trees
    for t in np.linspace(0, 1, 3000):
        x, y = p1 + t * (p2 - p1)
        if tree[int(y), int(x)]:
            continue  # hidden behind the trees
        sl = (slice(int(y) - 5, int(y) + 6), slice(int(x) - 5, int(x) + 6))
        img[sl] += 8.0 * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * 1.2**2))
    return rng.poisson(np.clip(img, 0, None)).astype(np.float32), tree, p1


def test_trees_do_not_make_streaks_or_stars_and_hide_the_end():
    img, tree, p1 = _foreground_scene()
    det = detect(img)
    assert det.sky_mask is not None
    assert len(det.streaks) == 1, [(s.p1, s.p2) for s in det.streaks]
    s = det.streaks[0]
    a, b = (s.p1, s.p2) if np.hypot(*np.subtract(s.p1, p1)) < np.hypot(*np.subtract(s.p2, p1)) else (s.p2, s.p1)
    ca, cb = (s.clipped1, s.clipped2) if a is s.p1 else (s.clipped2, s.clipped1)
    assert np.hypot(*np.subtract(a, p1)) < 2.5 and not ca  # the real start in open sky is kept
    assert cb  # the end that disappears behind the trees is not trusted for timing
    assert all(not tree[int(round(st.y)), int(round(st.x))] for st in det.stars)
    # without the mask the leaf gaps produce false streaks (checks that this test can fail)
    assert len(detect(img, use_sky_mask=False).streaks) > 1


def test_fading_end_is_not_trusted():
    H, W = 800, 1200
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 100.0)
    p1, p2 = np.array([200.0, 300.0]), np.array([900.0, 450.0])
    for t in np.linspace(0, 1, 4000):
        x, y = p1 + t * (p2 - p1)
        f = min(1.0, t / 0.3)  # brightness ramps up over the first 30 % (entering sunlight / fading)
        sl = (slice(int(y) - 5, int(y) + 6), slice(int(x) - 5, int(x) + 6))
        img[sl] += 2.5 * f * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * 1.2**2))
    det = detect(rng.poisson(img).astype(np.float32))
    assert len(det.streaks) == 1
    s = det.streaks[0]
    faded_is_1 = np.hypot(*np.subtract(s.p1, p1)) < np.hypot(*np.subtract(s.p2, p1))
    assert (s.clipped1 if faded_is_1 else s.clipped2)
    assert not (s.clipped2 if faded_is_1 else s.clipped1)
