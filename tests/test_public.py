import importlib
from datetime import datetime, timedelta, timezone

import pytest


def _load(monkeypatch, public, n=1):
    monkeypatch.setenv("SKYCLOCK_PUBLIC", "1" if public else "0")
    monkeypatch.setenv("SKYCLOCK_MAX_CONCURRENT", str(n))
    import skyclock.public as P
    return importlib.reload(P)


def test_window_cap(monkeypatch):
    P = _load(monkeypatch, True)
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert P.check_window(t, t + timedelta(hours=5)) is None
    assert "at most 6 h" in P.check_window(t, t + timedelta(hours=7))
    P = _load(monkeypatch, False)
    assert P.check_window(t, t + timedelta(hours=70)) is None  # private copy: no cap


def test_concurrency_limit(monkeypatch):
    P = _load(monkeypatch, True, n=1)
    with P.analysis_slot():
        with pytest.raises(P.Busy):
            with P.analysis_slot(wait_s=0.05):
                pass
    with P.analysis_slot(wait_s=0.05):  # released again
        pass
    _load(monkeypatch, False)
