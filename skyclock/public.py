"""Settings for running sky-clock as a public website (SKYCLOCK_PUBLIC=1).

A public site is shared by strangers on a small free server, so it gets limits that a private copy does
not need:

* searches are capped (``MAX_WINDOW_H`` hours) and at most ``MAX_CONCURRENT`` analyses run at once;
* nobody is asked for passwords: the Space-Track login field is hidden (use the public archive or
  CelesTrak instead, or run your own copy for Space-Track);
* every mode, not only the searches, asks for the responsible-use confirmation.

Photos are never written to disk, in either setting.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager

PUBLIC = os.environ.get("SKYCLOCK_PUBLIC", "") not in ("", "0", "false", "False")
MAX_WINDOW_H = float(os.environ.get("SKYCLOCK_MAX_WINDOW_H", "6"))
MAX_CONCURRENT = int(os.environ.get("SKYCLOCK_MAX_CONCURRENT", "2"))

_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


class Busy(RuntimeError):
    pass


@contextmanager
def analysis_slot(wait_s: float = 20.0):
    """Hold one of the shared analysis slots; raise Busy if none frees up within ``wait_s``."""
    if not PUBLIC:
        yield
        return
    if not _slots.acquire(timeout=wait_s):
        raise Busy("The public server is busy with other analyses. Please try again in a minute, or run "
                   "your own copy (see the README).")
    try:
        yield
    finally:
        _slots.release()


def check_window(start, end):
    """Error text if a search window is too long for the public site, else None."""
    if not PUBLIC or start is None or end is None:
        return None
    h = (end - start).total_seconds() / 3600
    if h > MAX_WINDOW_H:
        return (f"On the public site a search window can be at most {MAX_WINDOW_H:.0f} h (yours is {h:.1f} h). "
                "Narrow it, or run your own copy for longer searches.")
    return None
