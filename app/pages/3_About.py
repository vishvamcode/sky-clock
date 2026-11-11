"""About sky-clock: what it does, how well, privacy and responsible use."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skyclock import public  # noqa: E402

st.set_page_config(page_title="About sky-clock", layout="wide")
st.title("About sky-clock")
st.markdown("""
sky-clock estimates **when and where a photo was taken** from what is in its sky.

| In the photo | What it gives | Accuracy so far |
|---|---|---|
| Satellite streaks + stars (night) | time and place | about 0.1 to 1 km and 0.2 to 0.5 s on simulated photos |
| Vertical edges or a sea horizon + stars | latitude; longitude if the time is known | tens of km |
| The Moon or a planet + stars | the date | minutes (Moon, place known) to days (Jupiter, Saturn) |
| Upright objects + their shadows (day) | Sun altitude, then place or time | about 0.6 deg: hundreds of km, or minutes |

**Honest status.** These numbers come from simulated photos and from comparisons with JPL ephemerides. One
real night photo has been tried; it could not confirm the tool, because only one streak was usable and its time
was uncertain. A field test with known times and places is planned (see the validation checklist in the
repository). Treat results as estimates with the error bars shown, not as proof.

### Privacy
* Photos are processed in memory and **never written to disk**. They are gone when you close the page.
* Plate solving runs on this server. If you choose nova.astrometry.net instead, an EXIF-free grey copy of the
  photo is sent there.
* The only files stored are public satellite catalogues (cache).
* sky-clock itself sets no tracking cookies and collects no usage statistics. The hosting platform keeps its own
  access logs, as any web host does.

### Responsible use
sky-clock is for checking claims about photos (journalism, science, your own pictures). **Do not use it to
locate private people from their personal photos.** Searches that locate the photographer ask you to confirm
this first.

### Run your own copy
The code is open: `docker compose up` runs everything locally, with no limits. See the README.
""")
if public.PUBLIC:
    st.caption(f"This public site limits searches to {public.MAX_WINDOW_H:.0f} h and runs at most "
               f"{public.MAX_CONCURRENT} analyses at a time.")
