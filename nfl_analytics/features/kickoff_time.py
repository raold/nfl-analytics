"""Calendar features from ``games.kickoff``.

Since the 2026-10-24 repair, ``games.kickoff`` holds the true instant (timestamptz, UTC).
Before it, 2000-2024 rows held the US/Eastern wall-clock time labelled UTC, and 2025 rows
held the date only. Every model trained on this database therefore saw Eastern wall-clock
hours, weekdays, dates and rest days. Derive those on the US/Eastern wall clock:

- SQL: ``EXTRACT(HOUR FROM g.kickoff AT TIME ZONE 'America/New_York')``
- pandas: ``kickoff_wall_clock(df["kickoff"]).dt.hour``

For 2000-2024 games this reproduces the pre-repair values exactly; for 2025 it supplies the
kickoff times that were missing. Comparisons between instants (``kickoff < NOW()``, odds
snapshot times) need no conversion.
"""

from __future__ import annotations

import pandas as pd

KICKOFF_TZ = "America/New_York"


def kickoff_wall_clock(kickoff) -> pd.Series:
    """Return kickoff times as naive US/Eastern wall-clock timestamps.

    Timezone-aware values (``timestamptz`` from the database) are converted to
    America/New_York. Naive values are assumed to be Eastern wall-clock times already
    (CSV exports and test fixtures) and are returned unchanged.
    """
    ts = pd.to_datetime(pd.Series(kickoff), errors="coerce")
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        return ts.dt.tz_convert(KICKOFF_TZ).dt.tz_localize(None)
    return ts
