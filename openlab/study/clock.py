"""The times openlab writes, and the one way it reads them back.

Every timestamp carries its offset (`2026-10-06T19:39:55-04:00`). A time without one is read in the reader's
zone: a container on UTC wrote `23:46`, a browser at -04:00 read it as four hours ahead, and runs a minute old
showed "in 4h" beside a seal that said 19:39. Older logs hold naive times; `instant` reads them as local, the
zone they were written in.
"""

from __future__ import annotations

import datetime as dt


def now() -> str:
    """Now, to the second, ISO 8601 with the local offset."""
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def utc() -> dt.datetime:
    """Now as a naive UTC datetime, the form a study's decision and label times compare in."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def stamp() -> str:
    """Now for a footer a person reads: `2026-10-06 19:55 -04:00`, or `... 23:55 UTC` on a UTC machine."""
    d = dt.datetime.now().astimezone()
    off = d.strftime("%z")
    return d.strftime("%Y-%m-%d %H:%M ") + ("UTC" if off == "+0000" else f"{off[:3]}:{off[3:]}")


def instant(v) -> float:
    """Epoch seconds of a written time (naive = local), so times with different offsets order correctly;
    0.0 when it is not a time, which sorts it oldest."""
    try:
        return dt.datetime.fromisoformat(str(v).strip().replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
