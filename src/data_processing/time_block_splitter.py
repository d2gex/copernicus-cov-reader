from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import List, Union

DateLike = Union[str, datetime, date]


@dataclass(frozen=True)
class TimeBlock:
    idx: int
    start: datetime  # UTC-aware
    end: datetime  # UTC-aware


class TimeBlockSplitter:
    """
    Split a time interval [start, end) into blocks without assuming a resolution.
    Policy:
      - Date-only inputs → normalize to midnight UTC (T00:00:00Z).
      - Naive datetimes → treated as UTC (no conversion).
      - UTC datetimes (Z or +00:00) → accepted and normalized to tz=UTC.
      - Any non-UTC offset (e.g., +02:00) → ValueError.
      - Output blocks are half-open [start, end), last block may be shorter.
    """

    def split_by_chunks(
        self, start: DateLike, end: DateLike, n_chunks: int
    ) -> List[TimeBlock]:
        """
        Split [start, end) into exactly n_chunks equal-duration blocks.
        """
        s = self._to_utc(start)
        e = self._to_utc(end)
        if not (e > s):
            raise ValueError("end must be strictly greater than start")
        if n_chunks < 1:
            raise ValueError("n_chunks must be >= 1")

        step = (e - s) / n_chunks  # timedelta
        blocks: List[TimeBlock] = []
        cur = s
        for i in range(n_chunks):
            nxt = e if i == n_chunks - 1 else cur + step
            blocks.append(TimeBlock(idx=i, start=cur, end=nxt))
            cur = nxt
        return blocks

    def split_by_duration(
        self, start: DateLike, end: DateLike, block: timedelta
    ) -> List[TimeBlock]:
        """
        Split [start, end) into blocks of fixed 'block' duration.
        The final block is truncated to land exactly on 'end'.
        """
        if block <= timedelta(0):
            raise ValueError("block duration must be positive")

        s = self._to_utc(start)
        e = self._to_utc(end)
        if not (e > s):
            raise ValueError("end must be strictly greater than start")

        blocks: List[TimeBlock] = []
        cur = s
        idx = 0
        while cur < e:
            nxt = cur + block
            if nxt > e:
                nxt = e
            blocks.append(TimeBlock(idx=idx, start=cur, end=nxt))
            idx += 1
            cur = nxt
        return blocks

    def _to_utc(self, x: DateLike) -> datetime:
        """Normalize input to a UTC-aware datetime; reject non-UTC offsets.
        Policy:
          - date-only (YYYY-MM-DD) → midnight UTC
          - naive datetime → treated as UTC
          - UTC-aware (+00:00 or Z) → normalized to tz=UTC
          - any other offset → ValueError
        """
        # 1) Coerce to a datetime (no tz handling yet)
        if isinstance(x, datetime):
            dt = x
        elif isinstance(x, date):
            dt = datetime(x.year, x.month, x.day)  # naive; will be treated as UTC
        elif isinstance(x, str):
            s = x.strip()
            # date-only?
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                dt = datetime.fromisoformat(s)  # naive date → midnight
            else:
                # normalize 'Z' suffix for fromisoformat
                if s.endswith(("Z", "z")):
                    s = s[:-1] + "+00:00"
                try:
                    dt = datetime.fromisoformat(s)
                except ValueError as e:
                    raise ValueError(f"Invalid ISO-8601 datetime string: {x!r}") from e
        else:
            raise TypeError(f"Unsupported date-like type: {type(x)!r}")

        # 2) Single normalization policy
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        if dt.utcoffset() == timedelta(0):
            return dt.astimezone(timezone.utc)
        raise ValueError(
            "Non-UTC timezone offsets are not supported; pass UTC (Z/+00:00) or naive UTC."
        )
