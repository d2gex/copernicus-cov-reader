from datetime import date, datetime, timezone
from typing import Any, List, Tuple

import pytest

from src.data_processing.time_block_splitter import TimeBlockSplitter


@pytest.fixture
def splitter() -> TimeBlockSplitter:
    return TimeBlockSplitter()


@pytest.fixture
def utc_table_happy() -> List[Tuple[Any, str]]:
    # (input, expected ISO string in UTC)
    return [
        ("2020-01-01", "2020-01-01T00:00:00+00:00"),
        ("2020-01-01T12:30:00Z", "2020-01-01T12:30:00+00:00"),
        ("2020-01-01T12:30:00", "2020-01-01T12:30:00+00:00"),
        ("2020-01-01T12:30:00+00:00", "2020-01-01T12:30:00+00:00"),
        ("2020-01-01T12:30:00.123Z", "2020-01-01T12:30:00.123000+00:00"),
        (date(2020, 1, 1), "2020-01-01T00:00:00+00:00"),
        (datetime(2020, 1, 1, 12, 30), "2020-01-01T12:30:00+00:00"),
        (
            datetime(2020, 1, 1, 12, 30, tzinfo=timezone.utc),
            "2020-01-01T12:30:00+00:00",
        ),
    ]


@pytest.fixture
def utc_table_rejects() -> list:
    return [
        "2020-01-01T12:30:00+02:00",
        "2020-01-01T12:30:00-05:00",
        "not-a-date",
    ]


@pytest.fixture
def interval_date_only_1d() -> tuple[str, str]:
    # 24h span, date-only
    return "2020-01-01", "2020-01-02"


@pytest.fixture
def interval_1d_midday() -> tuple[str, str]:
    # 24h span, preserving time-of-day
    return "2020-01-01T06:00:00Z", "2020-01-02T06:00:00Z"


@pytest.fixture
def interval_multi_days() -> tuple[str, str]:
    # 72h span (3 days)
    return "2020-01-01", "2020-01-04"
