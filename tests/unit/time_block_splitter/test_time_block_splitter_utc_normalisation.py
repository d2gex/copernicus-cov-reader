from datetime import timezone

import pytest

from src.data_processing.time_block_splitter import TimeBlockSplitter


def test_to_utc_happy_table(splitter: TimeBlockSplitter, utc_table_happy):
    for inp, expected_iso in utc_table_happy:
        dt = splitter._to_utc(inp)
        assert dt.isoformat() == expected_iso


@pytest.mark.parametrize(
    "bad",
    [
        "2020-01-01T12:30:00+02:00",
        "2020-01-01T12:30:00-05:00",
    ],
)
def test_to_utc_rejects_non_utc_offsets_strings(splitter: TimeBlockSplitter, bad):
    with pytest.raises(ValueError):
        splitter._to_utc(bad)


def test_to_utc_rejects_invalid_string(splitter: TimeBlockSplitter):
    with pytest.raises(ValueError):
        splitter._to_utc("not-a-date")


def test_to_utc_outputs_are_utc_aware(splitter: TimeBlockSplitter, utc_table_happy):
    for inp, _ in utc_table_happy:
        dt = splitter._to_utc(inp)
        assert dt.tzinfo is not None
        assert dt.utcoffset() == splitter._to_utc("2020-01-01").utcoffset()  # zero
        assert dt.tzinfo is timezone.utc
