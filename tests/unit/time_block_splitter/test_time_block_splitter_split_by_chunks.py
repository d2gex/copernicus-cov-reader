from datetime import timedelta

import pytest

from src.data_processing.time_block_splitter import TimeBlockSplitter
from tests.unit.time_block_splitter.helper import assert_partition


def test_chunks_date_only_day_24(splitter: TimeBlockSplitter, interval_date_only_1d):
    start, end = interval_date_only_1d
    blocks = splitter.split_by_chunks(start, end, n_chunks=24)
    assert len(blocks) == 24
    one_hour = timedelta(hours=1)
    assert all((b.end - b.start) == one_hour for b in blocks)
    assert_partition(blocks, start, end, splitter)


def test_chunks_midday_24_preserves_time(
    splitter: TimeBlockSplitter, interval_1d_midday
):
    start, end = interval_1d_midday
    blocks = splitter.split_by_chunks(start, end, n_chunks=24)
    assert len(blocks) == 24
    one_hour = timedelta(hours=1)
    assert blocks[0].start.isoformat() == splitter._to_utc(start).isoformat()
    assert blocks[-1].end.isoformat() == splitter._to_utc(end).isoformat()
    assert all((b.end - b.start) == one_hour for b in blocks)
    assert_partition(blocks, start, end, splitter)


def test_chunks_one_block(splitter: TimeBlockSplitter, interval_date_only_1d):
    start, end = interval_date_only_1d
    blocks = splitter.split_by_chunks(start, end, n_chunks=1)
    assert len(blocks) == 1
    assert blocks[0].start == splitter._to_utc(start)
    assert blocks[0].end == splitter._to_utc(end)
    assert_partition(blocks, start, end, splitter)


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_chunks_invalid_n_raises(
    splitter: TimeBlockSplitter, interval_date_only_1d, bad
):
    start, end = interval_date_only_1d
    with pytest.raises(ValueError):
        splitter.split_by_chunks(start, end, n_chunks=bad)


@pytest.mark.parametrize(
    "pair",
    [
        ("2020-01-01", "2020-01-01"),  # equal
        ("2020-01-02", "2020-01-01"),  # reversed
    ],
)
def test_chunks_end_le_start_raises(splitter: TimeBlockSplitter, pair):
    s, e = pair
    with pytest.raises(ValueError):
        splitter.split_by_chunks(s, e, n_chunks=2)
