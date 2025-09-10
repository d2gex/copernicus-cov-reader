from datetime import timedelta

import pytest

from src.data_processing.time_block_splitter import TimeBlockSplitter
from tests.unit.time_block_splitter.helper import assert_partition


def test_duration_date_only_24x1h(splitter: TimeBlockSplitter, interval_date_only_1d):
    start, end = interval_date_only_1d
    blocks = splitter.split_by_duration(start, end, block=timedelta(hours=1))
    assert len(blocks) == 24
    assert all((b.end - b.start) == timedelta(hours=1) for b in blocks)
    assert_partition(blocks, start, end, splitter)


def test_duration_6h_over_3days(splitter: TimeBlockSplitter, interval_multi_days):
    start, end = interval_multi_days
    blocks = splitter.split_by_duration(start, end, block=timedelta(hours=6))
    # 72h / 6h = 12 blocks
    assert len(blocks) == 12
    assert all((b.end - b.start) == timedelta(hours=6) for b in blocks)
    assert_partition(blocks, start, end, splitter)


def test_duration_block_larger_than_span(
    splitter: TimeBlockSplitter, interval_date_only_1d
):
    start, end = interval_date_only_1d
    blocks = splitter.split_by_duration(start, end, block=timedelta(days=2))
    assert len(blocks) == 1
    assert_partition(blocks, start, end, splitter)


def test_duration_tail_shorter(splitter: TimeBlockSplitter):
    start = "2020-01-01T00:00:00Z"
    end = "2020-01-01T05:30:00Z"
    blocks = splitter.split_by_duration(start, end, block=timedelta(hours=2))
    # Expect 3 blocks: [00:00,02:00), [02:00,04:00), [04:00,05:30)
    assert len(blocks) == 3
    assert (blocks[0].end - blocks[0].start) == timedelta(hours=2)
    assert (blocks[1].end - blocks[1].start) == timedelta(hours=2)
    assert (blocks[2].end - blocks[2].start) == timedelta(hours=1, minutes=30)
    assert_partition(blocks, start, end, splitter)


@pytest.mark.parametrize("delta", [timedelta(0), timedelta(days=-1)])
def test_duration_invalid_block_raises(
    splitter: TimeBlockSplitter, interval_date_only_1d, delta
):
    start, end = interval_date_only_1d
    with pytest.raises(ValueError):
        splitter.split_by_duration(start, end, block=delta)


@pytest.mark.parametrize(
    "pair",
    [
        ("2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),
        ("2020-01-02T00:00:00Z", "2020-01-01T00:00:00Z"),
    ],
)
def test_duration_end_le_start_raises(splitter: TimeBlockSplitter, pair):
    s, e = pair
    with pytest.raises(ValueError):
        splitter.split_by_duration(s, e, block=timedelta(hours=1))


# --------- D) Equivalence sanity between methods ---------


def test_equivalence_chunks_vs_duration_date_only(
    splitter: TimeBlockSplitter, interval_date_only_1d
):
    start, end = interval_date_only_1d
    a = splitter.split_by_chunks(start, end, n_chunks=24)
    b = splitter.split_by_duration(start, end, block=timedelta(hours=1))
    assert [(x.start, x.end) for x in a] == [(y.start, y.end) for y in b]


def test_equivalence_chunks_vs_duration_midday(
    splitter: TimeBlockSplitter, interval_1d_midday
):
    start, end = interval_1d_midday
    a = splitter.split_by_chunks(start, end, n_chunks=8)  # 24h / 8 = 3h
    b = splitter.split_by_duration(start, end, block=timedelta(hours=3))
    assert [(x.start, x.end) for x in a] == [(y.start, y.end) for y in b]
