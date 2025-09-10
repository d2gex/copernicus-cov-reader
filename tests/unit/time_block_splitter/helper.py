from datetime import timedelta

from src.data_processing.time_block_splitter import TimeBlock, TimeBlockSplitter


def assert_partition(
    blocks: list[TimeBlock], start_iso: str, end_iso: str, splitter: TimeBlockSplitter
):
    """Structural invariants: sorted, adjacent, covering, positive lengths."""
    s = splitter._to_utc(start_iso)
    e = splitter._to_utc(end_iso)
    assert blocks, "No blocks produced"
    # Sorted by start
    starts = [b.start for b in blocks]
    assert starts == sorted(starts)
    # Coverage
    assert blocks[0].start == s
    assert blocks[-1].end == e
    # Positivity + adjacency + no overlap
    for i, b in enumerate(blocks):
        assert b.start < b.end, f"non-positive block at {i}"
        if i > 0:
            prev = blocks[i - 1]
            assert prev.end == b.start, f"gap/overlap between {i - 1} and {i}"
    # Sum durations equals span
    total = sum((b.end - b.start for b in blocks), timedelta(0))
    assert total == (e - s)
