from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

from .artifact_enumerator import Artifact
from .storage_sink import StorageSink


class Policy(Enum):
    SKIP_IF_EXISTS = "skip_if_exists"
    ALWAYS_PUT = "always_put"


@dataclass
class SinkSummary:
    total: int
    skipped: int
    uploaded: int
    failed: int


def bulk_sink(
    artifacts: Iterable[Artifact],
    sink: StorageSink,
    policy: Policy,
    on_tick: Callable[[], None],
) -> SinkSummary:
    skipped = uploaded = failed = total = 0

    for a in artifacts:
        total += 1
        try:
            if policy is Policy.SKIP_IF_EXISTS and sink.exists(a.relative_key):
                skipped += 1
            else:
                sink.ingest(a.local_path, a.relative_key)
                uploaded += 1
        except Exception:
            failed += 1
            logging.exception("Failed to sink: %s", a.relative_key)
        finally:
            on_tick()

    return SinkSummary(total=total, skipped=skipped, uploaded=uploaded, failed=failed)
