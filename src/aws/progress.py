from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass
class ProgressReporter:
    total: int
    step: float = 0.10  # 10%
    _milestone: int = 0
    _seen: int = 0

    def start(self, label: str) -> None:
        self._milestone = 0
        self._seen = 0
        logging.info("%s: 0%% (0/%d)", label, self.total)

    def tick(self, label: str) -> None:
        if self.total <= 0:
            return
        self._seen += 1
        pct = int((self._seen / self.total) * 100)
        next_threshold = int(self.step * 100) * (self._milestone + 1)
        if pct >= next_threshold:
            self._milestone += 1
            logging.info(
                "%s: %d%% (%d/%d)", label, min(pct, 100), self._seen, self.total
            )

    def finish(self, label: str, summary: str) -> None:
        logging.info("%s: 100%% (%d/%d) — %s", label, self._seen, self.total, summary)
