from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.application.app_ds import BBoxPlan, Plan
from src.bounding_box.bounding_box import BoundingBox
from src.bounding_box.lat_bb_splitter import LatBandSplitter
from src.data_processing.time_block_splitter import TimeBlockSplitter

# ----------------- Step 1: planning -----------------


class PlanBuilder:
    """Builds a Plan using LatBandSplitter and TimeBlockSplitter.

    - BBoxes are split into equal-height latitude bands using LatBandSplitter.
    - Periods are produced either by `n_chunks` **or** by fixed `duration`.
      Exactly one must be provided.
    - BBoxes are indexed **north→south** (top→bottom) as 1..N and named `bbox_00X`.
    """

    def __init__(
        self,
        project_root: Path,
        master_bbox: BoundingBox,
        n_bands: int,
        start: str | datetime,
        end: str | datetime,
        n_chunks: Optional[int] = None,
        duration: Optional[timedelta] = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.master_bbox = master_bbox
        self.n_bands = int(n_bands)
        self.start = start
        self.end = end
        self.n_chunks = n_chunks
        self.duration = duration

    def build(self) -> Plan:
        if self.n_bands < 1:
            raise ValueError("n_bands must be >= 1.")
        if (self.n_chunks is None) == (self.duration is None):
            raise ValueError("Provide exactly one of n_chunks or duration.")

        # Split bbox using existing LatBandSplitter. We don't need input points for equal bands.
        splitter = LatBandSplitter(self.master_bbox, self.n_bands)
        empty_df = pd.DataFrame({"lon": [], "lat": []})
        band_results = splitter.split(empty_df)

        # Sort **north→south** and assign indices 1..N
        sorted_bands = sorted(
            band_results, key=lambda br: br.bbox.max_lat, reverse=True
        )
        bboxes: List[BBoxPlan] = [
            BBoxPlan(index=i + 1, bbox=br.bbox) for i, br in enumerate(sorted_bands)
        ]

        # Split time using the canonical TimeBlockSplitter
        tbs = TimeBlockSplitter()
        if self.n_chunks is not None:
            periods = tbs.split_by_chunks(self.start, self.end, self.n_chunks)
        else:
            periods = tbs.split_by_duration(self.start, self.end, self.duration)  # type: ignore[arg-type]

        return Plan(
            project_root=self.project_root, bboxes=bboxes, periods=list(periods)
        )
