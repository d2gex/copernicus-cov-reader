from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from src.bounding_box.bounding_box import BoundingBox
from src.data_processing.time_block_splitter import (
    TimeBlock as TBBlock,
)


@dataclass(frozen=True)
class ProductSpec:
    """Single Copernicus product we will download and flatten."""

    name: str
    dataset_id: str
    variables: Sequence[str]
    sea_land_mask: np.ndarray

    def slug(self) -> str:
        # Simple, constrained slug: lowercase, spaces→-, drop leading dots/underscores/hyphens.
        s = self.name.strip().lower()
        s = s.replace(" ", "-")
        s = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in s)
        while s and s[0] in {"-", "_", "."}:
            s = s[1:]
        return s or "product"


@dataclass(frozen=True)
class BBoxPlan:
    """A lat-band split result with a stable index for folder naming."""

    index: int  # 1-based, top (north) → bottom (south)
    bbox: BoundingBox

    def folder_name(self) -> str:
        return f"bbox_{self.index:03d}"


@dataclass(frozen=True)
class Plan:
    project_root: Path
    bboxes: List[BBoxPlan]
    periods: List[TBBlock]  # Use the canonical TimeBlock from TimeBlockSplitter


@dataclass(frozen=True)
class StaticSpec:
    """Static layer configuration used to build the sea/land boolean mask."""

    dataset_id: str
    mask_var: str
    is_bit: bool  # True if mask is bitfield; False if categorical/boolean
    sea_value: int | None = (
        None  # required when is_bit=True; optional when non-bit (can be inferred from long_name)
    )
