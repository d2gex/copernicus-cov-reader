from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ColumnConfig:
    tile_id: str = "tile_id"
    lon_center: str = "tile_lon_center"
    lat_center: str = "tile_lat_center"
    time: str = "time"
    depth: str = "depth"


class TileDaysBuilder:
    """
    Pure in-memory builder:
      per-day rows: tile_id, tile_lon_center, tile_lat_center, time, deepest_depth

    - 'time' is parsed and floored to day; column name remains 'time'
    - 'deepest_depth' = max depth across ALL hauls of the tile
    - fails fast if any dates are unparseable
    - asserts that (tile_id, time) pairs in output match those in enriched hauls
    - raises if any tile has all-NaN depths
    """

    def __init__(
        self,
        columns: ColumnConfig | None = None,
        *,
        dayfirst: bool = True,
    ) -> None:
        self.columns = columns or ColumnConfig()
        self.dayfirst = dayfirst

    def build_per_day(self, enriched: pd.DataFrame) -> pd.DataFrame:
        cols = self.columns
        required = (
            cols.tile_id,
            cols.lon_center,
            cols.lat_center,
            cols.time,
            cols.depth,
        )
        missing = [c for c in required if c not in enriched.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        # Parse & floor time (FAIL FAST if anything is unparseable)
        t = pd.to_datetime(
            enriched[cols.time],
            errors="coerce",
            utc=True,
            dayfirst=self.dayfirst,
        )
        if t.isna().any():
            bad = enriched.loc[t.isna(), cols.time].astype(str)
            examples = bad.dropna().unique()[:10].tolist()
            raise ValueError(
                f"Unparseable dates in '{cols.time}': {t.isna().sum()} rows. "
                f"Examples: {examples}"
            )

        enriched = enriched.copy()
        enriched[cols.time] = t.dt.floor("D")

        # Compute per-tile deepest depth (Series indexed by tile_id)
        depth_num = pd.to_numeric(enriched[cols.depth], errors="coerce")
        max_depth_per_tile = depth_num.groupby(enriched[cols.tile_id]).max()

        bad_tiles = max_depth_per_tile[max_depth_per_tile.isna()].index.tolist()
        if bad_tiles:
            raise ValueError(f"Tiles with all-NaN depths: {bad_tiles}")

        # Build per-day unique rows per tile
        per_day = (
            enriched[[cols.tile_id, cols.lon_center, cols.lat_center, cols.time]]
            .drop_duplicates([cols.tile_id, cols.time])
            .sort_values([cols.tile_id, cols.time])
            .reset_index(drop=True)
        )

        # Attach deepest depth
        per_day = per_day.copy()
        per_day["deepest_depth"] = per_day[cols.tile_id].map(max_depth_per_tile)

        # Assert tile/date coverage matches enriched hauls exactly
        expected_pairs = set(
            map(tuple, enriched[[cols.tile_id, cols.time]].drop_duplicates().to_numpy())
        )
        actual_pairs = set(
            map(tuple, per_day[[cols.tile_id, cols.time]].drop_duplicates().to_numpy())
        )

        if expected_pairs != actual_pairs:
            missing_pairs = list(expected_pairs - actual_pairs)[:10]
            extra_pairs = list(actual_pairs - expected_pairs)[:10]
            raise AssertionError(
                "Mismatch between haul tile/dates and tiles_with_date_db tile/dates. "
                f"missing={len(expected_pairs - actual_pairs)} "
                f"extra={len(actual_pairs - expected_pairs)} "
                f"missing_examples={missing_pairs} "
                f"extra_examples={extra_pairs}"
            )

        return per_day
