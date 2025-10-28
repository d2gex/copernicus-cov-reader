from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Tuple

import pandas as pd


@dataclass(frozen=True)
class ColumnConfig:
    tile_id: str = "tile_id"
    lon_center: str = "tile_lon_center"
    lat_center: str = "tile_lat_center"
    time: str = "time"


class TileDaysBuilder:
    """
    Pure in-memory builder:
      (a) per-day rows: tile_id, tile_lon_center, tile_lat_center, date
      (b) minimal contiguous date ranges: tile_id, centers, start_date, end_date, n_days

    No reading/writing here. Pass in DataFrames, get DataFrames back.
    """

    def __init__(self, columns: ColumnConfig | None = None) -> None:
        self.columns = columns or ColumnConfig()

    def build_per_day(self, enriched: pd.DataFrame) -> pd.DataFrame:
        cols = self.columns
        required = (cols.tile_id, cols.lon_center, cols.lat_center, cols.time)
        missing = [c for c in required if c not in enriched.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        t = pd.to_datetime(enriched[cols.time], errors="coerce", utc=True)
        valid = ~t.isna()
        if not valid.all():
            enriched = enriched.loc[valid].copy()
            t = t.loc[valid]
        enriched[cols.time] = t.dt.floor("D")

        out = (
            enriched[[cols.tile_id, cols.lon_center, cols.lat_center, cols.time]]
            .drop_duplicates([cols.tile_id, cols.time])
            .sort_values([cols.tile_id, cols.time])
            .rename(columns={cols.time: "date"})
            .reset_index(drop=True)
        )
        return out

    @staticmethod
    def _ranges_from_sorted_days(
        days: Iterable[pd.Timestamp],
    ) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp, int]]:
        start = None
        prev = None
        for d in days:
            if start is None:
                start = d
                prev = d
            else:
                is_consecutive = d == (prev + timedelta(days=1))
                if is_consecutive:
                    prev = d
                else:
                    n_days = (prev - start).days + 1
                    yield (start, prev, n_days)
                    start = d
                    prev = d
        if start is not None:
            n_days = (prev - start).days + 1  # type: ignore[arg-type]
            yield (start, prev, n_days)

    def build_ranges(self, per_day_df: pd.DataFrame) -> pd.DataFrame:
        cols = self.columns
        required = (cols.tile_id, cols.lon_center, cols.lat_center, "date")
        missing = [c for c in required if c not in per_day_df.columns]
        if missing:
            raise KeyError(f"Missing required columns in per_day_df: {missing}")

        rows = []
        gcols = [cols.tile_id, cols.lon_center, cols.lat_center]
        for (tile_id, lon_c, lat_c), g in per_day_df.groupby(
            gcols, sort=True, as_index=False
        ):
            # g["date"] is already sorted by build_per_day
            days = list(g["date"])
            for s, e, n in self._ranges_from_sorted_days(days):
                rows.append(
                    {
                        cols.tile_id: tile_id,
                        cols.lon_center: float(lon_c),
                        cols.lat_center: float(lat_c),
                        "start_date": s.date().isoformat(),
                        "end_date": e.date().isoformat(),
                        "n_days": int(n),
                    }
                )
        return pd.DataFrame(rows)
