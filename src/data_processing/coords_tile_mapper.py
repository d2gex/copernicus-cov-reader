from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.data_processing.kd_index import KDIndex


class CoordinatesToTileMapper:
    """Assigns tile_id to coordinate_data using a KD-tree over sea cells."""

    def __init__(self, kd_index: KDIndex) -> None:
        self.kd_index = kd_index

    def map(
        self,
        coordinates: pd.DataFrame,
        lon_col: str = "lon",
        lat_col: str = "lat",
        tolerance_deg: Optional[float] = None,
        out_col: str = "tile_id",
    ) -> pd.DataFrame:
        """Returns a copy of coordinates with a new tile_id column (-1 when no match)."""
        lons = np.asarray(coordinates[lon_col].values, dtype=np.float64)
        lats = np.asarray(coordinates[lat_col].values, dtype=np.float64)
        ids = self.kd_index.query_many(lons, lats, tolerance_deg=tolerance_deg)

        out = coordinates.copy()
        out[out_col] = ids
        return out
