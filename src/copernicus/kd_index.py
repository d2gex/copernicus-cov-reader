from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from src.copernicus.tile_catalog import TileCatalog


class KDIndex:
    """KD-tree over sea cell centers for nearest-tile lookup."""

    def __init__(self, catalog: TileCatalog, lat0: Optional[float] = None) -> None:
        """
        Builds a local planar index: x = lon * cos(lat0), y = lat.
        tile_id == KD index position (0..K-1).
        """
        self.catalog = catalog
        lon_sea, lat_sea = catalog.sea_tile_coords()
        self.lat0 = float(lat0) if lat0 is not None else float(np.nanmean(lat_sea))
        x = lon_sea * math.cos(math.radians(self.lat0))
        y = lat_sea
        coords = np.column_stack([x, y])
        self._tree = cKDTree(coords)

    def query_many(
        self, lons: np.ndarray, lats: np.ndarray, tolerance_deg: Optional[float] = None
    ) -> np.ndarray:
        """Returns nearest sea tile_id for each lon/lat; -1 when outside tolerance."""
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)
        if lons.size != lats.size:
            raise ValueError("lons and lats must have the same length.")
        if lons.size == 0:
            return np.empty((0,), dtype=np.int64)

        x = lons * math.cos(math.radians(self.lat0))
        y = lats
        dist, idx = self._tree.query(np.column_stack([x, y]), workers=-1)
        if tolerance_deg is not None:
            idx = np.where(dist <= float(tolerance_deg), idx, -1)
        return idx.astype(np.int64, copy=False)
