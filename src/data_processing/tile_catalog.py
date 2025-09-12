from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import xarray as xr

from src.data_processing.grid_spec import GridSpec


class TileCatalog:
    """Stable tile IDs on a fixed grid (sea cells only)."""

    def __init__(self, grid: GridSpec, sea_land_mask: np.ndarray) -> None:
        """
        sea_land_mask: boolean ny×nx; True=sea, False=land.
        Assigns consecutive IDs to sea cells; land=-1.
        """
        if sea_land_mask.shape != (grid.ny, grid.nx):
            raise ValueError("sea_land_mask shape must be (ny, nx).")
        self.grid = grid

        # Order sea_land_mask by rows and convert it to boo
        self.sea_land_mask = sea_land_mask
        # Build sea tile ID map
        self.tile_id_map, self._sea_j, self._sea_i = self.__build_sea_tile_id_map()
        # Build two separate arrays where each tile ID is mapped to (lon, lat)
        self._sea_lat = self.grid.lats[self._sea_j]
        self._sea_lon = self.grid.lons[self._sea_i]

    def __build_sea_tile_id_map(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns array of sea tile IDs (0..K-1). tile_id_map"""
        tile_id_map = np.full((self.grid.ny, self.grid.nx), -1, dtype=np.int64)
        # get the index of sea cells only
        sea_flat_idx = np.flatnonzero(self.sea_land_mask.ravel(order="C"))
        # Flatten the tile_id_map and assign the sea tile IDs in ascending order
        tile_id_map.ravel(order="C")[sea_flat_idx] = np.arange(
            sea_flat_idx.size, dtype=np.int64
        )

        # It gets the index of sea cells only and splits into two separate j, i dimensions.
        jj, ii = np.nonzero(self.sea_land_mask)
        return tile_id_map, jj, ii

    @classmethod
    def from_dataset(
        cls, ds: xr.Dataset, mask: Optional[np.ndarray] = None
    ) -> "TileCatalog":
        grid = GridSpec.from_dataset(ds)
        return cls(grid=grid, sea_land_mask=mask)

    def sea_cell_ids(self, tile_id: int) -> Tuple[int, int]:
        """Returns (j,i) indices for a tile ID that matches a sea tile."""
        if tile_id < 0 or tile_id >= self._sea_i.size:
            raise IndexError("tile_id out of range.")
        return int(self._sea_j[tile_id]), int(self._sea_i[tile_id])

    def sea_cell_coords(self, tile_id: int) -> Tuple[float, float]:
        """Returns (lon, lat) cell for a tile ID."""
        return float(self._sea_lon[tile_id]), float(self._sea_lat[tile_id])

    def sea_tile_ids(self) -> np.ndarray:
        """Returns all sea tile IDs."""
        return self.tile_id_map[self.tile_id_map >= 0]

    def sea_tile_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns all sea tile coordinates."""
        return self._sea_lon, self._sea_lat
