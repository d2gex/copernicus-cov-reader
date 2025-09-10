from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import xarray as xr

from src.data_processing.tile_catalog import TileCatalog


class TimeSeriesExtractor:
    """Emits tidy (time, tile_id, value) rows for a variable on a fixed grid."""

    def __init__(
        self,
        catalog: TileCatalog,
        var_name: str,
        depth: Optional[float] = None,
        depth_name: Optional[str] = None,
    ) -> None:
        self.catalog = catalog
        self.var_name = var_name
        self.depth = depth
        self.depth_name = depth_name

    def extract(self, ds: xr.Dataset) -> pd.DataFrame:
        """Validates grid; selects var/depth; applies catalog mask; returns (time, tile_id, value)."""
        self.catalog.grid.validate(ds)

        da = ds[self.var_name]
        if self.depth is not None and self.depth_name and self.depth_name in da.dims:
            da = da.sel({self.depth_name: self.depth})

        # Order dimensions to (time?, lat, lon, ...).
        time_name = "time" if "time" in da.dims else None
        order: Sequence[str] = tuple(
            n
            for n in (time_name, self.catalog.grid.lat_name, self.catalog.grid.lon_name)
            if n
        )
        da = da.transpose(*order, ...)

        # Apply sea mask; stack lat/lon into single dimension with row-major order.
        sea_mask = xr.DataArray(
            self.catalog.valid_mask,
            dims=(self.catalog.grid.lat_name, self.catalog.grid.lon_name),
        )
        da = da.where(sea_mask)
        da_stacked = da.stack(
            cell=(self.catalog.grid.lat_name, self.catalog.grid.lon_name)
        )

        # Align stacked cells with stable tile IDs; keep only sea cells.
        tile_id_flat = self.catalog.tile_id_map.ravel(order="C")
        da_stacked = da_stacked.assign_coords(cell_tile_id=("cell", tile_id_flat))
        da_sea = da_stacked.where(da_stacked["cell_tile_id"] >= 0, drop=True)

        ser = da_sea.to_series().dropna()  # MultiIndex → Series
        df = ser.rename("value").reset_index()

        # Standardize column names and order.
        if "cell_tile_id" in df.columns:
            df = df.rename(columns={"cell_tile_id": "tile_id"})
        cols = [c for c in ("time", "tile_id", "value") if c in df.columns]
        return df[cols]
