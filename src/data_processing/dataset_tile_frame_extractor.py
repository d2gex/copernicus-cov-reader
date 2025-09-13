from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from src.data_processing.tile_catalog import TileCatalog


class DatasetTileFrameExtractor:
    """
    Flatten one or multiple variables from an xarray.Dataset into tidy rows aligned
    with stable sea-only tile_ids.

    """

    def __init__(
        self,
        catalog: TileCatalog,
        bbox_id: int,
        time_dim: str = "time",
        depth_dim: Optional[str] = "depth",
    ) -> None:
        self.catalog = catalog
        self.bbox_id = int(bbox_id)
        self.time_dim = time_dim
        self.depth_dim = depth_dim

    def _slice_2d(self, da_slice: xr.DataArray, ny: int, nx: int) -> np.ndarray:
        """
        Return a 2-D array shaped (ny, nx) for the catalog grid.

        """
        arr = da_slice.values
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        if arr.shape == (ny, nx):
            return arr
        if arr.shape == (nx, ny):
            return arr.T
        raise ValueError(
            f"Spatial slice shape {arr.shape} does not match catalog grid {(ny, nx)}."
        )

    def to_frame_single(
        self,
        ds: xr.Dataset,
        *,
        var_name: str,
        with_coords: bool = True,
    ) -> pd.DataFrame:
        """
        Return a tidy DataFrame for a single variable, optionally including tile coordinates.

        Depth semantics:
          - No depth dim → depth_idx = -1, depth_value = NaN
          - With depth dim → depth_idx as index, depth_value from the depth coordinate
        """
        da = ds[var_name]  # KeyError if missing
        time_vals = da[self.time_dim].values  # KeyError if missing

        # Sea tile metadata (aligned and stable)
        tile_ids = self.catalog.sea_tile_ids().astype(np.int64, copy=False)  # 0..K-1
        num_tiles = tile_ids.size
        if with_coords:
            tile_lon, tile_lat = self.catalog.sea_tile_coords()

        # Flat sea mask to extract values in row-major order
        tim = self.catalog.tile_id_map  # (ny, nx), -1 on land
        ny, nx = tim.shape
        sea_mask_flat = tim.ravel(order="C") >= 0

        has_depth = (self.depth_dim is not None) and (self.depth_dim in da.dims)
        frames: list[pd.DataFrame] = []

        # For each time slice get all variable values associated to a particular depth
        for t_idx, t_val in enumerate(time_vals):
            base = da.isel({self.time_dim: t_idx})
            depth_iter = range(len(base[self.depth_dim])) if has_depth else (0,)  #
            depth_vals = base[self.depth_dim].values if has_depth else None

            for d_idx in depth_iter:
                # Resolve slice + depth value/index
                slice_da = base.isel({self.depth_dim: d_idx}) if has_depth else base
                depth_idx = d_idx if has_depth else -1

                # Extract sea-only values in tile_id order
                slice2d = self._slice_2d(slice_da, ny, nx)
                sea_vec = slice2d.ravel(order="C")[sea_mask_flat]

                record = {
                    "time": np.repeat(t_val, num_tiles),
                    "depth_idx": np.repeat(depth_idx, num_tiles),
                    "tile_id": tile_ids,
                    var_name: sea_vec.astype(sea_vec.dtype, copy=False),
                }
                # Avoid adding float columns for potential multiple variable merges to avoid float comparison
                if with_coords:
                    depth_value = depth_vals[d_idx] if has_depth else np.nan
                    record["depth_value"] = np.repeat(depth_value, num_tiles)
                    record["tile_lon"] = tile_lon
                    record["tile_lat"] = tile_lat

                frames.append(pd.DataFrame(record))

        out = pd.concat(frames, axis=0, ignore_index=True)
        out["tile_id"] = out["tile_id"].astype(np.int64, copy=False)
        out["depth_idx"] = out["depth_idx"].astype(np.int64, copy=False)

        # Avoid bbox_id_{{x}} effect when merging multiple variables - added to one variable only
        if with_coords:
            out["bbox_id"] = np.int64(self.bbox_id)

        return out

    def to_frame_multi(self, ds: xr.Dataset, var_names: Sequence[str]) -> pd.DataFrame:
        """
        Return a tidy, wide DataFrame for multiple variables.
        """
        if not var_names:
            raise ValueError("var_names must be a non-empty sequence.")

        base_name = var_names[0]
        base = self.to_frame_single(ds, var_name=base_name, with_coords=True)

        if len(var_names) == 1:
            return base

        keys = ["time", "depth_idx", "tile_id"]
        for name in var_names[1:]:
            df = self.to_frame_single(ds, var_name=name, with_coords=False)
            base = pd.merge(base, df, on=keys, how="inner", validate="one_to_one")

        return base
