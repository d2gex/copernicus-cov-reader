from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from src.data_processing.coords_tile_mapper import CoordinatesToTileMapper
from src.data_processing.grid_spec import GridSpec
from src.data_processing.kd_index import KDIndex
from src.data_processing.sea_mask_builder import SeaMaskBuilder
from src.data_processing.tile_catalog import TileCatalog


@dataclass(frozen=True)
class StaticSpec:
    """Static (mask) dataset specification."""

    path: Path
    mask_var: str = "mask"
    is_bit: bool = True
    sea_value: int = 1  # bit value when is_bit=True, or integer class when is_bit=False


class HaulTileAssigner:
    """
    Assigns each haul to the nearest *sea* model cell (tile_id) and appends
    the sea-cell center coordinates. Uses your KD-index over sea-only centers.
    """

    def __init__(
        self,
        static_spec: StaticSpec,
        lon_name: str = "longitude",
        lat_name: str = "latitude",
        lon_col: str = "lon",
        lat_col: str = "lat",
        time_col: str = "time",
        out_tile_col: str = "tile_id",
        out_lon_center_col: str = "tile_lon_center",
        out_lat_center_col: str = "tile_lat_center",
    ) -> None:
        self.static_spec = static_spec
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.time_col = time_col
        self.out_tile_col = out_tile_col
        self.out_lon_center_col = out_lon_center_col
        self.out_lat_center_col = out_lat_center_col

        # Deferred/DI-injected components:
        self._grid: Optional[GridSpec] = None
        self._catalog: Optional[TileCatalog] = None
        self._kd: Optional[KDIndex] = None

    # -------------------- lifecycle --------------------

    def load_static_and_build_index(self, lat0_hint: Optional[float] = None) -> None:
        """Open static dataset, build sea mask, tile catalog, and KD index (sea-only)."""
        spec = self.static_spec
        path = spec.path
        if not path.exists():
            raise FileNotFoundError(f"Static dataset not found: {path}")

        # Open static dataset
        with xr.open_dataset(path) as ds:
            # Build grid spec (strict hash from lon/lat 1-D arrays)
            grid = GridSpec.from_dataset(
                ds,
                lon_candidates=(self.lon_name, "lon", "x"),
                lat_candidates=(self.lat_name, "lat", "y"),
            )

            # Build boolean sea mask: True = sea, False = land
            mask_builder = SeaMaskBuilder(
                mask_name=spec.mask_var,
                is_bit=spec.is_bit,
                sea_value=spec.sea_value,
            )
            sea = mask_builder.build(ds)  # 2-D bool, shape ny×nx in (lat, lon) order

        # Create tile catalog (stable sea tile IDs and their centers)
        catalog = TileCatalog(grid=grid, sea_land_mask=sea)

        # KD over *sea* centers; use provided lat0 or the median grid latitude
        if lat0_hint is None:
            lat0_hint = float(np.nanmedian(grid.lats))
        kd = KDIndex(catalog=catalog, lat0=lat0_hint)

        # Store components
        self._grid = grid
        self._catalog = catalog
        self._kd = kd

    # -------------------- main operation --------------------

    def assign(
        self, hauls: pd.DataFrame, tolerance_deg: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Return a copy of hauls with three new columns:
          - tile_id
          - tile_lon_center
          - tile_lat_center
        """
        self._ensure_ready()

        # Normalize input columns are present
        for col in (self.lon_col, self.lat_col):
            if col not in hauls.columns:
                raise KeyError(f"Required column '{col}' not found in hauls dataframe.")

        # 1) tile_id via KD-index over sea-only cell centers
        mapper = CoordinatesToTileMapper(self._kd)  # type: ignore[arg-type]
        assigned = mapper.map(
            coordinates=hauls,
            lon_col=self.lon_col,
            lat_col=self.lat_col,
            tolerance_deg=tolerance_deg,
            out_col=self.out_tile_col,
        )

        # 2) append sea-cell center coords from catalog for the mapped tile_ids
        tile_ids = assigned[self.out_tile_col].to_numpy(copy=False)
        if np.any(tile_ids < 0):
            # Keep behavior explicit: mark unmapped with NaN centers; caller can filter/log.
            pass

        # Vectorized lookup using sea arrays; tile_id is an index into sea arrays
        sea_lons, sea_lats = self._catalog.sea_tile_coords()  # type: ignore[union-attr]
        centers_lon = np.full(tile_ids.shape, np.nan, dtype=float)
        centers_lat = np.full(tile_ids.shape, np.nan, dtype=float)

        ok = tile_ids >= 0
        if np.any(ok):
            centers_lon[ok] = sea_lons[tile_ids[ok]]
            centers_lat[ok] = sea_lats[tile_ids[ok]]

        out = assigned.copy()
        out[self.out_lon_center_col] = centers_lon
        out[self.out_lat_center_col] = centers_lat
        return out

    # -------------------- helpers --------------------

    def _ensure_ready(self) -> None:
        if (self._grid is None) or (self._catalog is None) or (self._kd is None):
            raise RuntimeError("Call load_static_and_build_index() before assign().")


# -------------------- thin entrypoint --------------------


def run(
    hauls_csv_in: Path,
    static_nc_path: Path,
    hauls_csv_out: Path,
    mask_var: str = "mask",
    is_bit: bool = True,
    sea_value: int = 1,
    lon_col: str = "lon",
    lat_col: str = "lat",
    time_col: str = "time",
    tolerance_deg: Optional[float] = None,
) -> Path:
    """
    Load hauls CSV, assign nearest *sea* tile_id via KD over static mask centers,
    append tile center coords, and write a new CSV.
    """
    hauls = pd.read_csv(hauls_csv_in)
    if hauls.empty:
        raise ValueError("Hauls CSV is empty.")

    # Lat0 hint = median haul latitude (keeps KD projection distances well-scaled)
    lat0_hint = float(np.nanmedian(hauls[lat_col].to_numpy()))

    assigner = HaulTileAssigner(
        static_spec=StaticSpec(
            path=static_nc_path, mask_var=mask_var, is_bit=is_bit, sea_value=sea_value
        ),
        lon_col=lon_col,
        lat_col=lat_col,
        time_col=time_col,
    )
    assigner.load_static_and_build_index(lat0_hint=lat0_hint)
    enriched = assigner.assign(hauls, tolerance_deg=tolerance_deg)

    # Write result
    hauls_csv_out = Path(hauls_csv_out)
    hauls_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with hauls_csv_out.open("w", newline="") as f:
        enriched.to_csv(f, index=False)
    return hauls_csv_out
