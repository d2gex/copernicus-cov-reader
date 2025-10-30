from __future__ import annotations

from typing import List

import pandas as pd

from src.bounding_box.bounding_box import BoundingBox
from src.bounding_box.lat_bb_splitter import LatBandSplitter

from .orchestrator import BBoxSpec


class BBoxFactory:
    def __init__(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        lat_band_count: int,
    ) -> None:
        self.min_lon = float(min_lon)
        self.min_lat = float(min_lat)
        self.max_lon = float(max_lon)
        self.max_lat = float(max_lat)
        self.lat_band_count = int(lat_band_count)

    def build(self, tiles_df: pd.DataFrame) -> List[BBoxSpec]:
        """
        Build latitude-band bboxes using the existing splitter.
        The splitter requires a DataFrame with columns 'lon' and 'lat'.
        We feed it the tile centers from the CSV.
        """
        region = BoundingBox(
            min_lon=self.min_lon,
            max_lon=self.max_lon,
            min_lat=self.min_lat,
            max_lat=self.max_lat,
        )
        splitter = LatBandSplitter(bbox=region, n_bands=self.lat_band_count)

        coords_df = tiles_df[["tile_lon_center", "tile_lat_center"]].rename(
            columns={"tile_lon_center": "lon", "tile_lat_center": "lat"}
        )

        bands = splitter.split(coords_df) or []

        out: List[BBoxSpec] = []
        for br in bands:
            bb = br.bbox
            out.append(
                BBoxSpec(
                    bbox_id=f"bbox_{int(br.band):02d}",
                    min_lon=bb.min_lon,
                    min_lat=bb.min_lat,
                    max_lon=bb.max_lon,
                    max_lat=bb.max_lat,
                )
            )
        return out
