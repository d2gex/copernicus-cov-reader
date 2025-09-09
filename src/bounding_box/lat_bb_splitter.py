from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.bounding_box.bounding_box import BoundingBox


@dataclass(frozen=True)
class BandResult:
    band: int
    bbox: BoundingBox
    coords_within_band_df: (
        pd.DataFrame
    )  # only 'lon','lat' rows inside this band's sub-bbox
    band_lat0: float  # representative latitude for this band (mean or mid)
    num_points: int


class LatBandSplitter:
    """
    Split a BoundingBox into N latitude bands and assign points (lon, lat) to bands.
    Use band.lat0 as the local scaling latitude for lon*cos(lat0) in each band.
    """

    def __init__(self, bbox: BoundingBox, n_bands: int) -> None:
        self._bbox = bbox
        self._n_bands = int(n_bands)

    @property
    def bbox(self) -> BoundingBox:
        return self._bbox

    @property
    def n_bands(self) -> int:
        return self._n_bands

    def split(self, df: pd.DataFrame) -> Optional[List[BandResult]]:
        """
        Perform the split. Keeps only rows with columns 'lon','lat' inside the bbox.
        """
        # filter to overall bbox
        inside_bbox_band = (
            (df["lon"] >= self._bbox.min_lon)
            & (df["lon"] <= self._bbox.max_lon)
            & (df["lat"] >= self._bbox.min_lat)
            & (df["lat"] <= self._bbox.max_lat)
        )
        df_in = df.loc[inside_bbox_band, ["lon", "lat"]].reset_index(drop=False)
        if not len(df_in):
            return None

        # lat band edges
        edges = np.linspace(self._bbox.min_lat, self._bbox.max_lat, self._n_bands + 1)

        bbox_list: List[BandResult] = []
        for band in range(self._n_bands):
            lat_lo = float(edges[band])
            lat_hi = float(edges[band + 1])

            # include top edge only on the last band
            if band < self._n_bands - 1:
                band_mask = (df_in["lat"] >= lat_lo) & (df_in["lat"] < lat_hi)
            else:
                band_mask = (df_in["lat"] >= lat_lo) & (df_in["lat"] <= lat_hi)

            df_band = df_in.loc[band_mask, ["lon", "lat"]].reset_index(drop=True)

            num_points = len(df_band)
            if num_points == 0:
                lat0 = float(df_band["lat"].mean())
            else:
                lat0 = (
                    lat_lo + lat_hi
                ) * 0.5  # there may not be coordinates in such band.

            sub_bbox = BoundingBox(
                self._bbox.min_lon, self._bbox.max_lon, lat_lo, lat_hi
            )
            bbox_list.append(
                BandResult(
                    band=band,
                    bbox=sub_bbox,
                    coords_within_band_df=df_band,
                    band_lat0=lat0,
                    num_points=num_points,
                )
            )
        return bbox_list
