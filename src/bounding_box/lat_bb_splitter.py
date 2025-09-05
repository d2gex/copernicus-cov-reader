# bbox_splitter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.bounding_box.bounding_box import BoundingBox


@dataclass(frozen=True)
class BandResult:
    index: int
    bbox: BoundingBox
    df: pd.DataFrame  # only 'lon','lat' rows inside this band's sub-bbox
    lat0: float  # representative latitude for this band (mean or mid)


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

    def split(self, df: pd.DataFrame) -> List[BandResult]:
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

        # lat band edges
        edges = np.linspace(self._bbox.min_lat, self._bbox.max_lat, self._n_bands + 1)

        bbox_list: List[BandResult] = []
        for band in range(self._n_bands):
            lo = float(edges[band])
            hi = float(edges[band + 1])

            # include top edge only on the last band
            if band < self._n_bands - 1:
                band_mask = (df_in["lat"] >= lo) & (df_in["lat"] < hi)
            else:
                band_mask = (df_in["lat"] >= lo) & (df_in["lat"] <= hi)

            df_k = df_in.loc[band_mask, ["index", "lon", "lat"]].reset_index(drop=True)

            if len(df_k):
                lat0 = float(df_k["lat"].mean())
            else:
                lat0 = (lo + hi) * 0.5

            sub_bbox = BoundingBox(self._bbox.min_lon, self._bbox.max_lon, lo, hi)
            bbox_list.append(BandResult(index=band, bbox=sub_bbox, df=df_k, lat0=lat0))
        return bbox_list
