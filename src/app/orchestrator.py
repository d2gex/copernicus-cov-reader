from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

from pandas import DataFrame

from .jobs import DownloadJob


@dataclass(frozen=True)
class BBoxSpec:
    bbox_id: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def contains(self, lon: float, lat: float) -> bool:
        return (self.min_lon <= lon <= self.max_lon) and (
            self.min_lat <= lat <= self.max_lat
        )


class TileDayOrchestrator:
    def __init__(
        self, dataset_id: str, variables: Sequence[str], spatial_resolution_deg: float
    ) -> None:
        self.dataset_id = dataset_id
        self.variables = tuple(variables)
        self.spatial_resolution_deg = float(spatial_resolution_deg)

    def build_jobs(
        self, df: DataFrame, bboxes: Iterable[BBoxSpec]
    ) -> list[DownloadJob]:
        required = [
            "tile_id",
            "tile_lon_center",
            "tile_lat_center",
            "time",
            "deepest_depth",
        ]
        _ = df[required]

        epsilon = self.spatial_resolution_deg / 8.0
        bbox_list = list(bboxes)
        df_sorted = df.sort_values(["tile_id", "time"])

        out: list[DownloadJob] = []
        for row in df_sorted.itertuples(index=False):
            lon = float(getattr(row, "tile_lon_center"))
            lat = float(getattr(row, "tile_lat_center"))
            day_val = getattr(row, "time")
            if isinstance(day_val, str):
                day = datetime.fromisoformat(day_val).date()
            else:
                try:
                    day = day_val.date()
                except AttributeError:
                    day = day_val

            deepest = float(getattr(row, "deepest_depth"))
            z_min = 0.6
            z_max = deepest if deepest > 0.6 else 0.6

            area = (lon - epsilon, lat - epsilon, lon + epsilon, lat + epsilon)

            picked = None
            for b in bbox_list:
                if b.contains(lon, lat):
                    picked = b
                    break
            if picked is None:
                raise ValueError(f"Tile center not in any bbox: lon={lon}, lat={lat}")

            tile_id_val = getattr(row, "tile_id")
            tile_id_padded = str(int(tile_id_val)).zfill(5)

            out.append(
                DownloadJob(
                    bbox_id=picked.bbox_id,
                    tile_id_padded=tile_id_padded,
                    lon=lon,
                    lat=lat,
                    day=day,
                    z_min=z_min,
                    z_max=z_max,
                    area=area,
                    dataset_id=self.dataset_id,
                    variables=self.variables,
                    request_opts=None,
                )
            )
        return out
