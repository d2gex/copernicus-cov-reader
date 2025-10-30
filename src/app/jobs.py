from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DownloadJob:
    bbox_id: str
    tile_id_padded: str
    lon: float
    lat: float
    day: date
    z_min: float
    z_max: float
    area: tuple[float, float, float, float]
    dataset_id: str
    variables: Sequence[str]
    request_opts: Mapping[str, object] | None = None
