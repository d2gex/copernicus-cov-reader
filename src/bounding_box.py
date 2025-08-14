from typing import NamedTuple


class BoundingBox(NamedTuple):
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
