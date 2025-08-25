import math

import numpy as np


def get_index_of_closest_coordinate_pair(
    lat0: float, q_lon: float, q_lat: float, sea_lon: np.ndarray, sea_lat: np.ndarray
) -> int:
    c = math.cos(math.radians(lat0))
    qx, qy = q_lon * c, q_lat
    x = sea_lon * c
    y = sea_lat
    d2 = (x - qx) ** 2 + (y - qy) ** 2
    return int(np.argmin(d2))
