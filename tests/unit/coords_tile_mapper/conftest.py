from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.copernicus.coords_tile_mapper import CoordinatesToTileMapper
from src.copernicus.kd_index import KDIndex
from src.copernicus.tile_catalog import TileCatalog


@pytest.fixture
def grid_4x4():
    lons = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    lats = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)
    g = Mock()
    g.lons, g.lats = lons, lats
    g.nx, g.ny = lons.size, lats.size
    return g


@pytest.fixture
def mask_all_sea_4x4():
    return np.ones((4, 4), dtype=bool)


@pytest.fixture
def df_five_points():
    data = [
        ("top_left", 0.1, 12.9, 12, 0.0, 13.0),
        ("top_right", 2.9, 12.9, 15, 3.0, 13.0),
        ("bottom_left", 0.1, 10.1, 0, 0.0, 10.0),
        ("bottom_right", 2.9, 10.1, 3, 3.0, 10.0),
        ("center", 1.6, 11.7, 10, 2.0, 12.0),
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "case",
            "lon",
            "lat",
            "expected_tile_id",
            "expected_lon",
            "expected_lat",
        ],
    ).set_index("case")
    return df


@pytest.fixture
def mapped_4x4(grid_4x4, mask_all_sea_4x4, df_five_points):
    catalog = TileCatalog(grid_4x4, mask_all_sea_4x4)
    kdi = KDIndex(catalog)
    mapper = CoordinatesToTileMapper(kdi)
    out = mapper.map(df_five_points, lon_col="lon", lat_col="lat", out_col="tile_id")
    sea_lon, sea_lat = catalog.sea_tile_coords()
    return out, sea_lon, sea_lat
