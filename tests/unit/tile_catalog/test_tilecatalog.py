from unittest.mock import Mock

import numpy as np

from src.data_processing.tile_catalog import TileCatalog
from tests.unit.tile_catalog import helpers


def test_sea_tile_ids(mock_grid_fields, mock_sea_land_mask) -> None:
    grid = Mock(**mock_grid_fields)
    catalog = TileCatalog(grid=grid, sea_land_mask=mock_sea_land_mask)

    ids = catalog.sea_tile_ids()
    assert ids.ndim == 1
    print(ids)
    print(helpers.expected_ids())
    assert np.array_equal(ids, helpers.expected_ids())


def test_sea_cell_ids(mock_grid_fields, mock_sea_land_mask) -> None:
    grid = Mock(**mock_grid_fields)
    catalog = TileCatalog(grid=grid, sea_land_mask=mock_sea_land_mask)

    exp = helpers.expected_ij()
    for k, (ej, ei) in exp.items():
        j, i = catalog.sea_cell_ids(k)
        assert (j, i) == (ej, ei)
        # sanity: returned indices must be sea
        assert mock_sea_land_mask[j, i]


def test_sea_cell_coords(mock_grid_fields, mock_sea_land_mask) -> None:
    grid = Mock(**mock_grid_fields)
    catalog = TileCatalog(grid=grid, sea_land_mask=mock_sea_land_mask)

    exp_lon, exp_lat = helpers.expected_lonlat()
    for k in helpers.expected_ids():
        lon, lat = catalog.sea_cell_coords(int(k))
        assert lon == float(exp_lon[k])
        assert lat == float(exp_lat[k])


def test_sea_tile_coords(mock_grid_fields, mock_sea_land_mask) -> None:
    grid = Mock(**mock_grid_fields)
    catalog = TileCatalog(grid=grid, sea_land_mask=mock_sea_land_mask)

    lon_all, lat_all = catalog.sea_tile_coords()
    exp_lon, exp_lat = helpers.expected_lonlat()

    assert lon_all.shape == exp_lon.shape
    assert lat_all.shape == exp_lat.shape
    assert np.array_equal(lon_all, exp_lon)
    assert np.array_equal(lat_all, exp_lat)
