import numpy as np

from src.copernicus.kd_index import KDIndex
from src.copernicus.tile_catalog import TileCatalog
from tests.unit.kd_index import helpers


def test_same_lon_closest_lat(grid_3x3, mask_all_sea):
    catalog = TileCatalog(grid_3x3, mask_all_sea)
    kdi = KDIndex(catalog)

    # Between (1,10) [idx=1] and (1,11) [idx=4]
    q_lon, q_lat = 1.0, 10.2
    sea_lon, sea_lat = catalog.sea_tile_coords()

    expected = helpers.get_index_of_closest_coordinate_pair(
        kdi.lat0, q_lon, q_lat, sea_lon, sea_lat
    )
    out = kdi.query_many(np.array([q_lon]), np.array([q_lat]))
    assert int(out[0]) == expected == 1


def test_same_lat_closest_lon(grid_3x3, mask_all_sea):
    catalog = TileCatalog(grid_3x3, mask_all_sea)
    kdi = KDIndex(catalog)

    # Between (0,11) [idx=3] and (1,11) [idx=4]; nearer to lon=1
    q_lon, q_lat = 0.6, 11.0
    sea_lon, sea_lat = catalog.sea_tile_coords()

    expected = helpers.get_index_of_closest_coordinate_pair(
        kdi.lat0, q_lon, q_lat, sea_lon, sea_lat
    )
    out = kdi.query_many(np.array([q_lon]), np.array([q_lat]))
    assert int(out[0]) == expected == 4


def test_diagonal_counterexample_east_wins_with_cos(
    grid_counterexample, mask_counterexample
):
    catalog = TileCatalog(grid_counterexample, mask_counterexample)
    kdi = KDIndex(catalog)

    # Raw degrees would favor north (0.45 < 0.5). With lon*cos(lat0), east should win.
    q_lon, q_lat = 0.5, 60.0
    sea_lon, sea_lat = catalog.sea_tile_coords()

    expected = helpers.get_index_of_closest_coordinate_pair(
        kdi.lat0, q_lon, q_lat, sea_lon, sea_lat
    )
    out = kdi.query_many(np.array([q_lon]), np.array([q_lat]))
    assert expected == 0
    assert int(out[0]) == 0
