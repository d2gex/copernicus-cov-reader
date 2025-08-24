import numpy as np

from src.copernicus.tile_catalog import TileCatalog
from tests.helpers import shallowest_layer


def test_build_mask_all_sea_when_none(grid_spec) -> None:
    out = TileCatalog.build_sea_land_mask(grid_spec, mask=None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(grid_spec.lats), len(grid_spec.lons))
    assert all([out.dtype == bool, out.all()])


def test_build_mask_matches_shallowest_layer(grid_spec, mask_da) -> None:
    # Reference via different route: shallowest-by-index, then align to (lat, lon)
    ref_da = shallowest_layer(mask_da).transpose(
        grid_spec.lat_name, grid_spec.lon_name, ...
    )
    ref = np.asarray(ref_da.values) == 1

    out = TileCatalog.build_sea_land_mask(grid_spec, mask=mask_da)

    assert out.dtype == bool
    assert out.shape == ref.shape
    assert np.array_equal(out, ref)
