import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="session")
def mask_da(ds_static: xr.Dataset) -> xr.DataArray:
    for name in ("mask", "sea_binary_mask", "landsea_mask", "sea_mask"):
        if name in ds_static.data_vars:
            return ds_static[name]


@pytest.fixture(scope="session")
def mock_grid_fields():
    """
    3×3 rectilinear grid fields for tests.
    Integers to avoid float comparisons.
    """
    lats = np.array([10, 20, 30], dtype=np.int32)
    lons = np.array([100, 200, 300], dtype=np.int32)
    return {"lats": lats, "lons": lons, "ny": lats.size, "nx": lons.size}


@pytest.fixture(scope="session")
def mock_sea_land_mask() -> np.ndarray:
    """
    3×3 boolean mask with exactly 5 sea cells (True).
    Row-major (C-order) sea tile order ⇒ IDs 0..4 at:
      (0,0), (0,2), (1,1), (2,0), (2,2)
    """
    return np.array(
        [
            [True, False, True],
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
