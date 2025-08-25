import pytest
import xarray as xr

from src.copernicus.grid_spec import GridSpec


@pytest.fixture(scope="function")
def ds_same(ds_ref: xr.Dataset) -> xr.Dataset:
    return ds_ref.copy(deep=False)


@pytest.fixture(scope="function")
def ds_other_same_grid(ds_ref_2: xr.Dataset) -> xr.Dataset:
    return ds_ref_2.copy(deep=False)


@pytest.fixture(scope="function")
def ds_modified(ds_ref: xr.Dataset, grid_spec: GridSpec) -> xr.Dataset:
    lon_name = grid_spec.lon_name
    if ds_ref[lon_name].ndim != 1:
        pytest.skip("Test expects 1-D longitude for rectilinear grid.")

    ds_bad = ds_ref.copy(deep=False)  # light copy; data shared, coords replaceable
    lons = ds_bad[lon_name].values.copy()
    lons[0] = lons[0] + 3  # Change in the longitude arrays
    ds_bad = ds_bad.assign_coords({lon_name: (ds_bad[lon_name].dims, lons)})
    return ds_bad
