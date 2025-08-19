from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from src.copernicus.grid_spec import GridSpec

DATA_DIR = Path(__file__).parent / "data"
DS_REF_FILE_1 = DATA_DIR / "sst_2020-01-01_2020-01-31.nc"
DS_REF_FILE_2 = DATA_DIR / "sst_2020-02-01_2020-02-29.nc"


@pytest.fixture(scope="session")
def ds_ref() -> xr.Dataset:
    if not DS_REF_FILE_1.exists():
        pytest.skip(f"Missing test data: {DS_REF_FILE_1}")
    with xr.open_dataset(DS_REF_FILE_1) as ds:
        return ds.load()


@pytest.fixture(scope="session")
def ds_ref_2() -> xr.Dataset:
    if not DS_REF_FILE_2.exists():
        pytest.skip(f"Missing test data: {DS_REF_FILE_2}")
    with xr.open_dataset(DS_REF_FILE_2) as ds:
        return ds.load()


@pytest.fixture(scope="session")
def grid_spec(ds_ref: xr.Dataset) -> GridSpec:
    """GridSpec built once from the reference dataset."""
    return GridSpec.from_dataset(ds_ref)


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
