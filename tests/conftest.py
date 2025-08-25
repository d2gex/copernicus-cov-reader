from pathlib import Path

import pytest
import xarray as xr

from src.copernicus.grid_spec import GridSpec

DATA_DIR = Path(__file__).parent / "data"
DS_REF_FILE_1 = DATA_DIR / "sst_2020-01-01_2020-01-31.nc"
DS_REF_FILE_2 = DATA_DIR / "sst_2020-02-01_2020-02-29.nc"
DS_STATIC = DATA_DIR / "sst_static_layer.nc"
MASK_CANDIDATES = ("mask", "sea_binary_mask", "landsea_mask", "sea_mask")


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
def ds_static() -> xr.Dataset:
    if not DS_STATIC.exists():
        pytest.skip(f"Missing test data: {DS_STATIC}")
    with xr.open_dataset(DS_STATIC) as ds:
        return ds.load()


@pytest.fixture(scope="session")
def grid_spec(ds_ref: xr.Dataset) -> GridSpec:
    """GridSpec built once from the reference dataset."""
    return GridSpec.from_dataset(ds_ref)
