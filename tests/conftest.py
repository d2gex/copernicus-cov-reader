from pathlib import Path

import pytest
import xarray as xr

from src.copernicus.grid_spec import GridSpec

DATA_DIR = Path(__file__).parent / "data"

# SST with one single depth level test data
SST_SINGLE_LEVEL_GALICIA = DATA_DIR / "sst_single_level_galicia"
SST_SLG_FILE_1 = SST_SINGLE_LEVEL_GALICIA / "sst_2020-01-01_2020-01-31.nc"
SST_SLG_FILE_2 = SST_SINGLE_LEVEL_GALICIA / "sst_2020-02-01_2020-02-29.nc"
SST_SLG_STATIC = SST_SINGLE_LEVEL_GALICIA / "sst_static_layer.nc"
MASK_CANDIDATES = ("mask", "sea_binary_mask", "landsea_mask", "sea_mask")


@pytest.fixture(scope="session")
def ds_sst_slg_ref() -> xr.Dataset:
    if not SST_SLG_FILE_1.exists():
        pytest.skip(f"Missing test data: {SST_SLG_FILE_1}")
    with xr.open_dataset(SST_SLG_FILE_1) as ds:
        return ds.load()


@pytest.fixture(scope="session")
def ds_sst_slg_ref_2() -> xr.Dataset:
    if not SST_SLG_FILE_2.exists():
        pytest.skip(f"Missing test data: {SST_SLG_FILE_2}")
    with xr.open_dataset(SST_SLG_FILE_2) as ds:
        return ds.load()


@pytest.fixture(scope="session")
def ds_static() -> xr.Dataset:
    if not SST_SLG_STATIC.exists():
        pytest.skip(f"Missing test data: {SST_SLG_STATIC}")
    with xr.open_dataset(SST_SLG_STATIC) as ds:
        return ds.load()


@pytest.fixture(scope="session")
def grid_spec(ds_sst_slg_ref: xr.Dataset) -> GridSpec:
    """GridSpec built once from the reference dataset."""
    return GridSpec.from_dataset(ds_sst_slg_ref)
