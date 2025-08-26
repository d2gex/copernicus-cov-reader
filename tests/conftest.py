import pytest
import xarray as xr

from src.copernicus.grid_spec import GridSpec
from tests import config as test_config

MASK_CANDIDATES = ("mask", "sea_binary_mask", "landsea_mask", "sea_mask")


@pytest.fixture(scope="session")
def ds_sst_slg_ref() -> xr.Dataset:
    if not test_config.SST_SLG_FILE_1.exists():
        pytest.skip(f"Missing test data: {test_config.SST_SLG_FILE_1}")
    with xr.open_dataset(test_config.SST_SLG_FILE_1) as ds:
        return ds.load()


@pytest.fixture(scope="session")
def ds_sst_slg_ref_2() -> xr.Dataset:
    if not test_config.SST_SLG_FILE_2.exists():
        pytest.skip(f"Missing test data: {test_config.SST_SLG_FILE_2}")
    with xr.open_dataset(test_config.SST_SLG_FILE_2) as ds:
        return ds.load()


@pytest.fixture(scope="session")
def ds_static() -> xr.Dataset:
    if not test_config.SST_SLG_STATIC.exists():
        pytest.skip(f"Missing test data: {test_config.SST_SLG_STATIC}")
    with xr.open_dataset(test_config.SST_SLG_STATIC) as ds:
        return ds.load()


@pytest.fixture(scope="session")
def grid_spec(ds_sst_slg_ref: xr.Dataset) -> GridSpec:
    """GridSpec built once from the reference dataset."""
    return GridSpec.from_dataset(ds_sst_slg_ref)
