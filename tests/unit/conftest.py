import numpy as np
import pytest
import xarray as xr

from src.data_processing.grid_spec import GridSpec
from src.data_processing.sea_mask_builder import SeaMaskBuilder
from tests.unit import config as test_config


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


@pytest.fixture(scope="session")
def mask_da(ds_static: xr.Dataset) -> np.ndarray:
    sea_land_builder = SeaMaskBuilder(
        mask_name="mask",
        is_bit=False,
        sea_value=0,
    )
    return sea_land_builder.build(ds_static)
