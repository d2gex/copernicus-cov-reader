import pytest
import xarray as xr

from src.data_processing.dataset_tile_frame_extractor import DatasetTileFrameExtractor
from src.data_processing.tile_catalog import TileCatalog
from tests.unit import config as test_config


@pytest.fixture(scope="session")
def ds_mdlg_ref() -> xr.Dataset:
    if not test_config.MDLG_FILE_1.exists():
        pytest.skip(f"Missing test data: {test_config.MDLG_FILE_1}")
    with xr.open_dataset(test_config.MDLG_FILE_1) as ds:
        return ds.load()


@pytest.fixture(scope="session")
def ds_zos_no_depth_ref() -> xr.Dataset:
    if not test_config.ZNDG_FILE_1.exists():
        pytest.skip(f"Missing test data: {test_config.ZNDG_FILE_1}")
    with xr.open_dataset(test_config.ZNDG_FILE_1) as ds:
        return ds.load()


@pytest.fixture(scope="session")
def ds_mdlg_static() -> xr.Dataset:
    if not test_config.MDLG_STATIC.exists():
        pytest.skip(f"Missing test data: {test_config.MDLG_STATIC}")
    with xr.open_dataset(test_config.MDLG_STATIC) as ds:
        return ds.load()


@pytest.fixture(scope="module")
def zos_catalog(ds_zos_no_depth_ref: xr.Dataset, mask_da: xr.DataArray) -> TileCatalog:
    return TileCatalog.from_dataset(ds_zos_no_depth_ref, mask=mask_da)


@pytest.fixture(scope="module")
def zos_extractor(zos_catalog: TileCatalog) -> DatasetTileFrameExtractor:
    return DatasetTileFrameExtractor(catalog=zos_catalog)
