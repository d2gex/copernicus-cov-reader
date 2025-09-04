from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# (1) SST with one single depth level test data
SST_SINGLE_LEVEL_GALICIA = DATA_DIR / "sst_single_level_galicia"
SST_SLG_FILE_1 = SST_SINGLE_LEVEL_GALICIA / "sst_2020-01-01_2020-01-31.nc"
SST_SLG_FILE_2 = SST_SINGLE_LEVEL_GALICIA / "sst_2020-02-01_2020-02-29.nc"
SST_SLG_STATIC = SST_SINGLE_LEVEL_GALICIA / "sst_static_layer.nc"

# (2) Multiple depth layer test data
MULTI_DEPTH_LAYER_GALICIA = DATA_DIR / "multi_depth_layer_galicia"
MDLG_FILE_1 = (
    MULTI_DEPTH_LAYER_GALICIA / "multiple_depth_layer_2020-01-01_2020-01-31.nc"
)
MDLG_STATIC = MULTI_DEPTH_LAYER_GALICIA / "multiple_depth_static_layer.nc"

# (3) No depth ZOS test data
ZOS_NO_DEPTH_LAYER_GALICIA = DATA_DIR / "no_depth_zos"
ZNDG_FILE_1 = ZOS_NO_DEPTH_LAYER_GALICIA / "no_depth_zos_2020-01-01_2020-01-31.nc"
ZNDG_STATIC = ZOS_NO_DEPTH_LAYER_GALICIA / "no_depth_zos_static.nc"
