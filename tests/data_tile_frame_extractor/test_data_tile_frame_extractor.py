import numpy as np
import pandas as pd
import xarray as xr

from src.copernicus.dataset_tile_frame_extractor import DatasetTileFrameExtractor
from src.copernicus.tile_catalog import TileCatalog


def test_single_and_multi_identical_for_zos(
    ds_zos_no_depth_ref: xr.Dataset,
    zos_extractor: DatasetTileFrameExtractor,
):
    single = zos_extractor.to_frame_single(
        ds_zos_no_depth_ref, var_name="zos", with_coords=True
    )
    multi = zos_extractor.to_frame_multi(ds_zos_no_depth_ref, ["zos"])

    # Normalize order before equality checks
    sort_keys = ["time", "depth_idx", "tile_id"]
    single = single.sort_values(sort_keys).reset_index(drop=True)
    multi = multi.sort_values(sort_keys).reset_index(drop=True)

    # Same columns and rows
    assert list(single.columns) == list(multi.columns)
    pd.testing.assert_frame_equal(single, multi, check_like=False, check_dtype=False)


def test_zos_shape_and_columns(
    ds_zos_no_depth_ref: xr.Dataset,
    zos_catalog: TileCatalog,
    zos_extractor: DatasetTileFrameExtractor,
):
    frame = zos_extractor.to_frame_single(
        ds_zos_no_depth_ref, var_name="zos", with_coords=True
    )

    # a) shape: len = time_count * sea_tile_count; depth_idx must be {0}
    num_time_slices = ds_zos_no_depth_ref["time"].sizes["time"]
    num_tiles = int(zos_catalog.sea_tile_ids().size)
    depth_values = set(frame["depth_value"].unique().tolist())
    assert len(frame) == num_time_slices * num_tiles
    assert set(frame["depth_idx"].unique().tolist()) == {-1}
    assert len(depth_values) == 1 and np.isnan(depth_values.pop())

    # b) required columns present
    expected_cols = {
        "time",
        "depth_idx",
        "depth_value",
        "tile_id",
        "tile_lon",
        "tile_lat",
        "zos",
    }
    assert set(frame.columns) == expected_cols


def test_zos_tile_order_and_coords_match_catalog(
    ds_zos_no_depth_ref: xr.Dataset,
    zos_catalog: TileCatalog,
    zos_extractor: DatasetTileFrameExtractor,
):
    frame = zos_extractor.to_frame_single(
        ds_zos_no_depth_ref, var_name="zos", with_coords=True
    )

    # take first timestamp slice
    t0 = frame["time"].iloc[0]
    sub = frame[frame["time"] == t0]

    # c1) tile_id order equals catalog.sea_tile_ids()
    expected_ids = zos_catalog.sea_tile_ids()
    got_ids = sub["tile_id"].to_numpy()
    assert np.array_equal(got_ids, expected_ids)

    # c2) coordinates equal catalog.sea_tile_coords()
    exp_lon, exp_lat = zos_catalog.sea_tile_coords()
    got_lon = sub["tile_lon"].to_numpy()
    got_lat = sub["tile_lat"].to_numpy()

    # These are copied from the catalog, so exact equality should hold.
    # If you prefer a belt-and-braces tolerance, replace with np.allclose(..., atol=1e-12).
    assert np.array_equal(got_lon, exp_lon)
    assert np.array_equal(got_lat, exp_lat)
