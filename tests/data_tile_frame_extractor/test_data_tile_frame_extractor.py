import numpy as np
import pandas as pd
import pytest
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

    assert np.array_equal(got_lon, exp_lon)
    assert np.array_equal(got_lat, exp_lat)

    # c3) depth_idx is always -1 and NaN
    depth_values = set(frame["depth_value"].unique().tolist())
    assert set(frame["depth_idx"]) == {-1}
    assert len(depth_values) == 1 and np.isnan(depth_values.pop())


@pytest.mark.parametrize("t_idx", [0, 1])
@pytest.mark.parametrize("d_idx", list(range(7)))
def test_thetao_multi_depth_tile_ids_coords_depth_and_values(
    ds_mdlg_ref: xr.Dataset,
    ds_mdlg_static: xr.Dataset,
    t_idx: int,
    d_idx: int,
):
    # Build catalog with the known mask variable from the static set
    mask_da = ds_mdlg_static["mask_thetao"]
    catalog = TileCatalog.from_dataset(ds_mdlg_ref, mask=mask_da)
    ext = DatasetTileFrameExtractor(catalog=catalog)

    # Guard against unexpectedly short time/depth in the fixture
    if t_idx >= ds_mdlg_ref.sizes["time"]:
        pytest.skip(f"time index {t_idx} out of range for this fixture")
    if "depth" not in ds_mdlg_ref.sizes or d_idx >= ds_mdlg_ref.sizes["depth"]:
        pytest.skip(f"depth index {d_idx} out of range for this fixture")

    # Run extractor once; slice the resulting frame at (t_idx, d_idx)
    frame = ext.to_frame_single(ds_mdlg_ref, var_name="thetao", with_coords=True)
    t_val = ds_mdlg_ref["time"].values[t_idx]
    sub = frame[(frame["time"] == t_val) & (frame["depth_idx"] == d_idx)].reset_index(
        drop=True
    )

    # Expectations from catalog
    expected_ids = catalog.sea_tile_ids()
    exp_lon, exp_lat = catalog.sea_tile_coords()
    sea_mask_flat = catalog.tile_id_map.ravel(order="C") >= 0
    ny, nx = catalog.tile_id_map.shape

    # Expectations from the dataset slice
    slice2d = ds_mdlg_ref["thetao"].isel(time=t_idx, depth=d_idx).values
    if slice2d.ndim != 2:
        slice2d = np.squeeze(slice2d)
    if slice2d.shape == (nx, ny):
        slice2d = slice2d.T
    assert slice2d.shape == (
        ny,
        nx,
    ), f"Unexpected spatial shape: {slice2d.shape} vs {(ny, nx)}"
    expected_vals = slice2d.ravel(order="C")[sea_mask_flat]
    expected_depth_val = ds_mdlg_ref["depth"].values[d_idx]

    # 1) tile_id order matches catalog
    got_ids = sub["tile_id"].to_numpy()
    assert np.array_equal(got_ids, expected_ids)

    # 2) coordinates match catalog (copied from catalog)
    assert np.array_equal(sub["tile_lon"].to_numpy(), exp_lon)
    assert np.array_equal(sub["tile_lat"].to_numpy(), exp_lat)

    # 3) depth_value equals the dataset depth at d_idx (all rows)
    got_depth_vals = sub["depth_value"].to_numpy()
    assert np.allclose(got_depth_vals, expected_depth_val, rtol=0, atol=0)

    # 4) variable values match the sea-vector from the dataset slice
    got_vals = sub["thetao"].to_numpy()
    assert np.allclose(got_vals, expected_vals, equal_nan=True)
