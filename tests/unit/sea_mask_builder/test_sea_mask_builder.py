from __future__ import annotations

import re

import numpy as np
import pytest

from src.data_processing.sea_mask_builder import SeaMaskBuilder

# ------------------ Bitwise tests ------------------


@pytest.mark.parametrize("use_real", [False, True])
def test_bit_wise_true(use_real, mock_bitwise_case, sst_embedded_mask_ds):
    """Bitwise mask: handles bit-AND and _FillValue/NaN as missing."""
    if use_real:
        ds = sst_embedded_mask_ds
        da = (
            ds["mask"]
            .isel(time=0)
            .transpose("latitude", "longitude")
            .squeeze(drop=True)
        )
        arr = np.asarray(da.values)
        # Missing: NaN/Inf + _FillValue (if present)
        miss = np.zeros(arr.shape, dtype=bool)
        if np.issubdtype(arr.dtype, np.floating):
            miss |= ~np.isfinite(arr)
        fv = da.attrs.get("_FillValue")
        if fv is not None:
            miss |= arr == fv
        safe = np.array(arr, copy=True)
        safe[miss] = 0
        expected = (safe.astype(np.uint16) & 1) != 0
        expected[miss] = False
    else:
        ds, expected = mock_bitwise_case

    builder = SeaMaskBuilder(mask_name="mask", is_bit=True, sea_value=1)
    got = builder.build(ds)

    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


# ------------------ Non-bitwise tests ------------------


@pytest.mark.parametrize("use_real", [False, True])
def test_bit_wise_false(use_real, mock_categorical_case, sst_static_mask_ds):
    """Categorical mask: equality only; no _FillValue path here (ints)."""
    if use_real:
        ds = sst_static_mask_ds
        da = (
            ds["mask"]
            .isel(depth=0)
            .transpose("latitude", "longitude")
            .squeeze(drop=True)
        )
        arr = np.asarray(da.values)
        long_name = str(da.attrs.get("long_name", ""))
        m = re.search(r"(\d+)\s*=\s*sea", long_name, flags=re.IGNORECASE)
        assert m, "Expected 'N = sea' in long_name for categorical mask"
        sea_val = int(m.group(1))
        expected = arr == sea_val
    else:
        ds, expected = mock_categorical_case

    builder = SeaMaskBuilder(mask_name="mask", is_bit=False, sea_value=None)
    got = builder.build(ds)

    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_bitmask_requires_sea_value(fx_error_bitmask_no_sea_value_ds):
    ds = fx_error_bitmask_no_sea_value_ds
    b = SeaMaskBuilder(mask_name="mask", is_bit=True, sea_value=None)
    with pytest.raises(KeyError):
        b.build(ds)


def test_nonbit_missing_long_name_with_no_sea_value(
    fx_error_categorical_missing_long_name_ds,
):
    ds = fx_error_categorical_missing_long_name_ds
    b = SeaMaskBuilder(mask_name="mask", is_bit=False, sea_value=None)
    with pytest.raises(KeyError):
        b.build(ds)


def test_requires_lat_lon_dims(fx_error_wrong_dims_ds):
    ds = fx_error_wrong_dims_ds
    b = SeaMaskBuilder(mask_name="mask", is_bit=False, sea_value=1)
    with pytest.raises(KeyError):
        b.build(ds)
