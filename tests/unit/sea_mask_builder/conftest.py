from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pytest
import xarray as xr

from tests.unit import config as test_config


@pytest.fixture(scope="session")
def sst_embedded_mask_path() -> Path:
    p = test_config.EMBEDDED_SEA_LAND_SAMPLE
    if not p.exists():
        pytest.skip(f"Missing test fixture: {p}")
    return p


@pytest.fixture(scope="session")
def sst_static_mask_path() -> Path:
    p = test_config.STATIC_FILE_SEA_LAND_SAMPLE
    if not p.exists():
        pytest.skip(f"Missing test fixture: {p}")
    return p


@pytest.fixture(scope="session")
def sst_embedded_mask_ds(sst_embedded_mask_path: Path) -> Iterator[xr.Dataset]:
    with xr.open_dataset(sst_embedded_mask_path, decode_times=True) as ds:
        yield ds


@pytest.fixture(scope="session")
def sst_static_mask_ds(sst_static_mask_path: Path) -> Iterator[xr.Dataset]:
    with xr.open_dataset(sst_static_mask_path, decode_times=True) as ds:
        yield ds


@pytest.fixture(scope="session")
def sea_bool_mask() -> np.ndarray:
    """A tiny 2x5 grid with 7 sea cells and 3 non-sea."""
    sea_bool = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 0, 1]], dtype=bool)
    return sea_bool


@pytest.fixture(scope="session")
def mock_bitwise_case(sea_bool_mask) -> Tuple[xr.Dataset, np.ndarray]:
    """A tiny 2x5 grid with 7 sea and 3 non-sea; bitfield semantics.

    Variable: "mask"
    Dims: (time=1, latitude=2, longitude=5)
    Values:
      - Sea => bit 1 set (value 1)
      - Non-sea => use other bits (2,4) and one explicit _FillValue to exercise missing
    """

    arr = np.where(sea_bool_mask, 1.0, 0.0).astype(np.float32)
    # Non-sea cells: set distinct patterns; make one of them FillValue (-128)
    arr[0, 4] = 2.0  # land bit
    arr[1, 2] = 4.0  # lake bit
    arr[1, 3] = -128.0  # _FillValue (treated as missing -> not sea)

    da = xr.DataArray(
        arr[np.newaxis, ...],  # add time dim
        dims=("time", "latitude", "longitude"),
        attrs={"long_name": "land sea ice lake bit mask", "_FillValue": -128.0},
    )
    ds = xr.Dataset({"mask": da})

    expected = sea_bool_mask.copy()
    # Missing cell should be False (already False in expected)
    return ds, expected


@pytest.fixture(scope="session")
def mock_categorical_case(sea_bool_mask) -> Tuple[xr.Dataset, np.ndarray]:
    """A tiny 2x5 grid with 7 sea and 3 non-sea; categorical semantics.

    Variable: "mask"
    Dims: (depth=2, latitude=2, longitude=5) -> we will use depth=0
    Values: 1 = sea, 0 = land (declared in long_name). No _FillValue here.
    """

    depth0 = np.where(sea_bool_mask, 1, 0).astype(np.int8)
    depth1 = np.zeros_like(depth0)

    da = xr.DataArray(
        np.stack([depth0, depth1], axis=0),
        dims=("depth", "latitude", "longitude"),
        attrs={"long_name": "Land-sea mask: 1 = sea ; 0 = land"},  # no _FillValue
    )
    ds = xr.Dataset({"mask": da})

    expected = sea_bool_mask.copy()
    return ds, expected
