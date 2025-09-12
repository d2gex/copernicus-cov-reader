# src/application/sea_mask_builder.py
from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import xarray as xr


class SeaMaskBuilder:
    """
    Build a 2-D boolean sea mask from an in-memory static dataset (xarray.Dataset).
    """

    def __init__(
        self, *, mask_name: str, is_bit: bool, sea_value: Optional[int] = None
    ) -> None:
        self.mask_name = mask_name
        self.is_bit = bool(is_bit)
        self.sea_value = sea_value

    # ---- Public API ----

    def build(self, ds: xr.Dataset) -> np.ndarray:
        da = self._select_mask_var(ds, self.mask_name)
        da2d = self._reduce_to_2d(da)
        da2d = da2d.transpose("latitude", "longitude")

        arr, is_missing = self._to_array_and_missing(da2d)
        if arr.ndim != 2:
            raise ValueError("Mask must be 2-D after reduction and transpose.")

        if self.is_bit:
            if self.sea_value is None:
                raise KeyError("sea_value must be provided when is_bit=True.")
            sea = self._bit_to_bool(arr, is_missing, int(self.sea_value))
        else:
            sea_val = self.sea_value
            if sea_val is None:
                sea_val = self._infer_sea_value_from_long_name(
                    da2d
                )  # may raise KeyError
            sea = self._categorical_to_bool(arr, is_missing, int(sea_val))

        return sea

    # ---- Helpers ----

    @staticmethod
    def _select_mask_var(ds: xr.Dataset, name: str) -> xr.DataArray:
        if name not in ds:
            raise KeyError(f"Mask variable '{name}' not found in dataset.")
        if not isinstance(ds[name], xr.DataArray):
            raise TypeError(f"Dataset variable '{name}' is not a DataArray.")
        return ds[name]

    @staticmethod
    def _reduce_to_2d(da: xr.DataArray) -> xr.DataArray:
        if "time" in da.dims:
            da = da.isel(time=0)
        if "depth" in da.dims:
            da = da.isel(depth=0)
        da = da.squeeze(drop=True)
        return da

    @staticmethod
    def _to_array_and_missing(da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(da.values)
        is_missing = np.zeros(arr.shape, dtype=bool)
        if np.issubdtype(arr.dtype, np.floating):
            is_missing |= ~np.isfinite(arr)
        fill_attr = da.attrs.get("_FillValue")
        is_missing |= arr == fill_attr
        return arr, is_missing

    @staticmethod
    def _bit_to_bool(
        arr: np.ndarray, is_missing: np.ndarray, sea_bit: int
    ) -> np.ndarray:
        if sea_bit <= 0:
            raise ValueError("sea_bit must be a positive integer mask (e.g., 1).")
        safe = np.array(arr, copy=True)
        safe[is_missing] = 0
        # Be tolerant of float inputs (e.g., 1.0) and signed sentinels
        uint = safe.astype(np.uint16, copy=False)
        sea = (uint & sea_bit) != 0
        return sea

    @staticmethod
    def _categorical_to_bool(
        arr: np.ndarray, is_missing: np.ndarray, sea_value: int
    ) -> np.ndarray:
        sea = arr == sea_value
        sea[is_missing] = False
        return sea

    @staticmethod
    def _infer_sea_value_from_long_name(da: xr.DataArray) -> int:
        # Only long_name is consulted; if missing or unparsable -> KeyError
        long_name = da.attrs.get("long_name")
        if not long_name:
            raise KeyError("Cannot infer sea_value: 'long_name' attribute is missing.")
        m = re.search(r"(\d+)\s*=\s*sea", str(long_name), flags=re.IGNORECASE)
        if not m:
            raise KeyError(
                "Cannot infer sea_value: expected pattern like '1 = sea' in 'long_name'."
            )
        return int(m.group(1))
