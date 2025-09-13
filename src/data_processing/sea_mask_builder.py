from __future__ import annotations

import re
from typing import Optional

import numpy as np
import xarray as xr


class SeaMaskBuilder:
    """
    Build a 2-D boolean sea mask from an in-memory static dataset (xarray.Dataset).

    Assumptions & rules (fail fast):
      - Spatial dims are ALWAYS named ("latitude", "longitude"). If not present → KeyError.
      - If a "time" dimension exists, we select t=0.
      - If a "depth" dimension exists, we select depth=0 (no other depth aliases).
      - For bitwise masks (is_bit=True):
          * sea_value is REQUIRED and interpreted as a bit mask (e.g., 1).
          * Missing cells are handled (NaN/Inf and variable-level _FillValue) and
            neutralized before applying bitwise operations.
      - For non-bitwise masks (is_bit=False):
          * If sea_value is None, parse ONLY the variable's long_name for a pattern
            like "1 = sea". If missing/unparseable → KeyError.
          * No missing handling is applied; equality test is used directly.

    Output: np.ndarray[bool] with shape (latitude, longitude), True for sea.
    """

    def __init__(
        self, *, mask_name: str, is_bit: bool, sea_value: Optional[int] = None
    ) -> None:
        self.mask_name = mask_name
        self.is_bit = bool(is_bit)
        self.sea_value = sea_value

    # ---------------- Public API ----------------

    def build(self, ds: xr.Dataset) -> np.ndarray:
        da = self._select_mask_var(ds, self.mask_name)
        da2d = self._reduce_to_2d(da)
        da2d = self._transpose_to_lat_lon(da2d)

        arr = np.asarray(da2d.values)
        if arr.ndim != 2:
            raise ValueError("Mask must be 2-D after reduction and transpose.")

        if self.is_bit:
            if self.sea_value is None:
                raise KeyError("sea_value must be provided when is_bit=True.")
            sea = self._bitmask_classify(da2d, int(self.sea_value))
        else:
            sea_val = self.sea_value
            if sea_val is None:
                sea_val = self._infer_sea_value_from_long_name(
                    da2d
                )  # may raise KeyError
            sea = arr == int(sea_val)
        return sea

    # ---------------- Helpers ----------------

    @staticmethod
    def _select_mask_var(ds: xr.Dataset, name: str) -> xr.DataArray:
        if name not in ds:
            raise KeyError(f"Mask variable '{name}' not found in dataset.")
        da = ds[name]
        if not isinstance(da, xr.DataArray):
            raise TypeError(f"Dataset variable '{name}' is not a DataArray.")
        return da

    @staticmethod
    def _reduce_to_2d(da: xr.DataArray) -> xr.DataArray:
        # Select first along known axes if present
        if "time" in da.dims:
            da = da.isel(time=0)
        if "depth" in da.dims:
            da = da.isel(depth=0)
        da = da.squeeze(drop=True)
        return da

    @staticmethod
    def _transpose_to_lat_lon(da: xr.DataArray) -> xr.DataArray:
        dims = set(da.dims)
        if "latitude" not in dims or "longitude" not in dims:
            raise KeyError(
                f"Required dims ('latitude','longitude') not found in {tuple(da.dims)}."
            )
        return da.transpose("latitude", "longitude")

    @staticmethod
    def _bitmask_classify(da: xr.DataArray, sea_bit: int) -> np.ndarray:
        if sea_bit <= 0:
            raise ValueError("sea_bit must be a positive integer (e.g., 1).")
        arr = np.asarray(da.values)
        # Build a missing mask ONLY for bitwise path (NaN/Inf + _FillValue)
        is_missing = np.zeros(arr.shape, dtype=bool)
        if np.issubdtype(arr.dtype, np.floating):
            is_missing |= ~np.isfinite(arr)
        fv = da.attrs.get("_FillValue")
        if fv is not None:
            is_missing |= arr == fv
        # Neutralize missing and safely bit-and
        safe = np.array(arr, copy=True)
        safe[is_missing] = 0
        uint = safe.astype(np.uint16, copy=False)
        sea = (uint & sea_bit) != 0
        # Explicitly mark missing as not-sea (redundant but explicit)
        sea[is_missing] = False
        return sea

    @staticmethod
    def _infer_sea_value_from_long_name(da: xr.DataArray) -> int:
        long_name = da.attrs.get("long_name")
        if not long_name:
            raise KeyError("Cannot infer sea_value: 'long_name' attribute is missing.")
        m = re.search(r"(\d+)\s*=\s*sea", str(long_name), flags=re.IGNORECASE)
        if not m:
            raise KeyError(
                "Cannot infer sea_value: expected pattern like '1 = sea' in 'long_name'."
            )
        return int(m.group(1))
