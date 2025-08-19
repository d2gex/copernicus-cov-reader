from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class GridSpec:
    """Rectilinear grid spec: 1-D lon/lat arrays, names, and a strict hash."""

    lon_name: str
    lat_name: str
    lons: np.ndarray  # shape (nx,), float64, contiguous
    lats: np.ndarray  # shape (ny,), float64, contiguous
    grid_hash: str  # SHA-256 over lon/lat bytes + coord names

    @property
    def nx(self) -> int:
        return int(self.lons.size)

    @property
    def ny(self) -> int:
        return int(self.lats.size)

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        lon_candidates: Tuple[str, ...] = ("lon", "longitude", "x"),
        lat_candidates: Tuple[str, ...] = ("lat", "latitude", "y"),
    ) -> "GridSpec":
        """Builds a grid spec from an xarray Dataset; fails for non-1D lon/lat."""
        lon_name = next((n for n in lon_candidates if n in ds), None)
        lat_name = next((n for n in lat_candidates if n in ds), None)
        if not lon_name or not lat_name:
            raise ValueError("Could not infer lon/lat names.")
        lons = np.asarray(ds[lon_name].values)
        lats = np.asarray(ds[lat_name].values)
        if lons.ndim != 1 or lats.ndim != 1:
            raise ValueError(
                "Curvilinear grids are not supported (lon/lat must be 1-D)."
            )

        # Canonicalize and hash raw bytes (strict equality check).
        lons64 = np.ascontiguousarray(lons, dtype="<f8")
        lats64 = np.ascontiguousarray(lats, dtype="<f8")
        h = hashlib.sha256()
        h.update(b"lon")
        h.update(lon_name.encode())
        h.update(lons64.tobytes())
        h.update(b"lat")
        h.update(lat_name.encode())
        h.update(lats64.tobytes())
        grid_hash = h.hexdigest()

        return cls(
            lon_name=lon_name,
            lat_name=lat_name,
            lons=lons64,
            lats=lats64,
            grid_hash=grid_hash,
        )

    def validate(self, ds: xr.Dataset) -> None:
        """Fails if ds grid differs from this spec."""
        other = GridSpec.from_dataset(ds, (self.lon_name,), (self.lat_name,))
        if other.grid_hash != self.grid_hash:
            raise ValueError(
                "Dataset grid differs from reference grid (hash mismatch)."
            )
