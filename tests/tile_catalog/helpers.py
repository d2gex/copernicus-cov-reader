import numpy as np
import xarray as xr


def shallowest_layer(mask: xr.DataArray) -> xr.DataArray:
    """Independent collapse: choose shallowest by index (argmin of depth coord)."""
    depth_dim = next((d for d in ("depth", "z", "lev") if d in mask.dims), None)
    if not depth_dim:
        return mask
    idx = int(np.argmin(mask[depth_dim].values))
    return mask.isel({depth_dim: idx})


def expected_ids() -> np.ndarray:
    # For the 3×3 mask in fixtures:
    # True at (0,0), (0,2), (1,1), (2,0), (2,2) → IDs 0..4 in row-major order.
    return np.arange(5, dtype=np.int64)


def expected_ij():
    # tile_id k → (j, i)
    return {
        0: (0, 0),
        1: (0, 2),
        2: (1, 1),
        3: (2, 0),
        4: (2, 2),
    }


def expected_lonlat():
    # Using lats=[10,20,30], lons=[100,200,300]
    # in the same order as IDs above.
    lon = np.array([100, 300, 200, 100, 300])
    lat = np.array([10, 10, 20, 30, 30])
    return lon, lat
