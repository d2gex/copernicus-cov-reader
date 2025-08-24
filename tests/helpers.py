import numpy as np
import xarray as xr


def shallowest_layer(mask: xr.DataArray) -> xr.DataArray:
    """Independent collapse: choose shallowest by index (argmin of depth coord)."""
    depth_dim = next((d for d in ("depth", "z", "lev") if d in mask.dims), None)
    if not depth_dim:
        return mask
    idx = int(np.argmin(mask[depth_dim].values))
    return mask.isel({depth_dim: idx})
