from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import xarray as xr


class NCTileToCSVConverter:
    def __init__(self, variables: Sequence[str]) -> None:
        self.variables = tuple(variables)

    def run(self, nc_files: Iterable[Path]) -> pd.DataFrame:
        paths = [str(p) for p in nc_files]
        if not paths:
            return pd.DataFrame()

        ds = xr.open_mfdataset(paths, combine="by_coords")
        try:
            sel = ds[list(self.variables)]  # let KeyError surface if misconfigured
            df = sel.to_dataframe().reset_index()  # raw data, no spatial reduction
            if "time" in df.columns:
                df = df.sort_values("time").reset_index(drop=True)
            return df
        finally:
            ds.close()
