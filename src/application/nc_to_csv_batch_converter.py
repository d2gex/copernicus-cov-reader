# src/application/nc_to_csv_batch_converter.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import xarray as xr

from src.application.app_ds import ProductSpec, StaticSpec
from src.application.plan_builder import Plan
from src.application.project_layout import ProjectLayout
from src.data_processing.nc_to_csv import NcToCsvConverter
from src.data_processing.sea_mask_builder import SeaMaskBuilder


class NcToCsvBatchConverter:
    """
    Flattens .nc to CSV using NcToCsvConverter, writing to {project}/data/{product}/{bbox}/csv/all.

    If a StaticSpec is provided, this class reads the per-bbox static .nc (downloaded previously)
    and builds the 2-D boolean sea/land mask via SeaMaskBuilder before converting each bbox.
    """

    def __init__(
        self,
        layout: ProjectLayout,
        *,
        var_names: Sequence[str],
        time_dim: str = "time",
        depth_dim: Optional[str] = "depth",
    ) -> None:
        if not var_names:
            raise ValueError("var_names must be non-empty.")
        self.layout = layout
        self.var_names = list(var_names)
        self.time_dim = time_dim
        self.depth_dim = depth_dim

    # ---- protected helpers ----

    def _extract_sea_land_mask(
        self, static_nc_path: Path, static: StaticSpec
    ) -> np.ndarray:
        """
        Open the static .nc and compute the boolean sea/land mask with SeaMaskBuilder.
        Returns a 2-D np.bool_ array (latitude, longitude).
        """
        if not static_nc_path.exists():
            raise FileNotFoundError(f"Static file not found: {static_nc_path}")

        builder = SeaMaskBuilder(
            mask_name=static.mask_var,
            is_bit=static.is_bit,
            sea_value=static.sea_value,
        )
        with xr.open_dataset(static_nc_path, decode_times=True) as ds:
            sea_mask = builder.build(ds)  # 2-D boolean (lat, lon)
        return sea_mask

    def run(
        self,
        plan: Plan,
        product: ProductSpec,
        static: Optional[StaticSpec] = None,
    ) -> List[Path]:
        written: List[Path] = []

        for bp in plan.bboxes:
            nc_dir = self.layout.nc_dir(product, bp)
            csv_all_dir = self.layout.csv_all_dir(product, bp)

            sea_land_mask = None
            if static is not None:
                static_path = self.layout.static_path(product, bp)
                sea_land_mask = self._extract_sea_land_mask(static_path, static)

            converter = NcToCsvConverter(
                var_names=self.var_names,
                time_dim=self.time_dim,
                depth_dim=self.depth_dim,
                sea_land_mask=sea_land_mask,  # boolean (lat, lon) or None, per product
            )

            csv_paths = converter.generate_period_csvs(nc_dir, csv_all_dir)
            written.extend(csv_paths)

        return written
