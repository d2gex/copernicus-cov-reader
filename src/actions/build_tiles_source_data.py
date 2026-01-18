from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import cfg  # unified config object
from src.data_processing.assign_hauls_to_tiles_id import HaulTileAssigner, StaticSpec
from src.data_processing.tile_days_builder import ColumnConfig, TileDaysBuilder


def assign_tiles_to_hauls(
    hauls: pd.DataFrame,
    static_nc_path: Path,
    mask_var: str = "mask",
    is_bit: bool = True,
    sea_value: int = 1,
    lon_col: str = "lon",
    lat_col: str = "lat",
    time_col: str = "time",
    tolerance_deg: Optional[float] = None,
) -> pd.DataFrame:
    lat0_hint = float(np.nanmedian(hauls[lat_col].to_numpy()))

    assigner = HaulTileAssigner(
        static_spec=StaticSpec(
            path=static_nc_path, mask_var=mask_var, is_bit=is_bit, sea_value=sea_value
        ),
        lon_col=lon_col,
        lat_col=lat_col,
        time_col=time_col,
    )
    assigner.load_static_and_build_index(lat0_hint=lat0_hint)
    enriched = assigner.assign(hauls, tolerance_deg=tolerance_deg)
    return enriched


def build_tiles_dbs(
    hauls_db: pd.DataFrame, static_layer_path: Path, mask_var: str = "mask"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    enriched = assign_tiles_to_hauls(
        hauls=hauls_db,
        static_nc_path=static_layer_path,
        mask_var=mask_var,
        is_bit=True,
        sea_value=1,
        tolerance_deg=None,
    )

    builder = TileDaysBuilder(columns=ColumnConfig(time="time"))
    per_day_df = builder.build_per_day(enriched)
    return enriched, per_day_df


def run() -> None:
    owner = cfg.product_owner

    # haul_df = pd.read_csv(cfg.input_path / owner / "capturas_2023_nueva.csv", encoding="latin-1")
    # haul_correction_df = pd.read_csv(cfg.input_path / owner / "original" / "land_faroff_at_sea_features_corrected.csv")
    static_nc = cfg.input_path / owner / "static_data" / cfg.static_filename

    # selected_columns = {
    #     "haul_id": "Idlance",
    #     "time": "date",
    #     "lat": "lat",
    #     "lon": "lon",
    #     "depth": "depth",
    # }

    # haul_db_builder = HaulDbBuilder(haul_df, haul_correction_df)
    # clean_haul_df = haul_db_builder.run(selected_columns)

    clean_haul_df = pd.read_csv(cfg.input_path / owner / "clean_haul_db.csv")
    haul_with_tiles_df, tiles_with_date_df = build_tiles_dbs(
        clean_haul_df, static_nc, mask_var="mask"
    )

    # clean_haul_df.to_csv(out_dir / "clean_haul_db.csv", index=False)
    out_dir = cfg.output_path / owner / cfg.product_slug
    haul_with_tiles_df.to_csv(out_dir / "haul_with_tiles_db.csv", index=False)
    tiles_with_date_df.to_csv(out_dir / "tiles_with_date_db.csv", index=False)


if __name__ == "__main__":
    run()
