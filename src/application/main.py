from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config
from src.application.app_ds import ProductSpec, StaticSpec
from src.application.nc_to_csv_batch_converter import NcToCsvBatchConverter
from src.application.plan_builder import PlanBuilder
from src.application.project_layout import ProjectLayout
from src.bounding_box.bounding_box import BoundingBox

logger = logging.getLogger(__name__)

# ------------------------------
# CONSTANTS (replace with real values)
# ------------------------------
PROJECT_ROOT: Path = config.OUTPUT_PATH / "utpb"

# Area of interest (min_lon, min_lat, max_lon, max_lat)
# BoundingBox(min_lon=-9.929866667, max_lon=-6.9418, min_lat=41.7667, max_lat=44.1524333333333)
MASTER_BBOX = BoundingBox(
    min_lon=-9.929866667, max_lon=-6.9418, min_lat=41.7667, max_lat=44.15243
)

# Number of latitude bands (bbox splits)
N_BANDS: int = 2

# Temporal window (string or datetime, TimeBlockSplitter handles normalization) YYYY-MM-DD
TIME_START: str = "1999-01-01"
TIME_END: str = "1999-12-31"
MIN_DEPTH: float = 0.51
MAX_DEPTH: float = 628

# Number of time blocks to split the period into
N_TIME_CHUNKS: int = 1

# Product configuration (single product for first run)
PRODUCT_NAME: str = "IBI_MULTIYEAR_BGC_005_003"
PRODUCT_DATASET_ID: str = "cmems_mod_ibi_bgc_my_0.083deg-3D_P1D-m"  # placeholder
PRODUCT_VARIABLES: list[str] = ["chl", "o2", "nppv"]

# Static (mask) specification — set to None if product has no static dataset
STATIC_SPEC: Optional[StaticSpec] = StaticSpec(
    dataset_id="cmems_mod_ibi_bgc_my_0.083deg-3D_static",
    mask_var="mask",
    is_bit=True,
    sea_value=1,
)

# CSV converter configuration
TIME_DIM: str = "time"
DEPTH_DIM: str | None = "depth"
SKIP_EXISTING: bool = True


def main() -> None:
    # Build plan
    coords_df = pd.read_csv(config.INPUT_PATH / "utpb" / "clean_haul_db.csv")
    assert len(coords_df) > 0, "No coordinates found"
    planner = PlanBuilder(
        project_root=PROJECT_ROOT,
        master_bbox=MASTER_BBOX,
        n_bands=N_BANDS,
        start=TIME_START,
        end=TIME_END,
        n_chunks=N_TIME_CHUNKS,
    )
    plan = planner.build(coords_df)

    # Product & layout
    product = ProductSpec(
        name=PRODUCT_NAME, dataset_id=PRODUCT_DATASET_ID, variables=PRODUCT_VARIABLES
    )
    layout = ProjectLayout(PROJECT_ROOT)

    # # Downloader
    # creds = CMCredentials()
    # creds.ensure_present()
    #
    # downloader = CMDownloader(
    #     layout=layout,
    #     skip_existing=SKIP_EXISTING,
    #     min_depth=MIN_DEPTH,
    #     max_depth=MAX_DEPTH,
    # )
    #
    # # 1) Static file(s): one per bbox, saved under bbox/static/. No extraction here.
    # logger.info("Downloading static data...")
    # downloader.get_static_details(plan, product, STATIC_SPEC)
    # logger.info("Downloading static data... done.")
    #
    # # 2) Time-sliced data
    # logger.info("Downloading time-sliced data...")
    # downloader.get_data(plan, [product])
    # logger.info("Downloading time-sliced data... done.")

    # 3) CSV conversion (extract masks here, per bbox)
    logger.info("Converting to CSV...")
    csv_batch = NcToCsvBatchConverter(
        layout=layout,
        var_names=product.variables,
        time_dim=TIME_DIM,
        depth_dim=DEPTH_DIM,
    )
    csv_batch.run(plan, product, static=STATIC_SPEC)
    logger.info("Converting to CSV... done.")

    # # 4) Group CSVs (simple pandas concatenation using ProjectLayout discovery)
    # logger.info("Amalgamating CSVs...")
    # d_amalgamator = DataAmalgamator(
    #     layout=layout,
    #     output_name="group.csv",
    #     logger=lambda m: print(m),  # or a structured logger if you have one
    # )
    # group_paths = d_amalgamator.run(product=product, plan=plan)
    # logger.info("Amalgamating CSVs... done.")

    # Minimal summary
    print("Project root:", PROJECT_ROOT)
    print("Product:", product.slug())
    print("BBoxes:", [bp.folder_name() for bp in plan.bboxes])
    print(
        "Periods:", [(tb.start.isoformat(), tb.end.isoformat()) for tb in plan.periods]
    )
    # print("Group CSVs written:", len(group_paths))


if __name__ == "__main__":
    main()
