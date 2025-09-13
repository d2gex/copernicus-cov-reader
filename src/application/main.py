from __future__ import annotations

from pathlib import Path
from typing import Optional

from src import config
from src.application.app_ds import ProductSpec, StaticSpec
from src.application.cm_downloader import CMDownloader
from src.application.nc_to_csv_batch_converter import NcToCsvBatchConverter
from src.application.plan_builder import PlanBuilder
from src.application.project_layout import ProjectLayout
from src.bounding_box.bounding_box import BoundingBox
from src.copernicus.cm_credentials import CMCredentials

# ------------------------------
# CONSTANTS (replace with real values)
# ------------------------------
PROJECT_ROOT: Path = config.OUTPUT_PATH / "agata"

# Area of interest (min_lon, min_lat, max_lon, max_lat)
MASTER_BBOX = BoundingBox(
    min_lon=-44.019036, max_lon=-8.43452, min_lat=14.494878, max_lat=43.514585
)

# Number of latitude bands (bbox splits)
N_BANDS: int = 5

# Temporal window (string or datetime, TimeBlockSplitter handles normalization)
TIME_START: str = "2018-01-01"
TIME_END: str = "2012-01-01"

# Number of time blocks to split the period into
N_TIME_CHUNKS: int = 24

# Product configuration (single product for first run)
PRODUCT_NAME: str = "C3S-GLO-SST-L4-REP-OBS-SST"
PRODUCT_DATASET_ID: str = "SST_GLO_SST_L4_REP_OBSERVATIONS_010_024"  # placeholder
PRODUCT_VARIABLES: list[str] = ["analysed_sst"]

# Static (mask) specification — set to None if product has no static dataset
STATIC_SPEC: Optional[StaticSpec] = StaticSpec(
    dataset_id="SST_GLO_SST_L4_REP_OBSERVATIONS_010_024",
    mask_var="mask",  # example
)

# CSV converter configuration
VAR_NAMES: list[str] = PRODUCT_VARIABLES
TIME_DIM: str = "time"
DEPTH_DIM: str | None = "depth"

SKIP_EXISTING: bool = True


def main() -> None:
    # Build plan
    planner = PlanBuilder(
        project_root=PROJECT_ROOT,
        master_bbox=MASTER_BBOX,
        n_bands=N_BANDS,
        start=TIME_START,
        end=TIME_END,
        n_chunks=N_TIME_CHUNKS,
    )
    plan = planner.build()

    # Product & layout
    product = ProductSpec(
        name=PRODUCT_NAME, dataset_id=PRODUCT_DATASET_ID, variables=PRODUCT_VARIABLES
    )
    layout = ProjectLayout(PROJECT_ROOT)

    # Downloader
    creds = CMCredentials()
    downloader = CMDownloader(
        layout=layout, credentials=creds, skip_existing=SKIP_EXISTING
    )

    # 1) Static file(s): one per bbox, saved under bbox/static/. No extraction here.
    downloader.get_static_details(plan, product, STATIC_SPEC)

    # 2) Time-sliced data
    downloader.get_data(plan, [product])

    # 3) CSV conversion (extract masks here, per bbox)
    csv_batch = NcToCsvBatchConverter(
        layout=layout, var_names=VAR_NAMES, time_dim=TIME_DIM, depth_dim=DEPTH_DIM
    )
    csv_batch.run(plan, product, static=STATIC_SPEC)

    # Minimal summary
    print("Project root:", PROJECT_ROOT)
    print("Product:", product.slug())
    print("BBoxes:", [bp.folder_name() for bp in plan.bboxes])
    print(
        "Periods:", [(tb.start.isoformat(), tb.end.isoformat()) for tb in plan.periods]
    )


if __name__ == "__main__":
    main()
