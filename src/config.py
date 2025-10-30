import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()  # expects a .env at project root


def _req(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise ValueError(f"Missing required environment variable: {name}")
    return v


@dataclass(frozen=True)
class Config:
    # Raw values from .env
    output_path: Path
    input_path: Path

    s3_bucket: str
    s3_output_prefix: str

    aws_nc_root: Path
    aws_csv_root: Path
    aws_clean: bool
    aws_verbose: bool
    aws_policy: str  # "skip_if_exists" | "always_put"
    aws_progress_step: float  # (0, 1]

    product_owner: str
    product_slug: str
    dataset_id: str
    variables: Tuple[str, ...]

    spatial_resolution_deg: float
    region_min_lon: float
    region_max_lon: float
    region_min_lat: float
    region_max_lat: float
    lat_band_count: int

    # Derived
    output_root: Path  # OUTPUT_PATH / PRODUCT_OWNER / "data"


cfg = Config(
    output_path=Path(_req("OUTPUT_PATH")),
    input_path=Path(_req("INPUT_PATH")),
    s3_bucket=_req("S3_BUCKET"),
    s3_output_prefix=_req("S3_OUTPUT_PREFIX"),
    aws_nc_root=Path(_req("AWS_NC_ROOT")),
    aws_csv_root=Path(_req("AWS_CSV_ROOT")),
    aws_clean=_req("AWS_CLEAN").lower() in {"1", "true", "yes"},
    aws_verbose=_req("AWS_VERBOSE").lower() in {"1", "true", "yes"},
    aws_policy=_req("AWS_POLICY"),
    aws_progress_step=float(_req("AWS_PROGRESS_STEP")),
    product_owner=_req("PRODUCT_OWNER"),
    product_slug=_req("PRODUCT_SLUG").lower(),
    dataset_id=_req("DATASET_ID"),
    variables=tuple(_req("VARIABLES").split(",")),
    spatial_resolution_deg=float(_req("SPATIAL_RESOLUTION_DEG")),
    region_min_lon=float(_req("REGION_MIN_LON")),
    region_max_lon=float(_req("REGION_MAX_LON")),
    region_min_lat=float(_req("REGION_MIN_LAT")),
    region_max_lat=float(_req("REGION_MAX_LAT")),
    lat_band_count=int(_req("LAT_BAND_COUNT")),
    output_root=(Path(_req("OUTPUT_PATH")) / _req("PRODUCT_OWNER") / "data"),
)
