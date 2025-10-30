from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import copernicusmarine as cm  # SDK handle passed through to Downloader
import pandas as pd

# Repo imports
from src import config
from src.app.bbox_factory import BBoxFactory
from src.app.downloader import Downloader
from src.app.layout import ProjectLayout
from src.app.orchestrator import TileDayOrchestrator
from src.copernicus.cm_credentials import CMCredentials


@dataclass(frozen=True)
class Config:
    output_root: Path
    product_slug: str
    dataset_id: str
    variables: Tuple[str, ...]
    csv_path: Path
    spatial_resolution_deg: float

    region_min_lon: float
    region_min_lat: float
    region_max_lon: float
    region_max_lat: float
    lat_band_count: int


def run(cfg: Config) -> None:
    # Ensure Copernicus Marine credentials exist (your existing helper)
    creds = CMCredentials()
    creds.ensure_present()

    # Read the tiles CSV
    df = pd.read_csv(cfg.csv_path)

    # Build latitude-band bboxes using the actual splitter (requires lon/lat coords)
    bboxes = BBoxFactory(
        min_lon=cfg.region_min_lon,
        min_lat=cfg.region_min_lat,
        max_lon=cfg.region_max_lon,
        max_lat=cfg.region_max_lat,
        lat_band_count=cfg.lat_band_count,
    ).build(df)  # pass the DataFrame so the splitter can assign bands

    # Orchestrate per-tile-per-day jobs (pure, in-memory)
    orch = TileDayOrchestrator(
        dataset_id=cfg.dataset_id,
        variables=list(cfg.variables),
        spatial_resolution_deg=cfg.spatial_resolution_deg,
    )
    jobs = orch.build_jobs(df=df, bboxes=bboxes)

    # Filesystem layout (mirrors old application layout + <tile_id>/ level)
    layout = ProjectLayout(root=cfg.output_root)

    # Downloader uses the real SDK handle and optional extra kwargs
    dl = Downloader(cm_handle=cm)

    # Deterministic execution: tile → days order comes from the orchestrator
    for job in jobs:
        layout.ensure_product_bbox(cfg.product_slug, job.bbox_id)
        layout.ensure_nc_tile_dir(cfg.product_slug, job.bbox_id, job.tile_id_padded)

        nc_path = layout.nc_path(
            product=cfg.product_slug,
            bbox_id=job.bbox_id,
            tile_id_padded=job.tile_id_padded,
            day_iso=job.day.isoformat(),
        )
        if not ProjectLayout.exists_nonempty(nc_path):
            dl.download_day(job, nc_path)


if __name__ == "__main__":
    cfg = Config(
        output_root=config.OUTPUT_PATH / "utpb" / "data",
        product_slug="IBI_MULTIYEAR_BGC_005_003".lower(),
        dataset_id="cmems_mod_ibi_bgc_my_0.083deg-3D_P1D-m",
        variables=("chl", "o2", "nppv"),
        csv_path=config.OUTPUT_PATH / "utpb" / "tiles_with_date_db.csv",
        spatial_resolution_deg=0.083,
        region_min_lon=-9.929866667,
        region_max_lon=-6.9418,
        region_min_lat=41.7667,
        region_max_lat=44.15243,
        lat_band_count=2,
    )
    run(cfg)
