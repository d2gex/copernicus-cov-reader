from __future__ import annotations

import copernicusmarine as cm
import pandas as pd

from src.app.bbox_factory import BBoxFactory
from src.app.downloader import Downloader
from src.app.layout import ProjectLayout
from src.app.nc_to_csv_batch_converter import NCTileToCSVBatchConverter
from src.app.orchestrator import TileDayOrchestrator
from src.config import cfg  # unified config object
from src.copernicus.cm_credentials import CMCredentials


def fetch_data(tiles_df: pd.DataFrame) -> None:
    bboxes = BBoxFactory(
        min_lon=cfg.region_min_lon,
        min_lat=cfg.region_min_lat,
        max_lon=cfg.region_max_lon,
        max_lat=cfg.region_max_lat,
        lat_band_count=cfg.lat_band_count,
    ).build(tiles_df)

    orch = TileDayOrchestrator(
        dataset_id=cfg.dataset_id,
        variables=list(cfg.variables),
        spatial_resolution_deg=cfg.spatial_resolution_deg,
    )
    jobs = orch.build_jobs(df=tiles_df, bboxes=bboxes)

    layout = ProjectLayout(root=cfg.output_root)
    dl = Downloader(cm_handle=cm)

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


def process_data() -> None:
    NCTileToCSVBatchConverter(
        output_root=cfg.output_root,
        product_slug=cfg.product_slug,
        variables=list(cfg.variables),
    ).run()


def run() -> None:
    CMCredentials().ensure_present()
    tiles_df = pd.read_csv(
        cfg.output_path
        / cfg.product_owner
        / cfg.product_slug
        / "test_tiles_with_date_db.csv"
    )
    fetch_data(tiles_df)  # uncomment when running the downloader stage
    process_data()


if __name__ == "__main__":
    run()
