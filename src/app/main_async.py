from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import copernicusmarine as cm  # SDK handle passed to the downloader
import pandas as pd

from src import config
from src.app.bbox_factory import BBoxFactory
from src.app.downloader_async import DownloaderAsync
from src.app.jobs import DownloadJob
from src.app.layout import ProjectLayout
from src.app.orchestrator import TileDayOrchestrator
from src.copernicus.cm_credentials import CMCredentials


@dataclass(frozen=True)
class Config:
    # Product & I/O
    output_root: Path
    product_slug: str
    dataset_id: str
    variables: Tuple[str, ...]
    csv_path: Path

    # Spatial / partitioning
    spatial_resolution_deg: float
    region_min_lon: float
    region_min_lat: float
    region_max_lon: float
    region_max_lat: float
    lat_band_count: int

    # Concurrency
    max_concurrency: int = 4  # 1 = serial (current behavior). >=2 = tiles in parallel.


async def _run_async(cfg: Config) -> None:
    ## Ensure Copernicus Marine credentials exist (your existing helper)
    creds = CMCredentials()
    creds.ensure_present()

    # Input
    df = pd.read_csv(cfg.csv_path)

    # Build bboxes (splitter needs lon/lat coords)
    bboxes = BBoxFactory(
        min_lon=cfg.region_min_lon,
        min_lat=cfg.region_min_lat,
        max_lon=cfg.region_max_lon,
        max_lat=cfg.region_max_lat,
        lat_band_count=cfg.lat_band_count,
    ).build(df)

    # Jobs (deterministic: sorted tile -> day)
    orch = TileDayOrchestrator(
        dataset_id=cfg.dataset_id,
        variables=list(cfg.variables),
        spatial_resolution_deg=cfg.spatial_resolution_deg,
    )
    jobs = orch.build_jobs(df=df, bboxes=bboxes)

    layout = ProjectLayout(root=cfg.output_root)
    dl = DownloaderAsync(cm_handle=cm)

    # Group by tile (bbox_id, tile_id_padded); keep order inside each group
    groups: Dict[tuple[str, str], List[DownloadJob]] = {}
    for job in jobs:
        key = (job.bbox_id, job.tile_id_padded)
        groups.setdefault(key, []).append(job)

    sem = asyncio.Semaphore(max(1, int(cfg.max_concurrency)))

    async def process_tile(
        bbox_id: str, tile_id: str, tile_jobs: List[DownloadJob]
    ) -> None:
        # Prepare dirs once per tile
        layout.ensure_product_bbox(cfg.product_slug, bbox_id)
        layout.ensure_nc_tile_dir(cfg.product_slug, bbox_id, tile_id)

        # Serial within a tile (preserve tile-day order)
        for job in tile_jobs:
            nc_path = layout.nc_path(
                product=cfg.product_slug,
                bbox_id=job.bbox_id,
                tile_id_padded=job.tile_id_padded,
                day_iso=job.day.isoformat(),
            )
            if ProjectLayout.exists_nonempty(nc_path):
                # skip silently; deterministic idempotency
                pass
            else:
                await dl.download_day_async(job, nc_path)

    async def worker(bbox_id: str, tile_id: str, tile_jobs: List[DownloadJob]) -> None:
        async with sem:
            await process_tile(bbox_id, tile_id, tile_jobs)

    # Schedule tiles in parallel, each tile runs sequentially
    tasks = [asyncio.create_task(worker(b, t, js)) for (b, t), js in groups.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Optional: surface first exception if any
    errs = [r for r in results if isinstance(r, Exception)]
    if errs:
        raise errs[0]


def run(cfg: Config) -> None:
    if cfg.max_concurrency <= 1:
        # Fallback to serial execution using the same async pipeline
        asyncio.run(_run_async(Config(**{**cfg.__dict__, "max_concurrency": 1})))
    else:
        asyncio.run(_run_async(cfg))


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
