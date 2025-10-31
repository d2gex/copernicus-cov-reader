from __future__ import annotations

import logging
import time
from typing import List

import copernicusmarine as cm
import pandas as pd

from src.app.bbox_factory import BBoxFactory
from src.app.jobs import DownloadJob
from src.app.layout import ProjectLayout
from src.app.nc_to_csv_batch_converter import NCTileToCSVBatchConverter
from src.app.orchestrator import TileDayOrchestrator
from src.app.scheduler import AsyncScheduler, SchedulerContext, SerialScheduler
from src.config import cfg


def timed(label: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info("%s: start", label)
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                logging.info("%s: done in %.2fs", label, dt)

        return wrapper

    return decorator


class DownloadAndConvert:
    def __init__(self, tiles_df: pd.DataFrame) -> None:
        self.tiles_df = tiles_df
        self._layout = ProjectLayout(root=cfg.output_root)

    @staticmethod
    def _build_jobs(tiles_df: pd.DataFrame) -> List[DownloadJob]:
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
        return orch.build_jobs(df=tiles_df, bboxes=bboxes)

    @timed("convert")
    def _convert(self) -> None:
        NCTileToCSVBatchConverter(
            output_root=cfg.output_root,
            product_slug=cfg.product_slug,
            variables=list(cfg.variables),
        ).run()

    @timed("download")
    def _download(self) -> None:
        jobs = self._build_jobs(self.tiles_df)

        ctx = SchedulerContext(layout=self._layout, product_slug=cfg.product_slug)

        max_conc = int(cfg.aws_max_concurrency)  # from .env via src.config
        if max_conc <= 1:
            SerialScheduler(cm_handle=cm).download(jobs, ctx)
        else:
            AsyncScheduler(cm_handle=cm, max_concurrency=max_conc).download(jobs, ctx)

    def run(self, download_and_convert: int = 0) -> None:
        if download_and_convert == 0:
            self._download()
            self._convert()
        elif download_and_convert == 1:
            self._download()
        else:
            self._convert()
