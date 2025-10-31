from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.app.downloader import Downloader
from src.app.downloader_async import DownloaderAsync
from src.app.jobs import DownloadJob
from src.app.layout import ProjectLayout


@dataclass(frozen=True)
class SchedulerContext:
    layout: ProjectLayout
    product_slug: str


class SerialScheduler:
    def __init__(self, cm_handle) -> None:
        self._downloader = Downloader(cm_handle=cm_handle)

    def download(self, jobs: List[DownloadJob], ctx: SchedulerContext) -> None:
        for job in jobs:
            ctx.layout.ensure_product_bbox(ctx.product_slug, job.bbox_id)
            ctx.layout.ensure_nc_tile_dir(
                ctx.product_slug, job.bbox_id, job.tile_id_padded
            )

            nc_path = ctx.layout.nc_path(
                product=ctx.product_slug,
                bbox_id=job.bbox_id,
                tile_id_padded=job.tile_id_padded,
                day_iso=job.day.isoformat(),
            )
            if ProjectLayout.exists_nonempty(nc_path):
                continue
            self._downloader.download_day(job, nc_path)


class AsyncScheduler:
    def __init__(self, cm_handle, *, max_concurrency: int) -> None:
        self._downloader = DownloaderAsync(cm_handle=cm_handle)
        self._max_concurrency = max(1, int(max_concurrency))

    def download(self, jobs: List[DownloadJob], ctx: SchedulerContext) -> None:
        asyncio.run(self._download_async(jobs, ctx))

    async def _download_async(
        self, jobs: List[DownloadJob], ctx: SchedulerContext
    ) -> None:
        groups: Dict[Tuple[str, str], List[DownloadJob]] = {}
        for job in jobs:
            key = (job.bbox_id, job.tile_id_padded)
            groups.setdefault(key, []).append(job)

        sem = asyncio.Semaphore(self._max_concurrency)

        async def run_tile(
            bbox_id: str, tile_id: str, tile_jobs: List[DownloadJob]
        ) -> None:
            async with sem:
                ctx.layout.ensure_product_bbox(ctx.product_slug, bbox_id)
                ctx.layout.ensure_nc_tile_dir(ctx.product_slug, bbox_id, tile_id)

                for job in tile_jobs:
                    nc_path = ctx.layout.nc_path(
                        product=ctx.product_slug,
                        bbox_id=job.bbox_id,
                        tile_id_padded=job.tile_id_padded,
                        day_iso=job.day.isoformat(),
                    )
                    if ProjectLayout.exists_nonempty(nc_path):
                        continue
                    await self._downloader.download_day_async(job, nc_path)

        tasks = [
            asyncio.create_task(run_tile(b, t, js)) for (b, t), js in groups.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errs = [r for r in results if isinstance(r, Exception)]
        if errs:
            raise errs[0]
