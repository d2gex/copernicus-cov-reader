from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Mapping, Optional, Sequence

from src.app.downloader import Downloader
from src.app.jobs import DownloadJob


class DownloaderAsync(Downloader):
    def __init__(
        self, cm_handle, *, extra_kwargs: Optional[Mapping[str, object]] = None
    ) -> None:
        super().__init__(cm_handle=cm_handle, extra_kwargs=extra_kwargs)

    async def download_day_async(self, job: DownloadJob, outfile: Path) -> None:
        # Run the existing synchronous download_day in a worker thread.
        await asyncio.to_thread(self.download_day, job, outfile)

    async def download_static_async(
        self,
        dataset_id: str,
        area: Sequence[float],  # [lon_min, lat_min, lon_max, lat_max]
        outfile: Path,
        variables: Sequence[str] | None = None,
        request_opts: Mapping[str, object] | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.download_static, dataset_id, area, outfile, variables, request_opts
        )
