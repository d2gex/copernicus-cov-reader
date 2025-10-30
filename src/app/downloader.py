from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Optional, Sequence

# Use your existing thin wrapper
from src.copernicus.cm_subset_client import CMSubsetClient

from .jobs import DownloadJob


class Downloader:
    def __init__(
        self, cm_handle, *, extra_kwargs: Optional[Mapping[str, object]] = None
    ) -> None:
        """
        cm_handle: the imported 'copernicusmarine' SDK object (i.e., 'import copernicusmarine as cm')
        extra_kwargs: forwarded to CMSubsetClient(extra_kwargs=...) for every request.
        """
        self._cm = cm_handle
        self._extra_kwargs = dict(extra_kwargs or {})

    def _new_client(
        self,
        dataset_id: str,
        variables: Sequence[str],
        min_depth: float | None,
        max_depth: float | None,
    ) -> CMSubsetClient:
        return CMSubsetClient(
            cm=self._cm,
            dataset_id=dataset_id,
            variables=list(variables),
            min_depth=min_depth,
            max_depth=max_depth,
            extra_kwargs=self._extra_kwargs or None,
        )

    def download_day(self, job: DownloadJob, outfile: Path) -> None:
        # YYYY-MM-DD strings as required by CMSubsetClient
        start = datetime(job.day.year, job.day.month, job.day.day, 0, 0, 0)
        end = start + timedelta(days=1) - timedelta(seconds=1)

        # job.area = (lon_min, lat_min, lon_max, lat_max)
        lon_min, lat_min, lon_max, lat_max = job.area
        bbox = (float(lon_min), float(lon_max), float(lat_min), float(lat_max))

        client = self._new_client(
            dataset_id=job.dataset_id,
            variables=job.variables,
            min_depth=float(job.z_min),
            max_depth=float(job.z_max),
        )

        out_dir = outfile.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        client.subset_one(
            bbox=bbox,
            output_filename=outfile.name,
            output_directory=str(out_dir),
            start_datetime=start.date().isoformat(),
            end_datetime=end.date().isoformat(),
        )

    def download_static(
        self,
        dataset_id: str,
        area: Sequence[float],  # [lon_min, lat_min, lon_max, lat_max]
        outfile: Path,
        variables: Sequence[str] | None = None,
        request_opts: Mapping[str, object] | None = None,
    ) -> None:
        # Static: no time window, no vertical range.
        lon_min, lat_min, lon_max, lat_max = area
        bbox = (float(lon_min), float(lon_max), float(lat_min), float(lat_max))

        # Merge per-call request_opts into extra_kwargs if provided
        extra = dict(self._extra_kwargs)
        if request_opts:
            extra.update(request_opts)

        client = CMSubsetClient(
            cm=self._cm,
            dataset_id=dataset_id,
            variables=list(variables) if variables is not None else [],
            min_depth=None,
            max_depth=None,
            extra_kwargs=extra or None,
        )

        out_dir = outfile.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        client.subset_one(
            bbox=bbox,
            output_filename=outfile.name,
            output_directory=str(out_dir),
            start_datetime=None,
            end_datetime=None,
        )
