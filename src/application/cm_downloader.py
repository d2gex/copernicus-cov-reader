from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import copernicusmarine as cm

from src.application.app_ds import ProductSpec, StaticSpec
from src.application.plan_builder import BBoxPlan, Plan
from src.application.project_layout import ProjectLayout
from src.copernicus.cm_subset_client import CMSubsetClient

logger = logging.getLogger(__name__)


class CMDownloader:
    """
    Download static and time-sliced data using the existing ProjectLayout helpers
    and the thin CMSubsetClient wrapper (no fallbacks).

    - get_static_details(): one static file per bbox into `static/`.
    - get_data(): one NetCDF per bbox×period into `nc/`.

    Assumptions (from application package):
      * ProjectLayout exposes: ensure_product_dirs(), static_dir(), static_path(), nc_dir().
      * Plan.bboxes: Iterable[BBoxPlan]; Plan.periods: Iterable[TBBlock] with .start/.end.
      * BBoxPlan has .bbox with min_lon/max_lon/min_lat/max_lat and .folder_name().
      * ProductSpec provides .slug(), .dataset_id, .variables.
    """

    def __init__(
        self,
        *,
        layout: ProjectLayout,
        skip_existing: bool = True,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        extra_subset_kwargs: Optional[dict] = None,
    ) -> None:
        self.layout = layout
        self.skip_existing = bool(skip_existing)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.extra_subset_kwargs = dict(extra_subset_kwargs or {})
        self._cm = None  # set lazily by _client()

    # ------------------------------
    # Public API
    # ------------------------------

    def get_static_details(
        self, plan: Plan, product: ProductSpec, static: Optional[StaticSpec]
    ) -> Dict[str, Path]:
        """Download one static file per bbox into `bbox/static/`.

        Returns a dict mapping bbox folder name -> written file path. If `static` is
        None, returns an empty dict.
        """
        paths_by_bbox: Dict[str, Path] = {}
        if static is None:
            return paths_by_bbox

        client = CMSubsetClient(
            cm=cm,
            dataset_id=static.dataset_id,
            variables=[static.mask_var],  # ensure a list, not a string
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            extra_kwargs=self.extra_subset_kwargs,
        )

        for bp in plan.bboxes:
            # Ensure tree exists (nc/ and csv/all/); we'll make static/ as needed.
            self.layout.ensure_product_dirs(product, bp)

            static_dir = self.layout.static_dir(product, bp)
            static_dir.mkdir(parents=True, exist_ok=True)
            target = self.layout.static_path(product, bp)

            if self.skip_existing and target.exists():
                logger.info("[static] skip existing: %s", target)
                paths_by_bbox[bp.folder_name()] = target
            else:
                bbox_t = (
                    float(bp.bbox.min_lon),
                    float(bp.bbox.max_lon),
                    float(bp.bbox.min_lat),
                    float(bp.bbox.max_lat),
                )
                client.subset_one(
                    bbox=bbox_t,
                    output_filename=target.name,
                    output_directory=static_dir,
                )
                paths_by_bbox[bp.folder_name()] = target

        return paths_by_bbox

    def get_data(self, plan: Plan, products: Sequence[ProductSpec]) -> List[Path]:
        """Download NetCDFs for each product × bbox × period into per-bbox `nc/`.

        Returns list of written (or skipped) target paths (only for newly written files
        when skip_existing=True).
        """
        written: List[Path] = []

        for product in products:
            client = CMSubsetClient(
                cm=cm,
                dataset_id=product.dataset_id,
                variables=list(product.variables),
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                extra_kwargs=self.extra_subset_kwargs,
            )

            for bp in plan.bboxes:
                nc_dir = self.layout.nc_dir(product, bp)
                nc_dir.mkdir(parents=True, exist_ok=True)

                bbox_t = (
                    float(bp.bbox.min_lon),
                    float(bp.bbox.max_lon),
                    float(bp.bbox.min_lat),
                    float(bp.bbox.max_lat),
                )

                for tb in plan.periods:
                    start = tb.start.isoformat().replace("+00:00", "Z")
                    end = tb.end.isoformat().replace("+00:00", "Z")

                    target = self._nc_target_path(product, bp, nc_dir, start, end)

                    if self.skip_existing and target.exists():
                        logger.info("[data] skip existing: %s", target)
                    else:
                        client.subset_one(
                            bbox=bbox_t,
                            output_filename=target.name,
                            output_directory=nc_dir,
                            start_datetime=start,  # e.g. "2018-01-01T00:00:00Z"
                            end_datetime=end,  # e.g. "2018-01-15T12:00:00Z"
                        )
                        written.append(target)

        return written

    def _iso_basic_utc(self, ts: str) -> str:
        """
        Return SMB-safe timestamp: YYYY-MM-DDTHHMMSSZ.
        Accepts:
          - 'YYYY-MM-DD' -> assumes midnight UTC
          - ISO8601 with or without 'Z'
          - With offsets (e.g., +01:00) -> converted to UTC
        """
        t = ts.strip()
        if re.compile(r"^\d{4}-\d{2}-\d{2}$").fullmatch(t):
            dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        else:
            # Normalize trailing Z so fromisoformat can parse it
            t = t.replace("Z", "+00:00")
            dt = datetime.fromisoformat(t)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H%M%SZ")

    def _span_slug(self, start: str, end: str) -> str:
        """START_END with colon-free timestamps."""
        return f"{self._iso_basic_utc(start)}_{self._iso_basic_utc(end)}"

    def _nc_target_path(
        self,
        product: ProductSpec,
        bp: BBoxPlan,
        nc_dir: Path,
        start: str,
        end: str,
    ) -> Path:
        """Build an nc filename consistent with the app's static naming style (SMB-safe)."""
        # static uses: f"{product.slug()}__{plan.folder_name()}__static.nc"
        # for periods: keep full precision but remove ':' (ISO basic-like)
        period = self._span_slug(start, end)
        fname = f"{product.slug()}__{bp.folder_name()}__{period}.nc"

        return nc_dir / fname
