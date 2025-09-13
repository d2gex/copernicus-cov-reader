from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.application.app_ds import ProductSpec, StaticSpec
from src.application.plan_builder import BBoxPlan, Plan
from src.application.project_layout import ProjectLayout
from src.copernicus.cm_credentials import CMCredentials
from src.copernicus.cm_subset_client import CMSubsetClient
from src.data_processing.time_block_splitter import TimeBlock as TBBlock


class CMDownloader:
    """
    Downloads optional static .nc (one per bbox) and time-sliced .nc files.
    Static files are saved under: {product}/{bbox_00X}/static/{product}__{bbox_00X}__static.nc
    """

    def __init__(
        self,
        layout: ProjectLayout,
        credentials: CMCredentials,
        *,
        skip_existing: bool = True,
    ) -> None:
        self.layout = layout
        self.credentials = credentials
        self.skip_existing = skip_existing

    def _nc_filename(self, product: ProductSpec, tb: TBBlock, bp: BBoxPlan) -> str:
        start_str = str(tb.start.date())
        last_day = (tb.end - timedelta(days=1)).date()
        end_str = str(last_day)
        return f"{product.slug()}__{bp.folder_name()}__{start_str}_to_{end_str}.nc"

    def get_static_details(
        self,
        plan: Plan,
        product: ProductSpec,
        static: Optional[StaticSpec],
    ) -> Dict[str, Path]:
        """
        Download one static .nc per bbox (if static spec provided).
        Store under bbox/static/, return paths keyed by bbox folder.
        NOTE: no mask extraction here; conversion does that later.
        """
        paths_by_bbox: Dict[str, Path] = {}
        if static is None:
            return paths_by_bbox

        self.credentials.ensure_present()
        client = CMSubsetClient()

        for bp in plan.bboxes:
            # Ensure bbox tree exists and static dir too
            self.layout.ensure_product_dirs(product, bp)
            static_dir = self.layout.static_dir(product, bp)
            static_dir.mkdir(parents=True, exist_ok=True)
            target = self.layout.static_path(product, bp)

            should_download = True
            if self.skip_existing and target.exists():
                should_download = False

            if should_download:
                client.subset_one(
                    dataset_id=static.dataset_id,
                    variables=[static.mask_var],
                    bbox=(
                        bp.bbox.min_lon,
                        bp.bbox.max_lon,
                        bp.bbox.min_lat,
                        bp.bbox.max_lat,
                    ),
                    output_filename=str(target),
                )
            else:
                # file exists; no-op
                pass

            paths_by_bbox[bp.folder_name()] = target

        return paths_by_bbox

    def get_data(self, plan: Plan, products: Iterable[ProductSpec]) -> List[Path]:
        """
        Download period .nc files into bbox/nc for each product.
        """
        self.credentials.ensure_present()
        client = CMSubsetClient()
        outputs: List[Path] = []

        for bp in plan.bboxes:
            for product in products:
                nc_dir, _ = self.layout.ensure_product_dirs(product, bp)

                for tb in plan.periods:
                    target = nc_dir / self._nc_filename(product, tb, bp)
                    should_download = True
                    if self.skip_existing and target.exists():
                        should_download = False

                    if should_download:
                        client.subset_one(
                            dataset_id=product.dataset_id,
                            variables=list(product.variables),
                            bbox=(
                                bp.bbox.min_lon,
                                bp.bbox.max_lon,
                                bp.bbox.min_lat,
                                bp.bbox.max_lat,
                            ),
                            start_datetime=tb.start,
                            end_datetime=tb.end,
                            output_filename=str(target),
                        )
                        outputs.append(target)
                    else:
                        # exists; no-op
                        pass

        return outputs
