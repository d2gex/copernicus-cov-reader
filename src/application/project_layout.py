from __future__ import annotations

from pathlib import Path
from typing import Tuple

from src.application.app_ds import ProductSpec
from src.application.plan_builder import BBoxPlan


class ProjectLayout:
    """
    Project tree (per product):

    {project_root}/
      data/
        {product_slug}/
          bbox_001/
            nc/
            csv/
              all/
            static/
              {product_slug}__bbox_001__static.nc
          bbox_002/
            ...
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)

    # ---- roots ----

    def data_root(self) -> Path:
        return self.project_root / "data"

    def product_root(self, product: ProductSpec) -> Path:
        return self.data_root() / product.slug()

    def bbox_dir(self, product: ProductSpec, plan: BBoxPlan) -> Path:
        return self.product_root(product) / plan.folder_name()

    # ---- leaves ----

    def nc_dir(self, product: ProductSpec, plan: BBoxPlan) -> Path:
        return self.bbox_dir(product, plan) / "nc"

    def csv_all_dir(self, product: ProductSpec, plan: BBoxPlan) -> Path:
        return self.bbox_dir(product, plan) / "csv" / "all"

    def static_dir(self, product: ProductSpec, plan: BBoxPlan) -> Path:
        return self.bbox_dir(product, plan) / "static"

    def static_path(self, product: ProductSpec, plan: BBoxPlan) -> Path:
        fname = f"{product.slug()}__{plan.folder_name()}__static.nc"
        return self.static_dir(product, plan) / fname

    # ---- creation ----

    def ensure_product_dirs(
        self, product: ProductSpec, plan: BBoxPlan
    ) -> Tuple[Path, Path]:
        """
        Ensure existence of product root, bbox root, nc, and csv/all.
        (static/ is created by the downloader when needed.)
        """
        root = self.product_root(product)
        root.mkdir(parents=True, exist_ok=True)

        bbox_root = self.bbox_dir(product, plan)
        bbox_root.mkdir(parents=True, exist_ok=True)

        nc = self.nc_dir(product, plan)
        csv_all = self.csv_all_dir(product, plan)

        nc.mkdir(parents=True, exist_ok=True)
        csv_all.mkdir(parents=True, exist_ok=True)

        return nc, csv_all
