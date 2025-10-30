from __future__ import annotations

from pathlib import Path


class ProjectLayout:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def product_root(self, product: str) -> Path:
        return self.root / product

    def bbox_root(self, product: str, bbox_id: str) -> Path:
        return self.product_root(product) / bbox_id

    def nc_tile_dir(self, product: str, bbox_id: str, tile_id_padded: str) -> Path:
        return self.bbox_root(product, bbox_id) / "nc" / tile_id_padded

    def nc_path(
        self, product: str, bbox_id: str, tile_id_padded: str, day_iso: str
    ) -> Path:
        return self.nc_tile_dir(product, bbox_id, tile_id_padded) / f"{day_iso}.nc"

    def csv_all_dir(self, product: str, bbox_id: str) -> Path:
        return self.bbox_root(product, bbox_id) / "csv" / "all"

    def csv_group_dir(self, product: str, bbox_id: str) -> Path:
        return self.bbox_root(product, bbox_id) / "csv" / "group"

    def ensure_product_bbox(self, product: str, bbox_id: str) -> None:
        (self.bbox_root(product, bbox_id) / "nc").mkdir(parents=True, exist_ok=True)
        self.csv_all_dir(product, bbox_id).mkdir(parents=True, exist_ok=True)
        self.csv_group_dir(product, bbox_id).mkdir(parents=True, exist_ok=True)

    def ensure_nc_tile_dir(
        self, product: str, bbox_id: str, tile_id_padded: str
    ) -> None:
        self.nc_tile_dir(product, bbox_id, tile_id_padded).mkdir(
            parents=True, exist_ok=True
        )

    @staticmethod
    def exists_nonempty(p: Path) -> bool:
        return p.exists() and p.stat().st_size > 0
