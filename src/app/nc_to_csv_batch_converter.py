from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from src.app.layout import ProjectLayout
from src.app.nc_to_csv_converter import NCTileToCSVConverter


class NCTileToCSVBatchConverter:
    def __init__(
        self, *, output_root: Path, product_slug: str, variables: Sequence[str]
    ) -> None:
        self.output_root = Path(output_root)
        self.product_slug = product_slug
        self.variables = tuple(variables)
        self.layout = ProjectLayout(root=self.output_root)
        self.converter = NCTileToCSVConverter(variables=self.variables)

    def run(self) -> None:
        product_root = self.layout.product_root(self.product_slug)
        if not product_root.exists():
            return

        bbox_dirs = [
            d for d in product_root.iterdir() if d.is_dir() and (d / "nc").is_dir()
        ]
        for bbox_dir in bbox_dirs:
            bbox_id = bbox_dir.name
            self.layout.ensure_product_bbox(self.product_slug, bbox_id)

            jobs = self._build_jobs_for_bbox(bbox_dir=bbox_dir, bbox_id=bbox_id)
            if jobs:
                self._write_jobs(jobs)

    def _build_jobs_for_bbox(
        self, *, bbox_dir: Path, bbox_id: str
    ) -> List[Tuple[Path, list[Path]]]:
        nc_root = bbox_dir / "nc"
        tile_dirs = [t for t in sorted(nc_root.iterdir()) if t.is_dir()]

        jobs: List[Tuple[Path, list[Path]]] = []
        for tile_dir in tile_dirs:
            tile_id = tile_dir.name  # padded, e.g., "0038"
            out_csv = (
                self.layout.csv_all_dir(self.product_slug, bbox_id) / f"{tile_id}.csv"
            )
            nc_files = sorted(
                p for p in tile_dir.iterdir() if p.suffix.lower() == ".nc"
            )

            needs_work = (not ProjectLayout.exists_nonempty(out_csv)) and bool(nc_files)
            if needs_work:
                jobs.append((out_csv, nc_files))
        return jobs

    def _write_jobs(self, jobs: List[Tuple[Path, list[Path]]]) -> None:
        for out_csv, nc_files in jobs:
            df = self.converter.run(nc_files)
            if not df.empty:
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_csv, index=False)
