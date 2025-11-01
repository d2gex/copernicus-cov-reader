from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from src import utils


class CSVAmalgamation:
    def __init__(
        self, product_root: Path, output_name: str = "all_csv_rows.csv"
    ) -> None:
        self.product_root = product_root
        self.output_path = product_root / output_name

    def _list_csv_files(self) -> List[Path]:
        # bbox dirs = immediate subdirectories of the product root
        bbox_dirs = [d for d in self.product_root.iterdir() if d.is_dir()]
        csv_all_dirs = [d / "csv" / "all" for d in bbox_dirs]
        csv_all_dirs = [p for p in csv_all_dirs if p.is_dir()]

        files: List[Path] = []
        for csv_dir in csv_all_dirs:
            files.extend(p for p in csv_dir.glob("*.csv") if p.is_file())

        files.sort(key=lambda p: p.as_posix())
        return files

    @utils.timed("csv amalgamation")
    def run(self) -> None:
        csv_files = self._list_csv_files()
        if not csv_files:
            logging.info(
                "csv amalgamation: no CSV files found under %s", self.product_root
            )
            return

        frames = [pd.read_csv(p) for p in csv_files]
        merged = pd.concat(frames, ignore_index=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(self.output_path, index=False)
        logging.info(
            "csv amalgamation: wrote %s (%d rows from %d files)",
            self.output_path,
            len(merged),
            len(csv_files),
        )
