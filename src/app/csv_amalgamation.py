from __future__ import annotations

import logging
import re
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

    @staticmethod
    def _parse_tile_id_from_filename(path: Path) -> int:
        """
        Parse tile_id from filename.
        Examples:
          '00146.csv' -> 146
          'tile_00325.csv' -> 325
        Raises ValueError if no digits can be found.
        """
        stem = path.stem
        if stem.isdigit():
            return int(stem)
        m = re.search(r"(\d+)", stem)
        if not m:
            raise ValueError(f"Cannot infer tile_id from filename: {path.name}")
        return int(m.group(1))

    @utils.timed("csv amalgamation")
    def run(self) -> None:
        csv_files = self._list_csv_files()
        if not csv_files:
            logging.info(
                "csv amalgamation: no CSV files found under %s", self.product_root
            )
            return

        frames: List[pd.DataFrame] = []
        for p in csv_files:
            df = pd.read_csv(p)
            tile_id = self._parse_tile_id_from_filename(p)
            # Insert tile_id as the first column (no copy; mutate local df only)
            df.insert(0, "tile_id", tile_id)
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True, sort=False)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(self.output_path, index=False)
        logging.info(
            "csv amalgamation: wrote %s (%d rows from %d files)",
            self.output_path,
            len(merged),
            len(csv_files),
        )
