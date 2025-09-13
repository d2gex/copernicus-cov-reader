# src/io/nc_to_csv_converter.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from src.data_processing.dataset_tile_frame_extractor import DatasetTileFrameExtractor
from src.data_processing.tile_catalog import TileCatalog


class NcToCsvConverter:
    def __init__(
        self,
        var_names: Sequence[str],
        bbox_id: int,
        time_dim: str = "time",
        depth_dim: Optional[str] = "depth",
        sea_land_mask: Optional[np.ndarray] = None,
    ) -> None:
        if not var_names:
            raise ValueError("var_names must be a non-empty sequence.")
        self._var_names: List[str] = list(var_names)
        self._vars_slug: str = self._build_vars_slug(self._var_names)
        self._bbox_id = int(bbox_id)
        self._time_dim = time_dim
        self._depth_dim = depth_dim
        self._sead_land_mask = sea_land_mask

    # ---------- PUBLIC ----------

    def generate_period_csvs(self, input_dir: Path, output_dir: Path) -> List[Path]:
        in_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        nc_paths = self._list_files(in_dir, suffix=".nc")
        if not nc_paths:
            return []

        written: List[Path] = []

        # Build catalog/extractor once; validate each subsequent dataset against the same grid.
        catalog = None
        extractor = None

        for nc_path in nc_paths:
            ds = self._nc_read(nc_path)

            if catalog is None:
                catalog = TileCatalog.from_dataset(ds, self._sead_land_mask)
                extractor = DatasetTileFrameExtractor(
                    catalog=catalog,
                    bbox_id=self._bbox_id,
                    time_dim=self._time_dim,
                    depth_dim=self._depth_dim,
                )
            else:
                catalog.grid.validate(ds)
                assert extractor is not None

            df = extractor.to_frame_multi(ds, var_names=self._var_names)
            if df is None or df.empty:
                raise ValueError(f"Extractor returned empty DataFrame for: {nc_path}")

            period_key = self._period_key_from_name(nc_path.name)  # e.g., "2020-03"
            out_path = out_dir / f"{period_key}__{self._vars_slug}.csv"
            self._csv_write(df, out_path)
            written.append(out_path)

        return written

    def join_csvs(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        files_per_block: Optional[int] = None,
    ) -> List[Path]:
        in_dir = Path(input_dir)
        if not in_dir.exists() or not in_dir.is_dir():
            raise ValueError(
                f"Input dir does not exist or is not a directory: {in_dir}"
            )

        csv_paths = self._sorted_by_name(self._list_files(in_dir, ".csv"))
        if not csv_paths:
            raise ValueError(f"No .csv files found in: {in_dir}")

        if files_per_block is not None and files_per_block <= 0:
            raise ValueError("files_per_block must be a positive integer when provided")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        outputs: List[Path] = []
        if files_per_block is None:
            outputs.append(
                self._concat_write(csv_paths, out_dir)
            )  # outputs is a list of paths
            return outputs

        for chunk_paths in self._chunk(csv_paths, files_per_block):
            outputs.append(
                self._concat_write(chunk_paths, out_dir)
            )  # outputs is a list of paths
        return outputs

    def _build_vars_slug(self, names: Sequence[str]) -> str:
        # Preserve input order; keep only [A-Za-z0-9_] and join with underscores.
        cleaned = [re.sub(r"[^A-Za-z0-9_]+", "", n) for n in names]
        cleaned = [c for c in cleaned if c]  # drop empties if any
        return "_".join(cleaned)

    def _csv_write(self, df: pd.DataFrame, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)

    def _csv_read(self, path: Path) -> pd.DataFrame:
        with path.open("r", encoding="utf-8") as f:
            return pd.read_csv(f)

    def _nc_read(self, path: Path) -> xr.Dataset:
        with xr.open_dataset(path) as ds:
            return ds.load()

    def _period_key_from_name(self, filename: str) -> str:
        # Accept "YYYY-MM", "YYYY_MM", "YYYYMM", or "YYYYMMDD" (use YYYY-MM for month).
        pats = (
            r"(?P<y>\d{4})[-_](?P<m>\d{2})",
            r"(?P<y>\d{4})(?P<m>\d{2})(?:\d{2})?",
        )
        for pat in pats:
            m = re.search(pat, filename)
            if m:
                return f"{m['y']}-{m['m']}"
        raise ValueError(f"Cannot derive period key from filename: {filename}")

    def _split_period_and_slug(self, csv_name: str) -> Tuple[str, str]:
        base = csv_name.rsplit(".", 1)[0]
        if "__" in base:
            first, rest = base.split("__", 1)
            return first, rest
        return base, "merged"

    def _concat_write(self, chunk_paths: Sequence[Path], out_dir: Path) -> Path:
        dfs = [self._csv_read(p) for p in chunk_paths]
        if not dfs:
            raise ValueError("Empty chunk encountered during join")

        cols0 = list(dfs[0].columns)
        for i, df in enumerate(dfs[1:], start=1):
            if list(df.columns) != cols0:
                a, b = chunk_paths[0].name, chunk_paths[i].name
                raise ValueError(f"Schema mismatch between CSVs: {a} vs {b}")

        out_df = pd.concat(dfs, ignore_index=True)

        start_key, slug1 = self._split_period_and_slug(chunk_paths[0].name)
        end_key, slug2 = self._split_period_and_slug(chunk_paths[-1].name)
        slug = slug1 or slug2
        out_name = f"{start_key}__to__{end_key}__{slug}.csv"
        out_path = out_dir / out_name

        self._csv_write(out_df, out_path)
        return out_path

    def _list_files(self, folder: Path, suffix_lower: str) -> List[Path]:
        return [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() == suffix_lower
        ]

    def _sorted_by_name(self, paths: List[Path]) -> List[Path]:
        return sorted(paths, key=lambda p: p.name)

    def _chunk(self, seq: Sequence[Path], n: int) -> List[Sequence[Path]]:
        return [seq[i : i + n] for i in range(0, len(seq), n)]
