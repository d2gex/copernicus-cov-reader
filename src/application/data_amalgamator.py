from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataAmalgamator:
    """
    Simple, pandas-based concatenation of CSVs in `csv/all/`.
    """

    def __init__(
        self,
        layout,
        *,
        output_name: str = "group.csv",
        logger: Callable[[str], None] = lambda _: None,
        read_csv: Callable[..., pd.DataFrame] = pd.read_csv,
    ) -> None:
        self.layout = layout
        self.output_name = output_name
        self.log = logger
        self._read_csv = read_csv

    def run(self, product, plan) -> List[Path]:
        """Concatenate CSVs in each `csv/all/` to a single CSV in `csv/group/`."""
        iter_fn = getattr(self.layout, "iter_csv_all_dirs", None)
        group_fn = getattr(self.layout, "csv_group_dir_from_all", None)
        if not callable(iter_fn) or not callable(group_fn):
            raise AttributeError(
                "ProjectLayout must expose 'iter_csv_all_dirs' and 'csv_group_dir_from_all'."
            )

        written: List[Path] = []
        for all_dir in iter_fn(product, plan):
            all_path = Path(all_dir)
            group_path = Path(group_fn(all_path))
            out = self._process_all_dir(all_path, group_path)
            if out is not None:
                written.append(out)
        return written

    # ---- Helpers ----

    def _process_all_dir(self, all_dir: Path, group_dir: Path) -> Optional[Path]:
        """Process one `csv/all/` directory and write its group CSV.

        Returns the written path, or None when the directory is missing or empty.
        """
        message: Optional[str] = None
        out_path: Optional[Path] = None

        try:
            csvs = self._list_csvs(all_dir)
        except FileNotFoundError:
            message = f"[amalgamation] skip (missing dir): {all_dir}"
        else:
            if not csvs:
                message = f"[amalgamation] skip (no CSVs): {all_dir}"

        if message is None:
            df_all = self._concat_csvs(csvs)
            group_dir.mkdir(parents=True, exist_ok=True)
            out_path = group_dir / self.output_name
            with out_path.open("w", encoding="utf-8", newline="") as fh:
                df_all.to_csv(fh, index=False)
            logger.info(
                f"[amalgamation] wrote: {out_path} "
                f"rows={len(df_all)} from={len(csvs)} files"
            )
        else:
            logger.warning(message)

        return out_path

    @staticmethod
    def _list_csvs(all_dir: Path) -> List[Path]:
        entries = [p for p in all_dir.iterdir() if p.is_file() and p.suffix == ".csv"]
        entries.sort(key=lambda p: p.name)
        return entries

    def _concat_csvs(self, files: List[Path]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for p in files:
            with p.open("r", encoding="utf-8", newline="") as fh:
                frames.append(self._read_csv(fh))
        return (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame()
        )
