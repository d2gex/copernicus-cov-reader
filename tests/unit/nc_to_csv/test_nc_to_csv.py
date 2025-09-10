import re
from pathlib import Path

import pandas as pd

from src.copernicus.nc_to_csv import NcToCsvConverter


def test_generate_writes_expected_count_and_files_exist(
    patched_converter: NcToCsvConverter,
    tmp_path: Path,
    twelve_nc_files: list[Path],
) -> None:
    out_dir = tmp_path / "out"
    written = patched_converter.generate_period_csvs(tmp_path, out_dir)

    assert len(written) == len(twelve_nc_files)
    for p in written:
        assert p.exists(), f"Missing CSV: {p}"


def test_generate_names_are_sorted_and_match_pattern(
    patched_converter: NcToCsvConverter,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out2"
    written = patched_converter.generate_period_csvs(tmp_path, out_dir)

    names = [p.name for p in written]
    assert names == sorted(names)

    slug = patched_converter._vars_slug
    pat = re.compile(rf"^\d{{4}}-\d{{2}}__{re.escape(slug)}\.csv$")
    assert all(pat.match(n) for n in names), f"Names must match YYYY-MM__{slug}.csv"


def test_generate_schema_and_lengths_for_all_csvs(
    patched_converter: NcToCsvConverter,
    tmp_path: Path,
    var_names: list[str],
    twelve_nc_files: list[Path],
) -> None:
    out_dir = tmp_path / "out3"
    written = patched_converter.generate_period_csvs(tmp_path, out_dir)

    # Expect one CSV per input .nc
    assert len(written) == len(twelve_nc_files) == 12

    required = {"time", "depth_idx", "tile_id", "tile_lon", "tile_lat", *var_names}

    for p in written:
        df = pd.read_csv(p)
        # schema
        missing = required.difference(df.columns)
        assert not missing, f"{p.name}: missing columns {missing}"
        # length (we fake 3 rows per period)
        assert len(df) == 3, f"{p.name}: expected 3 rows, got {len(df)}"
