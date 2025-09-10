from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.copernicus.dataset_tile_frame_extractor import DatasetTileFrameExtractor
from src.copernicus.nc_to_csv import NcToCsvConverter
from src.copernicus.tile_catalog import TileCatalog


@pytest.fixture
def var_names() -> list[str]:
    return ["thetao", "so"]


@pytest.fixture
def converter(var_names: list[str]) -> NcToCsvConverter:
    return NcToCsvConverter(var_names=var_names)


@pytest.fixture
def twelve_nc_files(tmp_path: Path) -> list[Path]:
    """Create 12 empty .nc files with names like sst_2020-01-01_2020-01-31.nc."""
    names = [
        f"sst_2020-{m:02d}-01_2020-{m:02d}-28.nc"
        if m == 2
        else f"sst_2020-{m:02d}-01_2020-{m:02d}-30.nc"
        if m in (4, 6, 9, 11)
        else f"sst_2020-{m:02d}-01_2020-{m:02d}-31.nc"
        for m in range(1, 13)
    ]
    paths: list[Path] = []
    for name in names:
        p = tmp_path / name
        p.touch()
        paths.append(p)
    return paths


@pytest.fixture
def patched_converter(
    tmp_path: Path, var_names: list[str], twelve_nc_files: list[Path]
) -> NcToCsvConverter:
    converter = NcToCsvConverter(var_names=var_names)
    # Tiny fake frame; copy per call to avoid accidental mutation sharing
    base = {
        "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "depth_idx": [0, 1, 2],
        "tile_id": [10, 11, 12],
        "tile_lon": [0.1, 0.2, 0.3],
        "tile_lat": [40.1, 40.2, 40.3],
    }
    for v in var_names:
        base[v] = [1.0, 2.0, 3.0]
    fake_df = pd.DataFrame(base)

    with (
        patch.object(NcToCsvConverter, "_list_files", return_value=twelve_nc_files),
        patch.object(NcToCsvConverter, "_nc_read", return_value=object()),
        patch.object(TileCatalog, "from_dataset", return_value=object()),
        patch.object(
            DatasetTileFrameExtractor,
            "to_frame_multi",
            side_effect=lambda *_args, **_kw: fake_df.copy(),
        ),
    ):
        yield converter  # patches auto-tear down after the test
