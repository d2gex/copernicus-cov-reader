import numpy as np
import pandas as pd
import pytest

from src.bounding_box.bounding_box import BoundingBox
from src.bounding_box.lat_bb_splitter import LatBandSplitter


def test_split_none_inside_returns_all_empty(
    coords20_df: pd.DataFrame, sample_bboxes: dict[str, BoundingBox]
) -> None:
    splitter = LatBandSplitter(sample_bboxes["none"], n_bands=3)
    bands = splitter.split(coords20_df)
    assert bands is None


def test_split_all_inside_but_some_bands_empty(
    coords20_df: pd.DataFrame, sample_bboxes: dict[str, BoundingBox]
) -> None:
    bbox = sample_bboxes["all_inside_wide"]
    splitter = LatBandSplitter(bbox, n_bands=4)  # edges: [0,10,20,30,40]
    bands = splitter.split(coords20_df)

    # All 20 points are inside this wide bbox.
    assert sum(b.num_points for b in bands) == len(coords20_df)

    # At least one band should be empty (the topmost [30,40]).
    assert all([band.num_points > 0 for band in bands[:-1]])
    assert bands[-1].num_points == 0  # last band is empty

    # No duplicates across bands.
    all_rows = pd.concat(
        [b.coords_within_band_df[["lon", "lat"]] for b in bands], ignore_index=True
    )
    assert len(all_rows) == len(all_rows.drop_duplicates())  # no dup across bands
    assert len(all_rows) == len(coords20_df)


def test_split_all_inside_all_bands_have_points(
    coords20_df: pd.DataFrame, sample_bboxes: dict[str, BoundingBox]
) -> None:
    bbox = sample_bboxes["all_inside_tight"]
    splitter = LatBandSplitter(bbox, n_bands=6)
    bands = splitter.split(coords20_df)

    # All 20 points are inside this bbox.
    assert sum(b.num_points for b in bands) == len(coords20_df)

    # Expect each band to contain at least one point for these edges.
    assert all(b.num_points > 0 for b in bands)

    # No duplicates across bands.
    all_rows = pd.concat(
        [b.coords_within_band_df[["lon", "lat"]] for b in bands], ignore_index=True
    )
    assert len(all_rows) == len(all_rows.drop_duplicates())  # no dup across bands
    assert len(all_rows) == len(coords20_df)


@pytest.mark.parametrize("lat_value", [15.0, 20.0])
def test_boundary_latitudes_go_to_upper_or_last_band(
    coords20_df, sample_bboxes, lat_value
):
    bbox = sample_bboxes["interior_line_10_20"]  # edges at 10, 15, 20 with n_bands=2
    splitter = LatBandSplitter(bbox, n_bands=2)
    bands = splitter.split(coords20_df)
    assert bands is not None and len(bands) == 2

    band0, band1 = bands[0], bands[1]  # [10,15) and [15,20]
    lat0 = band0.coords_within_band_df["lat"].to_numpy()
    lat1 = band1.coords_within_band_df["lat"].to_numpy()

    # Boundary latitude must not be in the lower band, but must be in the upper/last band.
    assert not np.isclose(lat0, lat_value).any()
    assert np.isclose(lat1, lat_value).any()


def test_mixed_inside_outside_filters_correctly(
    coords20_df: pd.DataFrame, sample_bboxes: dict[str, BoundingBox]
) -> None:
    bbox = sample_bboxes["mixed_lon_lat"]
    splitter = LatBandSplitter(bbox, n_bands=3)
    bands = splitter.split(coords20_df)

    # Expected inside set from fixture and bbox.
    inside = (
        (coords20_df["lon"] >= bbox.min_lon)
        & (coords20_df["lon"] <= bbox.max_lon)
        & (coords20_df["lat"] >= bbox.min_lat)
        & (coords20_df["lat"] <= bbox.max_lat)
    )

    outside = (
        (coords20_df["lat"] < bbox.min_lat)
        | (coords20_df["lat"] > bbox.max_lat)
        | (coords20_df["lon"] < bbox.min_lon)
        | (coords20_df["lon"] > bbox.max_lon)
    )
    assert inside.any()
    assert outside.any()
    assert sum(band.num_points for band in bands) < len(coords20_df)
