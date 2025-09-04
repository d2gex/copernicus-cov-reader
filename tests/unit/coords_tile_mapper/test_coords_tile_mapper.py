import numpy as np
import pytest


@pytest.mark.parametrize(
    "case", ["top_left", "top_right", "bottom_left", "bottom_right", "center"]
)
def test_mapper_assigns_expected_tile_and_coords_per_case(
    case, df_five_points, mapped_4x4
):
    out, sea_lon, sea_lat = mapped_4x4

    got_id = int(out.loc[case, "tile_id"])
    exp_id = int(df_five_points.loc[case, "expected_tile_id"])
    assert got_id == exp_id, f"{case}: expected tile_id {exp_id}, got {got_id}"

    got_lon = float(sea_lon[got_id])
    got_lat = float(sea_lat[got_id])
    assert got_lon == float(
        df_five_points.loc[case, "expected_lon"]
    ), f"{case}: lon mismatch"
    assert got_lat == float(
        df_five_points.loc[case, "expected_lat"]
    ), f"{case}: lat mismatch"


def test_mapper_batch_shape_and_dtype(mapped_4x4, df_five_points):
    out, _, _ = mapped_4x4
    assert len(out) == len(df_five_points)
    assert out["tile_id"].dtype == np.int64
