import pandas as pd
import pytest

from src.bounding_box.bounding_box import BoundingBox


@pytest.fixture
def coords20_df() -> pd.DataFrame:
    # 20 deterministic points; some inside [lon≈-2..+3, lat≈9.5..21], some on band edges.
    lons = [
        -2.0,
        -1.8,
        -1.6,
        -1.4,
        -1.2,
        -1.0,
        -0.8,
        -0.6,
        -0.4,
        -0.2,
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6,
        3.0,
    ]
    lats = [
        9.5,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
        20.5,
        12.5,
        17.5,
        14.5,
        10.5,
        19.5,
        21.0,
    ]
    df = pd.DataFrame({"lon": lons, "lat": lats})
    return df


@pytest.fixture
def sample_bboxes() -> dict[str, BoundingBox]:
    # Multiple bboxes to exercise different behaviors.
    return {
        # a) none of the points fall inside
        "none": BoundingBox(min_lon=50.0, max_lon=60.0, min_lat=50.0, max_lat=60.0),
        # b) all points are inside, but some bands will be empty by design (lat span is wide)
        #    e.g., with n_bands=4 over [0, 40], the top band [30, 40] is empty.
        "all_inside_wide": BoundingBox(
            min_lon=-10.0, max_lon=10.0, min_lat=0.0, max_lat=40.0
        ),
        # c) all points inside and likely no empty bands with reasonable n_bands
        "all_inside_tight": BoundingBox(
            min_lon=-3.0, max_lon=4.0, min_lat=9.0, max_lat=21.0
        ),
        # d) interior-line case with edges at 10, 15, 20 when n_bands=2
        "interior_line_10_20": BoundingBox(
            min_lon=-3.0, max_lon=3.0, min_lat=10.0, max_lat=20.0
        ),
        # e) mixed: only a subset inside (both lon and lat filtering)
        "mixed_lon_lat": BoundingBox(
            min_lon=-1.0, max_lon=1.0, min_lat=10.0, max_lat=20.0
        ),
    }
