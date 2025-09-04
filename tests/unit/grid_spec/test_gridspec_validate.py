import pytest
import xarray as xr

from src.copernicus.grid_spec import GridSpec


@pytest.mark.parametrize(
    "ds_test, expected_to_fail",
    [
        ("ds_same", False),
        ("ds_other_same_grid", False),
        ("ds_modified", True),
    ],
)
def test_validate_hash(
    grid_spec: GridSpec,
    request: pytest.FixtureRequest,
    ds_test: str,
    expected_to_fail: bool,
) -> None:
    """GridSpec.validate passes on identical grids and fails on any hash change."""
    ds: xr.Dataset = request.getfixturevalue(ds_test)
    if expected_to_fail:
        with pytest.raises(ValueError, match="hash mismatch"):
            grid_spec.validate(ds)
    else:
        grid_spec.validate(ds)
