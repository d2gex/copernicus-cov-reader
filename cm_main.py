from datetime import date, timedelta

import copernicusmarine as cm

from src import config
from src.copernicus.cm_credentials import CMCredentials
from src.copernicus.cm_subset_client import CMSubsetClient


def month_periods(year: int) -> list[tuple[str, str]]:
    """Return (start, end) date strings for each month in the given year."""
    periods = []
    for month in range(1, 13):
        start = date(year, month, 1)
        if month == 12:
            end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(year, month + 1, 1) - timedelta(days=1)
        periods.append((start.isoformat(), end.isoformat()))
    return periods


def main():
    dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"  # example SST dataset
    variables = ["thetao"]  # SST variable name in Copernicus
    min_depth, max_depth = 0, 1

    # Example: one-off coarse bounding box for Galicia
    bboxes = [(-10.0, -7.0, 41.8, 44.0)]

    output_dir = config.OUTPUT_PATH / "sst_coarse_bb"
    year = 2020

    creds = CMCredentials()
    creds.ensure_present()

    client = CMSubsetClient(
        cm,
        dataset_id=dataset_id,
        variables=variables,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    # Download monthly subsets for all bounding boxes
    client.subset_many(
        bboxes=bboxes,
        periods=month_periods(year),
        output_directory=output_dir,
        filename_fn=lambda bbox, start, end: f"sst_{start}_{end}.nc",
    )


if __name__ == "__main__":
    main()
