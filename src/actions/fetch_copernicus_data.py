from __future__ import annotations

import pandas as pd

from src.app.download_and_convert import DownloadAndConvert
from src.config import cfg
from src.copernicus.cm_credentials import CMCredentials


def run(download_and_convert: int) -> None:
    CMCredentials().ensure_present()
    tiles_df = pd.read_csv(
        cfg.input_path
        / cfg.product_owner
        / cfg.product_slug
        / "test_tiles_with_date_db.csv"
    )
    dac = DownloadAndConvert(tiles_df)
    dac.run(download_and_convert)


if __name__ == "__main__":
    run(cfg.download_and_convert)
