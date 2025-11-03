import copernicusmarine as cm

from src.app.downloader import Downloader
from src.config import cfg

if __name__ == "__main__":
    # Ensure hauls are unique

    downloader = Downloader(cm_handle=cm)
    downloader.download_static(
        dataset_id=cfg.dataset_id,
        area=[
            cfg.region_min_lon,
            cfg.region_min_lat,
            cfg.region_max_lon,
            cfg.region_max_lat,
        ],
        outfile=cfg.input_path / cfg.product_owner / "static_data" / cfg.static_name,
    )
