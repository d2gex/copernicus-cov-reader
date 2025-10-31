from __future__ import annotations

import logging
import shutil
from pathlib import Path

from src import utils
from src.config import cfg  # unified config object


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _s3_key_for_product_zip(product_slug: str) -> str:
    # Store as: <prefix>/<product_slug>.zip
    prefix = cfg.s3_output_prefix.rstrip("/")
    return f"{prefix}/{product_slug}.zip" if prefix else f"{product_slug}.zip"


@utils.timed("zip+upload product")
def _zip_and_upload_product(product_root: Path, product_slug: str) -> None:
    zipper = utils.ProductZipper()
    archive_path = zipper.zip_dir(
        product_root, out_dir=product_root.parent, name=product_slug
    )
    size = archive_path.stat().st_size
    logging.info("Archive created: %s (%s)", archive_path.name, utils.human_bytes(size))

    key = _s3_key_for_product_zip(product_slug)
    logging.info("Uploading to s3://%s/%s", cfg.s3_bucket, key)
    utils.upload_file_to_s3(archive_path, cfg.s3_bucket, key)
    logging.info("Upload complete")

    try:
        archive_path.unlink()
        logging.debug("Removed local archive %s", archive_path)
    except Exception:
        logging.warning("Could not remove local archive %s", archive_path)


def run() -> None:
    _configure_logging(cfg.aws_verbose)

    product_root = cfg.output_root / cfg.product_slug
    utils.assert_dir(product_root)

    # Optional clean-up (pre-run), if you keep this behavior:
    # It will remove the product folder entirely—use with care.
    # If you prefer to disable clean here, just remove the block.
    if cfg.aws_clean:
        logging.info("AWS_CLEAN is set; removing %s", product_root)
        shutil.rmtree(product_root, ignore_errors=True)
        product_root.mkdir(parents=True, exist_ok=True)

    _zip_and_upload_product(product_root, cfg.product_slug)


if __name__ == "__main__":
    run()
