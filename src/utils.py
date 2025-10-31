from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import boto3


@dataclass(frozen=True)
class ProductZipper:
    """
    Creates a .zip archive of a directory, preserving its internal structure.
    """

    def zip_dir(
        self, root: Path, *, out_dir: Path | None = None, name: str | None = None
    ) -> Path:
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Not a directory: {root}")

        out_dir = out_dir or root.parent
        name = name or root.name

        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / name  # shutil.make_archive will add .zip
        archive_path = shutil.make_archive(
            base_name=str(base),
            format="zip",
            root_dir=str(root.parent),
            base_dir=root.name,
        )
        return Path(archive_path)


def timed(label: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info("%s: start", label)
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                logging.info("%s: done in %.2fs", label, dt)

        return wrapper

    return decorator


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    for u in units:
        if s < 1024 or u == units[-1]:
            return f"{s:.2f}{u}"
        s /= 1024.0
    return f"{n}B"


def assert_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Not a directory: {path}")


def upload_file_to_s3(local_path: Path, bucket: str, key: str) -> None:
    s3 = boto3.client("s3")
    s3.upload_file(Filename=str(local_path), Bucket=bucket, Key=key)
