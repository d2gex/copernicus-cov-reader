from __future__ import annotations

import logging
import shutil
from pathlib import Path

import boto3

from src.aws.artifact_enumerator import enumerate_artifacts
from src.aws.bulk_sinker import Policy, bulk_sink
from src.aws.progress import ProgressReporter
from src.aws.storage_sink import S3Config, S3StorageSink
from src.config import cfg  # unified config object


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _assert_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Not a directory: {path}")


def _clean_roots(nc_root: Path, csv_root: Path) -> None:
    for p in (nc_root, csv_root):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)


def _move_tree(
    label: str, root: Path, policy: Policy, bucket: str, prefix: str, step: float
) -> None:
    artifacts = enumerate_artifacts(root)
    pr = ProgressReporter(total=len(artifacts), step=step)

    s3 = boto3.client("s3")
    sink = S3StorageSink(s3, S3Config(bucket=bucket, prefix=prefix))

    pr.start(label)
    summary = bulk_sink(artifacts, sink, policy, on_tick=lambda: pr.tick(label))
    pr.finish(
        label,
        summary=f"uploaded={summary.uploaded} skipped={summary.skipped} failed={summary.failed}",
    )


def run() -> None:
    _configure_logging(cfg.aws_verbose)

    _assert_dir(cfg.aws_nc_root)
    _assert_dir(cfg.aws_csv_root)

    if cfg.aws_clean:
        logging.info("Cleaning local roots...")
        _clean_roots(cfg.aws_nc_root, cfg.aws_csv_root)

    policy = Policy(cfg.aws_policy)
    step = cfg.aws_progress_step

    _move_tree(
        "NC sinking", cfg.aws_nc_root, policy, cfg.s3_bucket, cfg.s3_output_prefix, step
    )
    _move_tree(
        "CSV sinking",
        cfg.aws_csv_root,
        policy,
        cfg.s3_bucket,
        cfg.s3_output_prefix,
        step,
    )


if __name__ == "__main__":
    run()
