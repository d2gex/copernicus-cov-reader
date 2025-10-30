from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import boto3

from .artifact_enumerator import enumerate_artifacts
from .bulk_sinker import Policy, bulk_sink
from .progress import ProgressReporter
from .storage_sink import S3Config, S3StorageSink

# ----------------------------
# Config (single source of truth, env-only)
# ----------------------------


@dataclass(frozen=True)
class AwsMainConfig:
    # Local roots
    nc_root: Path
    csv_root: Path

    # S3
    s3_bucket: str
    s3_prefix: str

    # Behavior
    clean: bool
    policy: Policy
    verbose: bool
    progress_step: float

    @classmethod
    def load_from_env(cls) -> "AwsMainConfig":
        bucket = os.getenv("S3_BUCKET")
        if not bucket:
            raise ValueError("S3_BUCKET is required")

        nc_root = os.getenv("AWS_NC_ROOT")
        csv_root = os.getenv("AWS_CSV_ROOT")
        if not nc_root or not csv_root:
            raise ValueError("AWS_NC_ROOT and AWS_CSV_ROOT are required")

        prefix = os.getenv("S3_OUTPUT_PREFIX", "")

        clean = os.getenv("AWS_CLEAN", "0").lower() in {"1", "true", "yes"}
        verbose = os.getenv("AWS_VERBOSE", "0").lower() in {"1", "true", "yes"}

        policy_raw = os.getenv("AWS_POLICY", Policy.SKIP_IF_EXISTS.value)
        try:
            policy = Policy(policy_raw)
        except ValueError as e:
            raise ValueError(
                "AWS_POLICY must be one of: skip_if_exists, always_put"
            ) from e

        step_raw = os.getenv("AWS_PROGRESS_STEP", "0.10")
        try:
            step = float(step_raw)
            if not (0 < step <= 1):
                raise ValueError
        except ValueError as e:
            raise ValueError("AWS_PROGRESS_STEP must be a float in (0, 1]") from e

        return cls(
            nc_root=Path(nc_root),
            csv_root=Path(csv_root),
            s3_bucket=bucket,
            s3_prefix=prefix,
            clean=clean,
            policy=policy,
            verbose=verbose,
            progress_step=step,
        )


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


# ----------------------------
# Public entrypoint (no args)
# ----------------------------


def run() -> None:
    cfg = AwsMainConfig.load_from_env()

    _configure_logging(cfg.verbose)

    _assert_dir(cfg.nc_root)
    _assert_dir(cfg.csv_root)

    if cfg.clean:
        logging.info("Cleaning local roots...")
        _clean_roots(cfg.nc_root, cfg.csv_root)

    _move_tree(
        "NC sinking",
        cfg.nc_root,
        cfg.policy,
        cfg.s3_bucket,
        cfg.s3_prefix,
        cfg.progress_step,
    )
    _move_tree(
        "CSV sinking",
        cfg.csv_root,
        cfg.policy,
        cfg.s3_bucket,
        cfg.s3_prefix,
        cfg.progress_step,
    )
