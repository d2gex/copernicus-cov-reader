from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError


class StorageSink(Protocol):
    def exists(self, relative_key: str) -> bool: ...
    def ingest(self, local_path: Path, relative_key: str) -> None: ...
    def cleanup_local(self, local_path: Path) -> None: ...


class LocalStorageSink:
    def exists(self, relative_key: str) -> bool:
        return False  # no remote state to check

    def ingest(self, local_path: Path, relative_key: str) -> None:
        return None  # already on disk where it belongs in local mode

    def cleanup_local(self, local_path: Path) -> None:
        return None  # cleanup is controlled by the caller/main


@dataclass(frozen=True)
class S3Config:
    bucket: str
    prefix: str = ""

    def key(self, relative_key: str) -> str:
        base = self.prefix.rstrip("/")
        rel = relative_key.lstrip("/")
        return f"{base}/{rel}" if base else rel


class S3StorageSink:
    def __init__(self, s3: BaseClient, cfg: S3Config) -> None:
        self._s3 = s3
        self._cfg = cfg

    def exists(self, relative_key: str) -> bool:
        key = self._cfg.key(relative_key)
        try:
            self._s3.head_object(Bucket=self._cfg.bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in {"404", "NotFound", "NoSuchKey"}:
                return False
            raise

    def ingest(self, local_path: Path, relative_key: str) -> None:
        key = self._cfg.key(relative_key)
        content_type, _ = mimetypes.guess_type(local_path.name)
        extra_args = {"ContentType": content_type} if content_type else {}
        self._s3.upload_file(
            Filename=str(local_path),
            Bucket=self._cfg.bucket,
            Key=key,
            ExtraArgs=extra_args,
        )

    def cleanup_local(self, local_path: Path) -> None:
        return None  # cleanup is driven by main; nc files are needed by CSV phase


def default_s3_client() -> BaseClient:
    return boto3.client("s3")
