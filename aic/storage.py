"""MinIO/S3: tạo bucket, upload idempotent, presigned URL."""

from __future__ import annotations

import logging
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from .config import CONFIG, Config

LOG = logging.getLogger("aic.storage")

CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".npy": "application/octet-stream",
    ".json": "application/json",
    ".csv": "text/csv",
}


def get_client(cfg: Config = CONFIG) -> Minio:
    return Minio(
        cfg.endpoint,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
    )


def ensure_buckets(client: Minio, buckets: tuple[str, ...] = CONFIG.buckets) -> None:
    for bucket in buckets:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            LOG.info("created bucket %s", bucket)


def object_exists(client: Minio, bucket: str, key: str, size: int | None = None) -> bool:
    try:
        stat = client.stat_object(bucket, key)
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NotFound"}:
            return False
        raise
    return size is None or stat.size == size


def upload_file(
    client: Minio,
    bucket: str,
    key: str,
    path: Path,
    *,
    skip_existing: bool = True,
    metadata: dict[str, str] | None = None,
) -> tuple[str, bool]:
    """Trả về (key, uploaded?). Bỏ qua nếu object đã tồn tại và đúng kích thước."""
    size = path.stat().st_size
    if skip_existing and object_exists(client, bucket, key, size):
        return key, False

    client.fput_object(
        bucket,
        key,
        str(path),
        content_type=CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        metadata=metadata or None,
    )
    return key, True


def video_key(video_id: str) -> str:
    return f"videos/{video_id}.mp4"


def keyframe_key(video_id: str, file_name: str) -> str:
    return f"keyframes/{video_id}/{file_name}"


def feature_key(video_id: str) -> str:
    return f"clip-features-32/{video_id}.npy"
