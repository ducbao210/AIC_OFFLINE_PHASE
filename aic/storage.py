"""Storage abstraction: MinIO/S3 or local filesystem (Google Drive / Colab compatible).

Backend selection via AIC_STORAGE_BACKEND env var:
  - "minio" (default) — MinIO/S3 via the minio library
  - "local"            — copy/link to a local directory (e.g. Google Drive mount)

Key layout preserved regardless of backend:
  aic-raw/     videos/{video_id}.mp4
  aic-frames/  keyframes/{video_id}/{name}.jpg
  aic-feats/   clip-features-32/{video_id}.npy
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

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


# ------------------------------------------------------------------ backend detection
def _backend() -> str:
    import os
    return os.environ.get("AIC_STORAGE_BACKEND", "minio").strip().lower()


# ------------------------------------------------------------------ MinIO backend
def _get_minio_client(cfg: Config = CONFIG):
    try:
        from minio import Minio
    except ImportError:
        raise RuntimeError(
            "MinIO library not installed. Run: pip install minio  "
            "or set AIC_STORAGE_BACKEND=local"
        )
    return Minio(
        cfg.endpoint,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
    )


def _minio_ensure_buckets(client, buckets: tuple[str, ...]) -> None:
    for bucket in buckets:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            LOG.info("created bucket %s", bucket)


def _minio_object_exists(client, bucket: str, key: str, size: int | None = None) -> bool:
    from minio.error import S3Error
    try:
        stat = client.stat_object(bucket, key)
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NotFound"}:
            return False
        raise
    return size is None or stat.size == size


def _minio_upload(client, bucket: str, key: str, path: Path, skip_existing: bool = True) -> tuple[str, bool]:
    size = path.stat().st_size
    if skip_existing and _minio_object_exists(client, bucket, key, size):
        return key, False
    client.fput_object(
        bucket, key, str(path),
        content_type=CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"),
    )
    return key, True


# ------------------------------------------------------------------ Local FS backend
class _LocalClient:
    """Mimics enough of the Minio client API for local filesystem."""
    def __init__(self, root: Path):
        self.root = Path(root)


def _get_local_client(cfg: Config = CONFIG):
    import os
    root = os.environ.get("AIC_STORAGE_LOCAL_ROOT", str(cfg.data_root / "storage"))
    return _LocalClient(Path(root))


def _local_ensure_buckets(client: _LocalClient, buckets: tuple[str, ...]) -> None:
    for bucket in buckets:
        (client.root / bucket).mkdir(parents=True, exist_ok=True)


def _local_object_exists(client: _LocalClient, bucket: str, key: str, size: int | None = None) -> bool:
    p = client.root / bucket / key
    if not p.is_file():
        return False
    return size is None or p.stat().st_size == size


def _local_upload(client: _LocalClient, bucket: str, key: str, path: Path, skip_existing: bool = True) -> tuple[str, bool]:
    size = path.stat().st_size
    dest = client.root / bucket / key
    if skip_existing and _local_object_exists(client, bucket, key, size):
        return key, False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return key, True


# ------------------------------------------------------------------ public API
_client_cache = None


def get_client(cfg: Config = CONFIG):
    """Return a storage client (Minio or _LocalClient). Cached per process."""
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    backend = _backend()
    if backend == "local":
        _client_cache = _get_local_client(cfg)
    else:
        _client_cache = _get_minio_client(cfg)
    return _client_cache


def ensure_buckets(client=None, buckets: tuple[str, ...] = CONFIG.buckets) -> None:
    if client is None:
        client = get_client()
    backend = _backend()
    if backend == "local":
        _local_ensure_buckets(client, buckets)
    else:
        _minio_ensure_buckets(client, buckets)


def object_exists(client, bucket: str, key: str, size: int | None = None) -> bool:
    backend = _backend()
    if backend == "local":
        return _local_object_exists(client, bucket, key, size)
    return _minio_object_exists(client, bucket, key, size)


def upload_file(
    client,
    bucket: str,
    key: str,
    path: Path,
    *,
    skip_existing: bool = True,
    metadata: dict[str, str] | None = None,  # ignored for local backend
) -> tuple[str, bool]:
    """Return (key, uploaded?). Skip if object already exists with correct size."""
    backend = _backend()
    if backend == "local":
        return _local_upload(client, bucket, key, path, skip_existing)
    return _minio_upload(client, bucket, key, path, skip_existing)


# ------------------------------------------------------------------ key helpers
def video_key(video_id: str) -> str:
    return f"videos/{video_id}.mp4"


def keyframe_key(video_id: str, file_name: str) -> str:
    return f"keyframes/{video_id}/{file_name}"


def feature_key(video_id: str) -> str:
    return f"clip-features-32/{video_id}.npy"