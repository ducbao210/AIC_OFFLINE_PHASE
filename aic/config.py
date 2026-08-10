"""Cấu hình dùng chung cho toàn bộ pipeline (đọc từ .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # dotenv là tuỳ chọn
    pass

PIPELINE_DIR = Path(__file__).resolve().parents[1]


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    data_root: Path
    sqlite_path: Path
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    bucket_raw: str
    bucket_frames: str
    bucket_feats: str
    workers: int
    object_min_score: float

    @property
    def buckets(self) -> tuple[str, str, str]:
        return (self.bucket_raw, self.bucket_frames, self.bucket_feats)


def load_config() -> Config:
    sqlite_path = Path(
        os.environ.get("AIC_SQLITE_PATH", PIPELINE_DIR / "data" / "aic.sqlite")
    ).expanduser()
    if not sqlite_path.is_absolute():
        sqlite_path = (PIPELINE_DIR / sqlite_path).resolve()

    return Config(
        data_root=Path(os.environ.get("AIC_DATA_ROOT", "/data/AIC_26_DATA")).expanduser(),
        sqlite_path=sqlite_path,
        endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=_bool(os.environ.get("MINIO_SECURE"), False),
        bucket_raw=os.environ.get("MINIO_BUCKET_RAW", "aic-raw"),
        bucket_frames=os.environ.get("MINIO_BUCKET_FRAMES", "aic-frames"),
        bucket_feats=os.environ.get("MINIO_BUCKET_FEATS", "aic-feats"),
        workers=int(os.environ.get("AIC_WORKERS", "8")),
        object_min_score=float(os.environ.get("AIC_OBJECT_MIN_SCORE", "0.25")),
    )


CONFIG = load_config()
