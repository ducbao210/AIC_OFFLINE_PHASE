"""Cấu hình dùng chung cho toàn bộ pipeline (đọc từ .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    # --- storage ---
    storage_backend: str            # "minio" | "local"
    storage_local_root: Path        # used when backend == "local"
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    bucket_raw: str
    bucket_frames: str
    bucket_feats: str
    # --- processing ---
    workers: int
    object_min_score: float
    # --- ASR (Whisper) ---
    asr_model: str                  # e.g. "small", "medium", "large-v3"
    asr_language: str               # "vi" for Vietnamese
    asr_device: str                 # "cuda" | "cpu"
    asr_batch_size: int
    # --- captioning (BLIP-2) ---
    caption_model: str              # e.g. "Salesforce/blip2-opt-2.7b"
    caption_device: str             # "cuda" | "cpu"
    caption_max_new_tokens: int

    @property
    def buckets(self) -> tuple[str, str, str]:
        return (self.bucket_raw, self.bucket_frames, self.bucket_feats)


def load_config() -> Config:
    sqlite_path = Path(
        os.environ.get("AIC_SQLITE_PATH", PIPELINE_DIR / "data" / "aic.sqlite")
    ).expanduser()
    if not sqlite_path.is_absolute():
        sqlite_path = (PIPELINE_DIR / sqlite_path).resolve()

    storage_backend = os.environ.get("AIC_STORAGE_BACKEND", "minio").strip().lower()
    storage_local_root = Path(
        os.environ.get("AIC_STORAGE_LOCAL_ROOT", PIPELINE_DIR / "data" / "storage")
    ).expanduser()

    return Config(
        data_root=Path(os.environ.get("AIC_DATA_ROOT", "/data/AIC_26_DATA")).expanduser(),
        sqlite_path=sqlite_path,
        storage_backend=storage_backend,
        storage_local_root=storage_local_root,
        endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=_bool(os.environ.get("MINIO_SECURE"), False),
        bucket_raw=os.environ.get("MINIO_BUCKET_RAW", "aic-raw"),
        bucket_frames=os.environ.get("MINIO_BUCKET_FRAMES", "aic-frames"),
        bucket_feats=os.environ.get("MINIO_BUCKET_FEATS", "aic-feats"),
        workers=int(os.environ.get("AIC_WORKERS", "8")),
        object_min_score=float(os.environ.get("AIC_OBJECT_MIN_SCORE", "0.25")),
        asr_model=os.environ.get("AIC_ASR_MODEL", "small"),
        asr_language=os.environ.get("AIC_ASR_LANGUAGE", "vi"),
        asr_device=os.environ.get("AIC_ASR_DEVICE", "cuda"),
        asr_batch_size=int(os.environ.get("AIC_ASR_BATCH_SIZE", "16")),
        caption_model=os.environ.get("AIC_CAPTION_MODEL", "Salesforce/blip2-opt-2.7b"),
        caption_device=os.environ.get("AIC_CAPTION_DEVICE", "cuda"),
        caption_max_new_tokens=int(os.environ.get("AIC_CAPTION_MAX_NEW_TOKENS", "64")),
    )


CONFIG = load_config()