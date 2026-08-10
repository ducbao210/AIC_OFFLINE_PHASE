"""SQLite: schema, kết nối, và manifest cho ingest idempotent."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import CONFIG

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Inventory thô của mọi file phát hiện được trên đĩa (stage 01)
CREATE TABLE IF NOT EXISTS assets (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,           -- video | keyframe_dir | map_csv | object_dir | media_json | clip_npy
    video_id    TEXT NOT NULL,
    rel_path    TEXT NOT NULL,
    size_bytes  INTEGER,
    mtime       REAL,
    UNIQUE (kind, video_id)
);

CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    group_id     TEXT NOT NULL,          -- L21, L22, ...
    rel_path     TEXT,
    s3_bucket    TEXT,
    s3_key       TEXT,
    sha256       TEXT,
    size_bytes   INTEGER,
    fps_num      INTEGER,
    fps_den      INTEGER,
    fps          REAL,
    n_frames     INTEGER,
    duration_s   REAL,
    width        INTEGER,
    height       INTEGER,
    codec        TEXT,
    audio_codec  TEXT,
    audio_sample_rate INTEGER,
    audio_channels    INTEGER,
    created_at   TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_videos_group ON videos(group_id);

-- 1 dòng = 1 keyframe của BTC (nguồn chân lý: map-keyframes/*.csv)
CREATE TABLE IF NOT EXISTS keyframes (
    id          INTEGER PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    n           INTEGER NOT NULL,        -- thứ tự trong CSV (1-based) = index trong .npy - 1
    frame_idx   INTEGER NOT NULL,        -- index tuyệt đối trong video gốc
    pts_time    REAL NOT NULL,
    fps         REAL,
    file_name   TEXT,                    -- 0001.jpg
    s3_bucket   TEXT,
    s3_key      TEXT,
    width       INTEGER,
    height      INTEGER,
    size_bytes  INTEGER,
    phash       TEXT,
    UNIQUE (video_id, n)
);
CREATE INDEX IF NOT EXISTS idx_keyframes_video_frame ON keyframes(video_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_keyframes_file ON keyframes(video_id, file_name);

-- Object detection (Faster R-CNN / OpenImages V4)
CREATE TABLE IF NOT EXISTS objects (
    id           INTEGER PRIMARY KEY,
    keyframe_id  INTEGER NOT NULL REFERENCES keyframes(id) ON DELETE CASCADE,
    video_id     TEXT NOT NULL,
    frame_idx    INTEGER NOT NULL,
    rank         INTEGER NOT NULL,       -- thứ hạng theo score trong file json
    entity       TEXT NOT NULL,          -- "Tomato"
    class_name   TEXT,                   -- "/m/07j87"
    class_label  INTEGER,
    score        REAL NOT NULL,
    ymin REAL, xmin REAL, ymax REAL, xmax REAL,
    area         REAL
);
CREATE INDEX IF NOT EXISTS idx_objects_entity ON objects(entity, score DESC);
CREATE INDEX IF NOT EXISTS idx_objects_keyframe ON objects(keyframe_id);
CREATE INDEX IF NOT EXISTS idx_objects_video ON objects(video_id, frame_idx);

-- CLIP features: .npy nằm trên MinIO, SQLite giữ con trỏ + offset dòng
CREATE TABLE IF NOT EXISTS clip_features (
    video_id    TEXT PRIMARY KEY REFERENCES videos(video_id) ON DELETE CASCADE,
    s3_bucket   TEXT,
    s3_key      TEXT,
    n_vectors   INTEGER,
    dim         INTEGER,
    dtype       TEXT,
    sha256      TEXT,
    normalized  INTEGER DEFAULT 0,
    matches_keyframes INTEGER            -- 1 nếu n_vectors == số keyframe
);

-- Metadata YouTube
CREATE TABLE IF NOT EXISTS media_info (
    video_id     TEXT PRIMARY KEY REFERENCES videos(video_id) ON DELETE CASCADE,
    title        TEXT,
    description  TEXT,
    author       TEXT,
    channel_id   TEXT,
    channel_url  TEXT,
    length_s     INTEGER,
    publish_date TEXT,
    thumbnail_url TEXT,
    watch_url    TEXT,
    raw_json     TEXT
);

CREATE TABLE IF NOT EXISTS media_keywords (
    video_id  TEXT NOT NULL REFERENCES media_info(video_id) ON DELETE CASCADE,
    keyword   TEXT NOT NULL,
    PRIMARY KEY (video_id, keyword)
);

-- Document gộp để full-text search
CREATE TABLE IF NOT EXISTS documents (
    id        INTEGER PRIMARY KEY,
    video_id  TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    scope     TEXT NOT NULL,             -- 'video'
    title     TEXT,
    body      TEXT,
    objects   TEXT,
    UNIQUE (video_id, scope)
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title, body, objects,
    content='documents', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);

-- Manifest: stage nào đã chạy xong cho video nào (resumable)
CREATE TABLE IF NOT EXISTS ingest_manifest (
    stage       TEXT NOT NULL,
    video_id    TEXT NOT NULL,
    fingerprint TEXT,
    status      TEXT NOT NULL DEFAULT 'ok',
    details     TEXT,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (stage, video_id)
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or CONFIG.sqlite_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ------------------------------------------------------------------ manifest


def mark_done(
    conn: sqlite3.Connection,
    stage: str,
    video_id: str,
    fingerprint: str | None = None,
    details: dict | None = None,
    status: str = "ok",
) -> None:
    conn.execute(
        """
        INSERT INTO ingest_manifest (stage, video_id, fingerprint, status, details, updated_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(stage, video_id) DO UPDATE SET
            fingerprint = excluded.fingerprint,
            status      = excluded.status,
            details     = excluded.details,
            updated_at  = datetime('now')
        """,
        (stage, video_id, fingerprint, status, json.dumps(details or {}, ensure_ascii=False)),
    )


def is_done(
    conn: sqlite3.Connection, stage: str, video_id: str, fingerprint: str | None = None
) -> bool:
    row = conn.execute(
        "SELECT fingerprint, status FROM ingest_manifest WHERE stage = ? AND video_id = ?",
        (stage, video_id),
    ).fetchone()
    if row is None or row["status"] != "ok":
        return False
    if fingerprint is None:
        return True
    return row["fingerprint"] == fingerprint


def done_videos(conn: sqlite3.Connection, stage: str) -> set[str]:
    return {
        r["video_id"]
        for r in conn.execute(
            "SELECT video_id FROM ingest_manifest WHERE stage = ? AND status = 'ok'", (stage,)
        )
    }
