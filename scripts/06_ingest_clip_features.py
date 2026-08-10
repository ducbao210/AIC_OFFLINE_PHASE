#!/usr/bin/env python3
"""Stage 06 — Upload CLIP features (.npy, clip-ViT-B-32) lên MinIO và ghi con trỏ vào SQLite.

Thứ tự vector trong .npy tăng dần theo chỉ số keyframe: hàng i ↔ keyframe n = i + 1.
Script kiểm tra n_vectors có khớp số keyframe hay không (cờ `matches_keyframes`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser, select_videos  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema, mark_done, transaction  # noqa: E402
from aic.storage import (
    ensure_buckets,
    feature_key,
    get_client,
    upload_file,
)  # noqa: E402
from aic.utils import find_clip_features, setup_logging, sha256_file  # noqa: E402

STAGE = "06_clip_features"


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--no-hash", action="store_true")
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    feats = find_clip_features(CONFIG.data_root)
    conn = connect()
    init_schema(conn)
    targets = select_videos(args, list(feats), conn, STAGE)
    log.info("xử lý %d/%d file .npy", len(targets), len(feats))

    client = None
    if not args.no_upload:
        client = get_client()
        ensure_buckets(client, (CONFIG.bucket_feats,))

    mismatch = 0
    for video_id in targets:
        path = feats[video_id]
        arr = np.load(path)
        if arr.ndim != 2:
            log.error("%s: shape bất thường %s", video_id, arr.shape)
            continue
        n_vectors, dim = int(arr.shape[0]), int(arr.shape[1])
        n_keyframes = conn.execute(
            "SELECT COUNT(*) AS c FROM keyframes WHERE video_id = ?", (video_id,)
        ).fetchone()["c"]
        matches = int(n_keyframes > 0 and n_keyframes == n_vectors)
        if n_keyframes and not matches:
            mismatch += 1
            log.warning(
                "%s: %d vector vs %d keyframe", video_id, n_vectors, n_keyframes
            )

        if args.dry_run:
            log.info("%s: %s %s", video_id, arr.shape, arr.dtype)
            continue

        key = None
        if client is not None:
            key, _ = upload_file(
                client, CONFIG.bucket_feats, feature_key(video_id), path
            )

        with transaction(conn):
            conn.execute(
                """
                INSERT INTO clip_features (
                    video_id, s3_bucket, s3_key, n_vectors, dim, dtype, sha256, matches_keyframes
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(video_id) DO UPDATE SET
                    s3_bucket=excluded.s3_bucket, s3_key=excluded.s3_key,
                    n_vectors=excluded.n_vectors, dim=excluded.dim, dtype=excluded.dtype,
                    sha256=COALESCE(excluded.sha256, clip_features.sha256),
                    matches_keyframes=excluded.matches_keyframes
                """,
                (
                    video_id,
                    CONFIG.bucket_feats if key else None,
                    key,
                    n_vectors,
                    dim,
                    str(arr.dtype),
                    None if args.no_hash else sha256_file(path),
                    matches,
                ),
            )
            mark_done(conn, STAGE, video_id, None, {"n_vectors": n_vectors, "dim": dim})

    if mismatch:
        log.warning(
            "%d video lệch số vector ↔ keyframe — kiểm tra trước khi build index",
            mismatch,
        )
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
