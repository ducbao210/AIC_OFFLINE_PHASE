#!/usr/bin/env python3
"""Stage 09 — Kiểm tra toàn vẹn: đếm bảng, khớp CSV ↔ ảnh ↔ .npy, object mồ côi, object MinIO."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema  # noqa: E402
from aic.storage import get_client, object_exists  # noqa: E402
from aic.utils import setup_logging  # noqa: E402


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--sample", type=int, default=100,
                        help="Số object MinIO lấy mẫu kiểm tra tồn tại.")
    parser.add_argument("--skip-minio", action="store_true")
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    conn = connect()
    init_schema(conn)
    problems = 0

    log.info("--- Thống kê ---")
    for table in ("videos", "keyframes", "objects", "clip_features",
                  "media_info", "media_keywords", "documents",
                  "transcripts", "captions"):
        c = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        log.info("%-15s %d", table, c)

    log.info("--- Kiểm tra ---")

    checks = {
        "keyframe thiếu frame_idx hợp lệ":
            "SELECT COUNT(*) AS c FROM keyframes WHERE frame_idx < 0",
        "keyframe chưa upload MinIO":
            "SELECT COUNT(*) AS c FROM keyframes WHERE s3_key IS NULL",
        "video chưa upload MinIO":
            "SELECT COUNT(*) AS c FROM videos WHERE s3_key IS NULL",
        "video thiếu thông tin ffprobe":
            "SELECT COUNT(*) AS c FROM videos WHERE fps IS NULL",
        "video thiếu CLIP features":
            "SELECT COUNT(*) AS c FROM videos v LEFT JOIN clip_features f USING(video_id) "
            "WHERE f.video_id IS NULL",
        "CLIP lệch số vector ↔ keyframe":
            "SELECT COUNT(*) AS c FROM clip_features WHERE matches_keyframes = 0",
        "video không có keyframe":
            "SELECT COUNT(*) AS c FROM videos v WHERE NOT EXISTS "
            "(SELECT 1 FROM keyframes k WHERE k.video_id = v.video_id)",
        "frame_idx không tăng đơn điệu":
            "SELECT COUNT(*) AS c FROM (SELECT video_id, frame_idx, "
            "LAG(frame_idx) OVER (PARTITION BY video_id ORDER BY n) AS prev FROM keyframes) "
            "WHERE prev IS NOT NULL AND frame_idx <= prev",
        "frame_idx vượt quá n_frames của video":
            "SELECT COUNT(*) AS c FROM keyframes k JOIN videos v USING(video_id) "
            "WHERE v.n_frames IS NOT NULL AND k.frame_idx >= v.n_frames",
        "pts_time lệch >0.5s so với frame_idx/fps":
            "SELECT COUNT(*) AS c FROM keyframes WHERE fps > 0 "
            "AND ABS(pts_time - (frame_idx * 1.0 / fps)) > 0.5",
        "video có transcript ASR":
            "SELECT COUNT(*) AS c FROM videos v WHERE NOT EXISTS "
            "(SELECT 1 FROM transcripts t WHERE t.video_id = v.video_id)",
        "video có caption":
            "SELECT COUNT(*) AS c FROM videos v WHERE NOT EXISTS "
            "(SELECT 1 FROM captions c WHERE c.video_id = v.video_id)",
        "caption mồ côi (không khớp keyframe)":
            "SELECT COUNT(*) AS c FROM captions c WHERE NOT EXISTS "
            "(SELECT 1 FROM keyframes k WHERE k.id = c.keyframe_id)",
    }
    for label, sql in checks.items():
        c = conn.execute(sql).fetchone()["c"]
        level = log.info if c == 0 else log.warning
        problems += int(c > 0)
        level("%-45s %d", label, c)

    if not args.skip_minio:
        rows = conn.execute(
            "SELECT s3_bucket, s3_key FROM keyframes WHERE s3_key IS NOT NULL"
        ).fetchall()
        rows += conn.execute(
            "SELECT s3_bucket, s3_key FROM videos WHERE s3_key IS NOT NULL"
        ).fetchall()
        rows += conn.execute(
            "SELECT s3_bucket, s3_key FROM clip_features WHERE s3_key IS NOT NULL"
        ).fetchall()
        if rows:
            client = get_client()
            sample = random.sample(rows, min(args.sample, len(rows)))
            missing = [r["s3_key"] for r in sample
                       if not object_exists(client, r["s3_bucket"], r["s3_key"])]
            if missing:
                problems += 1
                log.warning("MinIO thiếu %d/%d object mẫu: %s",
                            len(missing), len(sample), missing[:5])
            else:
                log.info("%-45s %d/%d OK", "object MinIO lấy mẫu", len(sample), len(sample))

    log.info("--- Ví dụ dữ liệu ---")
    for row in conn.execute(
        "SELECT video_id, n, frame_idx, pts_time, fps, file_name FROM keyframes "
        "WHERE file_name IS NOT NULL ORDER BY RANDOM() LIMIT 5"
    ):
        log.info("  %s n=%s frame_idx=%s pts=%s fps=%s %s", *tuple(row))

    conn.close()
    log.info("Hoàn tất — %d hạng mục cần xem lại.", problems)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
