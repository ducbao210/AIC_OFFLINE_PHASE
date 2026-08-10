#!/usr/bin/env python3
"""Stage 02 — ffprobe + sha256 video, upload .mp4 lên MinIO (bucket aic-raw)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser, select_videos  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema, mark_done, transaction  # noqa: E402
from aic.storage import ensure_buckets, get_client, upload_file, video_key  # noqa: E402
from aic.utils import (  # noqa: E402
    ffprobe,
    find_video_files,
    has_ffprobe,
    parallel_map,
    setup_logging,
    sha256_file,
    video_group,
)

STAGE = "02_videos"


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--no-upload", action="store_true", help="Chỉ probe, không upload.")
    parser.add_argument("--no-hash", action="store_true", help="Bỏ qua sha256 (nhanh hơn).")
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    videos = find_video_files(CONFIG.data_root)
    conn = connect()
    init_schema(conn)
    targets = select_videos(args, list(videos), conn, STAGE)
    log.info("xử lý %d/%d video", len(targets), len(videos))
    if args.dry_run or not targets:
        conn.close()
        return 0

    probe_ok = has_ffprobe()
    client = None
    if not args.no_upload:
        client = get_client()
        ensure_buckets(client, (CONFIG.bucket_raw,))

    def work(video_id: str) -> dict:
        path = videos[video_id]
        info = ffprobe(path) if probe_ok else {"size_bytes": path.stat().st_size}
        info["video_id"] = video_id
        info["sha256"] = None if args.no_hash else sha256_file(path)
        info["rel_path"] = str(path.relative_to(CONFIG.data_root))
        info["s3_key"] = None
        if client is not None:
            key, uploaded = upload_file(client, CONFIG.bucket_raw, video_key(video_id), path)
            info["s3_key"] = key
            info["uploaded"] = uploaded
        return info

    uploaded_count = 0
    for info in parallel_map(work, targets, workers=args.workers, desc="videos"):
        fps_num, fps_den = info.get("fps_num") or 0, info.get("fps_den") or 1
        with transaction(conn):
            conn.execute(
                """
                INSERT INTO videos (
                    video_id, group_id, rel_path, s3_bucket, s3_key, sha256, size_bytes,
                    fps_num, fps_den, fps, n_frames, duration_s, width, height, codec,
                    audio_codec, audio_sample_rate, audio_channels
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(video_id) DO UPDATE SET
                    rel_path=excluded.rel_path, s3_bucket=excluded.s3_bucket,
                    s3_key=excluded.s3_key, sha256=COALESCE(excluded.sha256, videos.sha256),
                    size_bytes=excluded.size_bytes, fps_num=excluded.fps_num,
                    fps_den=excluded.fps_den, fps=excluded.fps, n_frames=excluded.n_frames,
                    duration_s=excluded.duration_s, width=excluded.width, height=excluded.height,
                    codec=excluded.codec, audio_codec=excluded.audio_codec,
                    audio_sample_rate=excluded.audio_sample_rate,
                    audio_channels=excluded.audio_channels
                """,
                (
                    info["video_id"], video_group(info["video_id"]), info["rel_path"],
                    CONFIG.bucket_raw if info["s3_key"] else None, info["s3_key"],
                    info["sha256"], info.get("size_bytes"),
                    info.get("fps_num"), info.get("fps_den"),
                    (fps_num / fps_den) if fps_den else None,
                    info.get("n_frames"), info.get("duration_s"),
                    info.get("width"), info.get("height"), info.get("codec"),
                    info.get("audio_codec"), info.get("audio_sample_rate"),
                    info.get("audio_channels"),
                ),
            )
            mark_done(conn, STAGE, info["video_id"], info["sha256"],
                      {"s3_key": info["s3_key"], "n_frames": info.get("n_frames")})
        uploaded_count += int(bool(info.get("uploaded")))

    log.info("xong. upload mới: %d object", uploaded_count)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
