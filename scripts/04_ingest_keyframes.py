#!/usr/bin/env python3
"""Stage 04 — Upload keyframe .jpg lên MinIO và gắn s3_key/kích thước/pHash vào `keyframes`.

Ảnh thứ k (sắp xếp tăng dần theo tên) ứng với dòng n = k+1 trong map-keyframes CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser, select_videos  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema, mark_done, transaction  # noqa: E402
from aic.storage import ensure_buckets, get_client, keyframe_key, upload_file  # noqa: E402
from aic.utils import find_keyframe_dirs, parallel_map, setup_logging, video_group  # noqa: E402

STAGE = "04_keyframes"


def _image_meta(path: Path, want_phash: bool) -> tuple[int | None, int | None, str | None]:
    try:
        from PIL import Image
    except ImportError:
        return None, None, None
    try:
        with Image.open(path) as img:
            width, height = img.size
            phash = None
            if want_phash:
                import imagehash

                phash = str(imagehash.phash(img))
        return width, height, phash
    except Exception:  # noqa: BLE001
        return None, None, None


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--phash", action="store_true", help="Tính pHash (chậm hơn).")
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    kf_dirs = find_keyframe_dirs(CONFIG.data_root)
    conn = connect()
    init_schema(conn)
    targets = select_videos(args, list(kf_dirs), conn, STAGE)
    log.info("xử lý %d/%d thư mục keyframe", len(targets), len(kf_dirs))

    client = None
    if not args.no_upload:
        client = get_client()
        ensure_buckets(client, (CONFIG.bucket_frames,))

    for video_id in targets:
        images = sorted(
            p for p in kf_dirs[video_id].iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not images:
            log.warning("%s: không có ảnh", video_id)
            continue

        n_rows = conn.execute(
            "SELECT COUNT(*) AS c FROM keyframes WHERE video_id = ?", (video_id,)
        ).fetchone()["c"]
        if n_rows and n_rows != len(images):
            log.warning(
                "%s: lệch số lượng — CSV %d dòng vs %d ảnh", video_id, n_rows, len(images)
            )
        if args.dry_run:
            log.info("%s: %d ảnh", video_id, len(images))
            continue

        def work(item: tuple[int, Path]) -> tuple:
            n, path = item
            key = keyframe_key(video_id, path.name)
            if client is not None:
                upload_file(client, CONFIG.bucket_frames, key, path)
            width, height, phash = _image_meta(path, args.phash)
            return (
                CONFIG.bucket_frames if client else None, key if client else None,
                path.name, width, height, path.stat().st_size, phash, video_id, n,
            )

        rows = list(parallel_map(
            work, list(enumerate(images, start=1)), workers=args.workers, desc=video_id
        ))
        with transaction(conn):
            conn.execute(
                "INSERT OR IGNORE INTO videos (video_id, group_id) VALUES (?, ?)",
                (video_id, video_group(video_id)),
            )
            # đảm bảo có dòng keyframe kể cả khi thiếu CSV
            conn.executemany(
                """
                INSERT OR IGNORE INTO keyframes (video_id, n, frame_idx, pts_time)
                VALUES (?, ?, -1, -1)
                """,
                [(video_id, n) for n, _ in enumerate(images, start=1)],
            )
            conn.executemany(
                """
                UPDATE keyframes SET
                    s3_bucket = ?, s3_key = ?, file_name = ?,
                    width = ?, height = ?, size_bytes = ?, phash = ?
                WHERE video_id = ? AND n = ?
                """,
                rows,
            )
            mark_done(conn, STAGE, video_id, None, {"n_images": len(images)})

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
