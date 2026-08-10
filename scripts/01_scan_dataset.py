#!/usr/bin/env python3
"""Stage 01 — Quét cây thư mục AIC_26_DATA, dựng inventory `assets` và khung `videos`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser, select_videos  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema, transaction  # noqa: E402
from aic.utils import (  # noqa: E402
    find_clip_features,
    find_keyframe_dirs,
    find_map_keyframes,
    find_media_info,
    find_object_dirs,
    find_video_files,
    setup_logging,
    video_group,
)


def _size_and_mtime(path: Path) -> tuple[int | None, float | None]:
    try:
        st = path.stat()
    except OSError:
        return None, None
    if path.is_dir():
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return total, st.st_mtime
    return st.st_size, st.st_mtime


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    log = setup_logging(args.verbose)
    root = CONFIG.data_root
    if not root.exists():
        log.error("AIC_DATA_ROOT không tồn tại: %s", root)
        return 1

    sources = {
        "video": find_video_files(root),
        "keyframe_dir": find_keyframe_dirs(root),
        "map_csv": find_map_keyframes(root),
        "object_dir": find_object_dirs(root),
        "media_json": find_media_info(root),
        "clip_npy": find_clip_features(root),
    }
    for kind, mapping in sources.items():
        log.info("%-13s : %d", kind, len(mapping))

    all_ids = sorted({vid for m in sources.values() for vid in m})
    selected = set(select_videos(args, all_ids))
    log.info("tổng video_id: %d (xử lý %d)", len(all_ids), len(selected))

    if args.dry_run:
        for vid in sorted(selected)[:20]:
            log.info("  %s -> %s", vid, [k for k, m in sources.items() if vid in m])
        return 0

    conn = connect()
    init_schema(conn)
    with transaction(conn):
        for kind, mapping in sources.items():
            rows = []
            for vid, path in mapping.items():
                if vid not in selected:
                    continue
                size, mtime = _size_and_mtime(path)
                rows.append((kind, vid, str(path.relative_to(root)), size, mtime))
            conn.executemany(
                """
                INSERT INTO assets (kind, video_id, rel_path, size_bytes, mtime)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(kind, video_id) DO UPDATE SET
                    rel_path = excluded.rel_path,
                    size_bytes = excluded.size_bytes,
                    mtime = excluded.mtime
                """,
                rows,
            )

        # tạo khung videos để các stage sau có khoá ngoại
        conn.executemany(
            """
            INSERT INTO videos (video_id, group_id, rel_path)
            VALUES (?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                rel_path = COALESCE(excluded.rel_path, videos.rel_path)
            """,
            [
                (
                    vid,
                    video_group(vid),
                    str(sources["video"][vid].relative_to(root)) if vid in sources["video"] else None,
                )
                for vid in sorted(selected)
            ],
        )

    missing_video = sorted(v for v in selected if v not in sources["video"])
    missing_meta = sorted(v for v in selected if v not in sources["media_json"])
    log.info("thiếu file mp4     : %d %s", len(missing_video), missing_video[:5])
    log.info("thiếu media-info   : %d %s", len(missing_meta), missing_meta[:5])
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
