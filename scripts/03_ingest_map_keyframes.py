#!/usr/bin/env python3
"""Stage 03 — Nạp map-keyframes/*.csv (n, pts_time, fps, frame_idx) vào bảng `keyframes`.

Đây là **nguồn chân lý** cho ánh xạ keyframe ↔ frame_idx tuyệt đối của video gốc.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser, select_videos  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema, mark_done, transaction  # noqa: E402
from aic.utils import (  # noqa: E402
    find_map_keyframes,
    setup_logging,
    video_group,
)

STAGE = "03_map_keyframes"


def parse_csv(path: Path) -> list[tuple[int, float, float, int]]:
    """Trả về [(n, pts_time, fps, frame_idx)] — chấp nhận cả dấu phẩy lẫn khoảng trắng."""
    rows: list[tuple[int, float, float, int]] = []
    text = path.read_text(encoding="utf-8-sig").strip().splitlines()
    if not text:
        return rows
    delimiter = "," if "," in text[0] else None

    reader = (
        csv.reader(text, delimiter=delimiter) if delimiter else ([c for c in line.split()] for line in text)
    )
    header: list[str] | None = None
    for parts in reader:
        parts = [p.strip() for p in parts if p.strip() != ""]
        if not parts:
            continue
        if header is None:
            header = [p.lower() for p in parts]
            if "frame_idx" in header:
                continue
            header = ["n", "pts_time", "fps", "frame_idx"]
        idx = {name: header.index(name) for name in ("n", "pts_time", "fps", "frame_idx")}
        try:
            rows.append((
                int(float(parts[idx["n"]])),
                float(parts[idx["pts_time"]]),
                float(parts[idx["fps"]]),
                int(float(parts[idx["frame_idx"]])),
            ))
        except (ValueError, IndexError):
            continue
    return rows


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    log = setup_logging(args.verbose)

    csvs = find_map_keyframes(CONFIG.data_root)
    conn = connect()
    init_schema(conn)
    targets = select_videos(args, list(csvs), conn, STAGE)
    log.info("xử lý %d/%d file map-keyframes", len(targets), len(csvs))

    total = 0
    for video_id in targets:
        rows = parse_csv(csvs[video_id])
        if not rows:
            log.warning("%s: CSV rỗng/không parse được", video_id)
            continue
        if args.dry_run:
            log.info("%s: %d keyframe (frame_idx %d..%d)",
                     video_id, len(rows), rows[0][3], rows[-1][3])
            continue
        with transaction(conn):
            conn.execute(
                "INSERT OR IGNORE INTO videos (video_id, group_id) VALUES (?, ?)",
                (video_id, video_group(video_id)),
            )
            conn.executemany(
                """
                INSERT INTO keyframes (video_id, n, frame_idx, pts_time, fps)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(video_id, n) DO UPDATE SET
                    frame_idx = excluded.frame_idx,
                    pts_time  = excluded.pts_time,
                    fps       = excluded.fps
                """,
                [(video_id, n, frame_idx, pts, fps) for n, pts, fps, frame_idx in rows],
            )
            mark_done(conn, STAGE, video_id, None,
                      {"n_keyframes": len(rows), "max_frame_idx": rows[-1][3]})
        total += len(rows)

    log.info("đã nạp %d dòng keyframe", total)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
