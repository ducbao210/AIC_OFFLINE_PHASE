#!/usr/bin/env python3
"""Stage 07 — Nạp media-info/*.json (metadata YouTube) vào `media_info` + `media_keywords`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser, select_videos  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema, mark_done, transaction  # noqa: E402
from aic.utils import find_media_info, nfc, read_json, setup_logging, video_group  # noqa: E402

STAGE = "07_media_info"


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    log = setup_logging(args.verbose)

    metas = find_media_info(CONFIG.data_root)
    conn = connect()
    init_schema(conn)
    targets = select_videos(args, list(metas), conn, STAGE)
    log.info("xử lý %d/%d file media-info", len(targets), len(metas))

    for video_id in targets:
        try:
            payload = read_json(metas[video_id])
        except Exception as exc:  # noqa: BLE001
            log.error("%s: JSON lỗi (%s)", video_id, exc)
            continue

        keywords = [nfc(str(k)).strip() for k in (payload.get("keywords") or []) if str(k).strip()]
        if args.dry_run:
            log.info("%s: %s (%d keyword)", video_id, payload.get("title"), len(keywords))
            continue

        with transaction(conn):
            conn.execute(
                "INSERT OR IGNORE INTO videos (video_id, group_id) VALUES (?, ?)",
                (video_id, video_group(video_id)),
            )
            conn.execute(
                """
                INSERT INTO media_info (
                    video_id, title, description, author, channel_id, channel_url,
                    length_s, publish_date, thumbnail_url, watch_url, raw_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(video_id) DO UPDATE SET
                    title=excluded.title, description=excluded.description,
                    author=excluded.author, channel_id=excluded.channel_id,
                    channel_url=excluded.channel_url, length_s=excluded.length_s,
                    publish_date=excluded.publish_date, thumbnail_url=excluded.thumbnail_url,
                    watch_url=excluded.watch_url, raw_json=excluded.raw_json
                """,
                (
                    video_id, nfc(payload.get("title")), nfc(payload.get("description")),
                    nfc(payload.get("author")), payload.get("channel_id"),
                    payload.get("channel_url"),
                    int(payload["length"]) if str(payload.get("length", "")).isdigit() else None,
                    payload.get("publish_date"), payload.get("thumbnail_url"),
                    payload.get("watch_url"),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            conn.execute("DELETE FROM media_keywords WHERE video_id = ?", (video_id,))
            conn.executemany(
                "INSERT OR IGNORE INTO media_keywords (video_id, keyword) VALUES (?, ?)",
                [(video_id, kw) for kw in set(keywords)],
            )
            mark_done(conn, STAGE, video_id, None, {"n_keywords": len(keywords)})

    total = conn.execute("SELECT COUNT(*) AS c FROM media_info").fetchone()["c"]
    no_meta = conn.execute(
        "SELECT COUNT(*) AS c FROM videos v LEFT JOIN media_info m USING(video_id) "
        "WHERE m.video_id IS NULL"
    ).fetchone()["c"]
    log.info("media_info: %d dòng; %d video không có metadata", total, no_meta)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
