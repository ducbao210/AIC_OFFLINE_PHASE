#!/usr/bin/env python3
"""Stage 08 — Dựng `documents` (title + description + keywords + object entities) và FTS5.

FTS5 dùng tokenizer unicode61 remove_diacritics=2 → gõ 'nang luong tich cuc' vẫn khớp
'năng lượng tích cực'.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser, select_videos  # noqa: E402
from aic.db import connect, init_schema, transaction  # noqa: E402
from aic.utils import nfc, setup_logging  # noqa: E402

STAGE = "08_documents"


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--top-objects", type=int, default=40,
                        help="Số entity phổ biến nhất gộp vào document mỗi video.")
    parser.add_argument("--object-min-score", type=float, default=0.4)
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    conn = connect()
    init_schema(conn)
    all_ids = [r["video_id"] for r in conn.execute("SELECT video_id FROM videos ORDER BY video_id")]
    targets = select_videos(args, all_ids)
    log.info("dựng document cho %d video", len(targets))

    for video_id in targets:
        meta = conn.execute(
            "SELECT title, description, author FROM media_info WHERE video_id = ?", (video_id,)
        ).fetchone()
        keywords = [
            r["keyword"]
            for r in conn.execute(
                "SELECT keyword FROM media_keywords WHERE video_id = ?", (video_id,)
            )
        ]
        entities = [
            f"{r['entity']}({r['c']})"
            for r in conn.execute(
                """
                SELECT entity, COUNT(*) AS c FROM objects
                WHERE video_id = ? AND score >= ?
                GROUP BY entity ORDER BY c DESC LIMIT ?
                """,
                (video_id, args.object_min_score, args.top_objects),
            )
        ]

        title = nfc(meta["title"]) if meta else None
        body_parts = [
            nfc(meta["description"]) if meta else None,
            nfc(meta["author"]) if meta else None,
            " ".join(keywords),
        ]
        body = "\n".join(p for p in body_parts if p)
        objects_text = " ".join(entities)
        if not (title or body or objects_text):
            continue
        if args.dry_run:
            log.info("%s: title=%s objects=%d", video_id, title, len(entities))
            continue

        with transaction(conn):
            conn.execute(
                """
                INSERT INTO documents (video_id, scope, title, body, objects)
                VALUES (?, 'video', ?, ?, ?)
                ON CONFLICT(video_id, scope) DO UPDATE SET
                    title=excluded.title, body=excluded.body, objects=excluded.objects
                """,
                (video_id, title, body, objects_text),
            )

    with transaction(conn):
        conn.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO documents_fts(documents_fts) VALUES('optimize')")

    n_docs = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
    log.info("documents: %d dòng, FTS đã rebuild", n_docs)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
