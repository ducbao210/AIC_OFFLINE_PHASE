#!/usr/bin/env python3
"""Prepare SQL-side object retrieval without embedding every detection row."""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--min-score", type=float, default=0.50)
    ap.add_argument(
        "--journal-mode",
        choices=("DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"),
        default=os.environ.get("AIC_SQLITE_JOURNAL_MODE", "DELETE").upper(),
        help="SQLite journal mode; DELETE is safest on Google Drive/FUSE mounts.",
    )
    args = ap.parse_args()
    try:
        con = sqlite3.connect(args.db, timeout=60)
        con.execute(f"PRAGMA journal_mode={args.journal_mode}")
        con.execute("PRAGMA foreign_keys=ON")
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(
            f"Cannot open SQLite database {args.db}. "
            "If it is on Google Drive, restore/copy a healthy aic.sqlite backup "
            "and remove stale aic.sqlite-wal/aic.sqlite-shm sidecars before rerunning Phase 8."
        ) from exc
    con.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_objects_entity_score
        ON objects(entity, score DESC);
        CREATE INDEX IF NOT EXISTS idx_objects_video_entity_score
        ON objects(video_id, entity, score DESC);
        CREATE INDEX IF NOT EXISTS idx_objects_keyframe_score
        ON objects(keyframe_id, score DESC);
        CREATE TABLE IF NOT EXISTS keyframe_object_summary (
            keyframe_id INTEGER PRIMARY KEY,
            video_id TEXT NOT NULL,
            frame_idx INTEGER NOT NULL,
            object_text TEXT NOT NULL,
            object_count INTEGER NOT NULL,
            max_score REAL NOT NULL,
            min_score REAL NOT NULL,
            min_score_used REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_kos_video_frame
        ON keyframe_object_summary(video_id, frame_idx);
        CREATE INDEX IF NOT EXISTS idx_kos_object_text
        ON keyframe_object_summary(object_text);
        """
    )
    con.execute("DELETE FROM keyframe_object_summary")
    con.execute(
        """
        INSERT INTO keyframe_object_summary
          (keyframe_id, video_id, frame_idx, object_text, object_count,
           max_score, min_score, min_score_used)
        SELECT keyframe_id, video_id, frame_idx,
               GROUP_CONCAT(entity, ' '), COUNT(*), MAX(score), MIN(score), ?
        FROM objects
        WHERE score >= ?
        GROUP BY keyframe_id, video_id, frame_idx
        """,
        (args.min_score, args.min_score),
    )
    con.commit()
    count = con.execute("SELECT COUNT(*) FROM keyframe_object_summary").fetchone()[0]
    entities = con.execute("SELECT COUNT(DISTINCT entity) FROM objects WHERE score >= ?", (args.min_score,)).fetchone()[0]
    con.close()
    print(f"object summaries: {count} keyframes, {entities} entities, min_score={args.min_score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
