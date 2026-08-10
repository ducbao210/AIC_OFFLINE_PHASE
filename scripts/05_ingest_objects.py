#!/usr/bin/env python3
"""Stage 05 — Nạp objects/<video_id>/<name>.json (Faster R-CNN OpenImages V4) vào bảng `objects`.

Mỗi file JSON tương ứng 1 keyframe: khớp theo tên file (011.json ↔ 011.jpg), fallback theo thứ tự.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser, select_videos  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema, mark_done, transaction  # noqa: E402
from aic.utils import find_object_dirs, read_json, setup_logging  # noqa: E402

STAGE = "05_objects"


def keyframe_index(conn, video_id: str) -> tuple[dict[str, tuple[int, int]], list[tuple[int, int]]]:
    """(map theo stem file_name, list theo thứ tự n) -> (keyframe_id, frame_idx)."""
    by_stem: dict[str, tuple[int, int]] = {}
    ordered: list[tuple[int, int]] = []
    for row in conn.execute(
        "SELECT id, n, frame_idx, file_name FROM keyframes WHERE video_id = ? ORDER BY n",
        (video_id,),
    ):
        ordered.append((row["id"], row["frame_idx"]))
        if row["file_name"]:
            by_stem[Path(row["file_name"]).stem] = (row["id"], row["frame_idx"])
    return by_stem, ordered


def parse_objects(payload: dict, min_score: float, limit: int | None) -> list[tuple]:
    scores = payload.get("detection_scores") or []
    entities = payload.get("detection_class_entities") or []
    names = payload.get("detection_class_names") or []
    labels = payload.get("detection_class_labels") or []
    boxes = payload.get("detection_boxes") or []

    out: list[tuple] = []
    for rank, raw_score in enumerate(scores):
        score = float(raw_score)
        if score < min_score:
            break  # danh sách đã sắp giảm dần
        if limit is not None and rank >= limit:
            break
        box = boxes[rank] if rank < len(boxes) else ["0", "0", "0", "0"]
        ymin, xmin, ymax, xmax = (float(v) for v in box)
        out.append((
            rank,
            entities[rank] if rank < len(entities) else None,
            names[rank] if rank < len(names) else None,
            int(labels[rank]) if rank < len(labels) and str(labels[rank]).isdigit() else None,
            score, ymin, xmin, ymax, xmax,
            max(0.0, ymax - ymin) * max(0.0, xmax - xmin),
        ))
    return out


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--min-score", type=float, default=CONFIG.object_min_score)
    parser.add_argument("--limit-per-frame", type=int, default=None,
                        help="Chỉ giữ N object điểm cao nhất mỗi keyframe.")
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    obj_dirs = find_object_dirs(CONFIG.data_root)
    conn = connect()
    init_schema(conn)
    targets = select_videos(args, list(obj_dirs), conn, STAGE)
    log.info("xử lý %d/%d thư mục objects (min_score=%.2f)",
             len(targets), len(obj_dirs), args.min_score)

    total = 0
    for video_id in targets:
        files = sorted(p for p in obj_dirs[video_id].glob("*.json"))
        if not files:
            continue
        by_stem, ordered = keyframe_index(conn, video_id)
        if not ordered:
            log.warning("%s: chưa có keyframe trong DB — chạy stage 03/04 trước", video_id)
            continue

        rows: list[tuple] = []
        unmatched = 0
        for pos, path in enumerate(files):
            target = by_stem.get(path.stem)
            if target is None:
                target = ordered[pos] if pos < len(ordered) else None
                unmatched += 1
            if target is None:
                continue
            keyframe_id, frame_idx = target
            try:
                payload = read_json(path)
            except Exception as exc:  # noqa: BLE001
                log.error("%s: JSON lỗi (%s)", path, exc)
                continue
            for det in parse_objects(payload, args.min_score, args.limit_per_frame):
                rows.append((keyframe_id, video_id, frame_idx, *det))

        if args.dry_run:
            log.info("%s: %d file → %d detection", video_id, len(files), len(rows))
            continue
        if unmatched:
            log.warning("%s: %d file JSON phải khớp theo thứ tự (thiếu file_name)",
                        video_id, unmatched)

        with transaction(conn):
            conn.execute("DELETE FROM objects WHERE video_id = ?", (video_id,))
            conn.executemany(
                """
                INSERT INTO objects (
                    keyframe_id, video_id, frame_idx, rank, entity, class_name,
                    class_label, score, ymin, xmin, ymax, xmax, area
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            mark_done(conn, STAGE, video_id,
                      f"score>={args.min_score}",
                      {"n_files": len(files), "n_detections": len(rows)})
        total += len(rows)

    log.info("đã nạp %d detection", total)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
