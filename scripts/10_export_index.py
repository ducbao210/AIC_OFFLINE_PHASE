#!/usr/bin/env python3
"""Stage 10 — Gộp toàn bộ CLIP features thành 1 ma trận đã chuẩn hoá + id_map cho FAISS.

Xuất ra:
  out/clip_b32.f32.npy   (N, 512) float32, L2-normalized → dot product = cosine
  out/id_map.json        [{row, keyframe_id, video_id, n, frame_idx, s3_key}, ...]

Tra ngược sau khi search: row → id_map[row] → (video_id, frame_idx) là đáp án nộp thi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from aic.cli import base_parser, select_videos  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema  # noqa: E402
from aic.utils import find_clip_features, setup_logging  # noqa: E402


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "out")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--strict", action="store_true",
                        help="Bỏ qua video có số vector lệch số keyframe.")
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    conn = connect()
    init_schema(conn)
    feats = find_clip_features(CONFIG.data_root)
    targets = select_videos(args, list(feats))

    matrices: list[np.ndarray] = []
    id_map: list[dict] = []
    row = 0

    for video_id in targets:
        keyframes = conn.execute(
            "SELECT id, n, frame_idx, s3_key FROM keyframes WHERE video_id = ? ORDER BY n",
            (video_id,),
        ).fetchall()
        arr = np.load(feats[video_id]).astype(np.float32)
        if len(keyframes) != arr.shape[0]:
            msg = f"{video_id}: {arr.shape[0]} vector vs {len(keyframes)} keyframe"
            if args.strict:
                log.warning("bỏ qua %s", msg)
                continue
            log.warning("%s — cắt theo min()", msg)
        limit = min(len(keyframes), arr.shape[0])
        if limit == 0:
            continue

        arr = arr[:limit]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrices.append(arr / norms)

        for kf in keyframes[:limit]:
            id_map.append({
                "row": row,
                "keyframe_id": kf["id"],
                "video_id": video_id,
                "n": kf["n"],
                "frame_idx": kf["frame_idx"],
                "s3_key": kf["s3_key"],
            })
            row += 1

    if not matrices:
        log.error("Không có vector nào để xuất.")
        return 1

    matrix = np.vstack(matrices).astype(args.dtype)
    args.out.mkdir(parents=True, exist_ok=True)
    npy_path = args.out / f"clip_b32.{'f32' if args.dtype == 'float32' else 'f16'}.npy"
    np.save(npy_path, matrix)
    (args.out / "id_map.json").write_text(
        json.dumps(id_map, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "index_manifest.json").write_text(
        json.dumps({
            "model": "clip-ViT-B-32 (BTC provided)",
            "n_vectors": int(matrix.shape[0]),
            "dim": int(matrix.shape[1]),
            "dtype": args.dtype,
            "normalized": True,
            "metric": "cosine (inner product on normalized vectors)",
            "n_videos": len({m["video_id"] for m in id_map}),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("đã xuất %s %s và id_map.json", npy_path, matrix.shape)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
