#!/usr/bin/env python3
"""Create ASR/caption embeddings and metadata artifacts only.

This stage does not search, rank, or build a vector database. It exports a
model-agnostic float32 matrix plus JSONL metadata for the downstream model.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

MISSING_ASR = {"L24_V006", "L24_V022", "L24_V040", "L30_V029"}


def load_rows(db_path: Path, max_chars: int) -> tuple[list[dict], dict]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows: list[dict] = []
    for r in con.execute(
        "SELECT video_id, segment_index, start_s, end_s, text, model_name, language "
        "FROM transcripts WHERE text IS NOT NULL AND trim(text) <> '' "
        "ORDER BY video_id, segment_index"
    ):
        rows.append({
            "kind": "asr", "video_id": r["video_id"],
            "segment_index": r["segment_index"], "start_s": r["start_s"],
            "end_s": r["end_s"], "text": " ".join(r["text"].split())[:max_chars],
            "model_name": r["model_name"], "language": r["language"],
        })
    for r in con.execute(
        "SELECT video_id, keyframe_id, frame_idx, caption_text, model_name "
        "FROM captions WHERE caption_text IS NOT NULL AND trim(caption_text) <> '' "
        "ORDER BY video_id, frame_idx"
    ):
        rows.append({
            "kind": "caption", "video_id": r["video_id"],
            "keyframe_id": r["keyframe_id"], "frame_idx": r["frame_idx"],
            "text": " ".join(r["caption_text"].split())[:max_chars],
            "model_name": r["model_name"],
        })
    all_videos = {r["video_id"] for r in con.execute("SELECT video_id FROM videos")}
    asr_videos = {r["video_id"] for r in con.execute("SELECT DISTINCT video_id FROM transcripts")}
    con.close()
    stats = {
        "n_rows": len(rows),
        "n_asr": sum(r["kind"] == "asr" for r in rows),
        "n_captions": sum(r["kind"] == "caption" for r in rows),
        "n_videos": len(all_videos),
        "n_videos_with_asr": len(asr_videos),
        "missing_asr_videos": sorted(all_videos - asr_videos),
    }
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--max-text-chars", type=int, default=512)
    args = ap.parse_args()

    rows, stats = load_rows(args.db, args.max_text_chars)
    if not rows:
        raise SystemExit("No ASR/caption rows found")
    print(json.dumps({"input": stats, "known_missing_asr": sorted(MISSING_ASR)}, ensure_ascii=False))

    from sentence_transformers import SentenceTransformer
    model_kwargs = {} if args.device is None else {"device": args.device}
    model = SentenceTransformer(args.model, **model_kwargs)
    dim = int(model.get_sentence_embedding_dimension())
    args.out.mkdir(parents=True, exist_ok=True)
    emb_path = args.out / "text_embeddings.f32.npy"
    mmap = np.lib.format.open_memmap(
        emb_path, mode="w+", dtype="float32", shape=(len(rows), dim)
    )
    metadata_path = args.out / "text_embedding_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as meta:
        for start in range(0, len(rows), args.batch_size):
            end = min(start + args.batch_size, len(rows))
            vectors = model.encode(
                [r["text"] for r in rows[start:end]],
                batch_size=args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).astype("float32")
            mmap[start:end] = vectors
            for row_id, row in enumerate(rows[start:end], start=start):
                meta.write(json.dumps({"row": row_id, **row}, ensure_ascii=False) + "\n")
            mmap.flush()
    del mmap
    manifest = {
        "artifact": "text_embeddings.f32.npy",
        "metadata": "text_embedding_metadata.jsonl",
        "model": args.model,
        "dimension": dim,
        "dtype": "float32",
        "normalized": True,
        "metric_hint": "cosine",
        "input_stats": stats,
        "batch_size": args.batch_size,
        "max_text_chars": args.max_text_chars,
    }
    (args.out / "text_embedding_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
