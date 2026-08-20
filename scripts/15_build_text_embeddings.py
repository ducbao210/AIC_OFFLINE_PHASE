#!/usr/bin/env python3
"""Create CLIP text embeddings for ASR/caption rows.

The text encoder must be the same CLIP checkpoint used to create the image
features. With the default OpenAI ViT-B/32 checkpoint, both modalities live in
the same 512-dimensional normalized space, so a text query can search the
image FAISS index directly. This stage also writes a separate text FAISS
artifact for ASR/caption retrieval.
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
        rows.append(
            {
                "kind": "asr",
                "video_id": r["video_id"],
                "segment_index": r["segment_index"],
                "start_s": r["start_s"],
                "end_s": r["end_s"],
                "text": " ".join(r["text"].split())[:max_chars],
                "model_name": r["model_name"],
                "language": r["language"],
            }
        )
    for r in con.execute(
        "SELECT video_id, keyframe_id, frame_idx, caption_text, model_name "
        "FROM captions WHERE caption_text IS NOT NULL AND trim(caption_text) <> '' "
        "ORDER BY video_id, frame_idx"
    ):
        rows.append(
            {
                "kind": "caption",
                "video_id": r["video_id"],
                "keyframe_id": r["keyframe_id"],
                "frame_idx": r["frame_idx"],
                "text": " ".join(r["caption_text"].split())[:max_chars],
                "model_name": r["model_name"],
            }
        )
    all_videos = {r["video_id"] for r in con.execute("SELECT video_id FROM videos")}
    asr_videos = {
        r["video_id"] for r in con.execute("SELECT DISTINCT video_id FROM transcripts")
    }
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


def encode_text(model, tokenizer, texts: list[str], device: str) -> np.ndarray:
    """Encode text in the CLIP space and return L2-normalized float32 vectors."""
    import torch

    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        vectors = model.encode_text(tokens)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vectors.float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument(
        "--model",
        default="ViT-B-32",
        help="open_clip model architecture; must match the image feature extractor",
    )
    ap.add_argument(
        "--pretrained",
        default="openai",
        help="open_clip checkpoint; must match the image feature extractor",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--max-text-chars", type=int, default=512)
    args = ap.parse_args()

    rows, stats = load_rows(args.db, args.max_text_chars)
    if not rows:
        raise SystemExit("No ASR/caption rows found")
    print(
        json.dumps(
            {"input": stats, "known_missing_asr": sorted(MISSING_ASR)},
            ensure_ascii=False,
        )
    )

    import open_clip
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.to(device).eval()

    # Derive dimension from the loaded checkpoint instead of hard-coding it.
    with torch.no_grad():
        probe = encode_text(model, tokenizer, ["dimension probe"], device)
    dim = int(probe.shape[1])
    if args.model == "ViT-B-32" and args.pretrained == "openai" and dim != 512:
        raise SystemExit(f"Unexpected CLIP ViT-B/32 OpenAI dimension: {dim}")

    args.out.mkdir(parents=True, exist_ok=True)
    emb_path = args.out / "text_embeddings.f32.npy"
    mmap = np.lib.format.open_memmap(
        emb_path, mode="w+", dtype="float32", shape=(len(rows), dim)
    )
    metadata_path = args.out / "text_embedding_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as meta:
        for start in range(0, len(rows), args.batch_size):
            end = min(start + args.batch_size, len(rows))
            vectors = encode_text(
                model, tokenizer, [r["text"] for r in rows[start:end]], device
            )
            if vectors.shape != (end - start, dim):
                raise RuntimeError(
                    f"Unexpected embedding shape {vectors.shape}; expected {(end - start, dim)}"
                )
            mmap[start:end] = vectors
            for row_id, row in enumerate(rows[start:end], start=start):
                meta.write(
                    json.dumps({"row": row_id, **row}, ensure_ascii=False) + "\n"
                )
            mmap.flush()
    del mmap

    manifest = {
        "artifact": "text_embeddings.f32.npy",
        "metadata": "text_embedding_metadata.json",
        "model_family": "open_clip",
        "model": args.model,
        "pretrained": args.pretrained,
        "space": f"CLIP:{args.model}:{args.pretrained}",
        "dimension": dim,
        "dtype": "float32",
        "normalized": True,
        "metric_hint": "cosine",
        "compatible_with_image_clip_space": True,
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
