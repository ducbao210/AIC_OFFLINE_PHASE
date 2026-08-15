#!/usr/bin/env python3
"""Stage 14 — Text-to-image search bằng FAISS index + CLIP ViT-B/32 text encoder."""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np
import open_clip
import torch
from aic.cli import base_parser
from aic.utils import setup_logging


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument(
        "query",
        type=str,
        help="Câu truy vấn (tiếng Anh khuyến nghị, CLIP gốc train trên EN)",
    )
    parser.add_argument(
        "--src", type=Path, default=Path(__file__).resolve().parents[1] / "out"
    )
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.to(device).eval()

    with torch.no_grad():
        tokens = tokenizer([args.query]).to(device)
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb.cpu().numpy().astype(np.float32)

    index = faiss.read_index(str(args.src / "faiss_clip.index"))
    id_map = json.loads((args.src / "id_map.json").read_text(encoding="utf-8"))

    scores, rows = index.search(text_emb, args.topk)
    for score, row in zip(scores[0], rows[0]):
        meta = id_map[row]
        print(
            f"{score:.4f}  {meta['video_id']}  frame_idx={meta['frame_idx']}  s3_key={meta['s3_key']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
