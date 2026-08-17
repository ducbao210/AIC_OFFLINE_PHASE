#!/usr/bin/env python3
"""Build the second, independent FAISS index from text embeddings only.

Input: text_embeddings.f32.npy + text_embedding_metadata.json
Output: faiss_text.index + text_faiss_manifest.json
No search or ranking logic is implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--index-type", choices=["flat", "ivf"], default="flat")
    ap.add_argument("--nlist", type=int, default=256)
    args = ap.parse_args()

    emb_path = args.src / "text_embeddings.f32.npy"
    metadata_path = args.src / "text_embedding_metadata.json"
    index_path = args.src / "faiss_text.index"
    if not emb_path.exists() or not metadata_path.exists():
        raise SystemExit(
            "Missing text_embeddings.f32.npy or text_embedding_metadata.json"
        )

    matrix = np.load(emb_path, mmap_mode="r").astype("float32")
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise SystemExit(f"Invalid embedding matrix shape: {matrix.shape}")
    dim = int(matrix.shape[1])
    if not np.isfinite(matrix[: min(1000, len(matrix))]).all():
        raise SystemExit("Embedding matrix contains non-finite values")

    if args.index_type == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(np.asarray(matrix, dtype="float32"))
    else:
        nlist = min(args.nlist, max(1, int(np.sqrt(len(matrix)))))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        train_sample = np.asarray(
            matrix[: min(len(matrix), max(10_000, nlist * 40))], dtype="float32"
        )
        index.train(train_sample)
        index.add(np.asarray(matrix, dtype="float32"))

    faiss.write_index(index, str(index_path))
    manifest = {
        "index": index_path.name,
        "source_embeddings": emb_path.name,
        "metadata": metadata_path.name,
        "model_manifest": "text_embedding_manifest.json",
        "index_type": args.index_type,
        "metric": "inner_product_on_l2_normalized_vectors",
        "dimension": dim,
        "n_vectors": int(index.ntotal),
        "source_sha256": sha256(emb_path),
        "search_not_implemented_in_this_stage": True,
    }
    (args.src / "text_faiss_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
