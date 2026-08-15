#!/usr/bin/env python3
"""Stage 13 — Build FAISS index từ out/clip_b32.f32.npy (đã L2-normalize)."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np
from aic.cli import base_parser
from aic.utils import setup_logging


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument(
        "--src", type=Path, default=Path(__file__).resolve().parents[1] / "out"
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf"],
        default="flat",
        help="flat = exact (đủ nhanh tới vài trăm nghìn vector); ivf = approximate cho scale lớn",
    )
    parser.add_argument(
        "--nlist", type=int, default=100, help="số cluster nếu dùng ivf"
    )
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    matrix = np.load(args.src / "clip_b32.f32.npy").astype(np.float32)
    dim = matrix.shape[1]

    if args.index_type == "flat":
        index = faiss.IndexFlatIP(dim)  # cosine = inner product vì vector đã normalize
    else:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(
            quantizer, dim, args.nlist, faiss.METRIC_INNER_PRODUCT
        )
        index.train(matrix)

    index.add(matrix)
    out_path = args.src / "faiss_clip.index"
    faiss.write_index(index, str(out_path))
    log.info("đã build FAISS index: %s vector, %s → %s", index.ntotal, dim, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
