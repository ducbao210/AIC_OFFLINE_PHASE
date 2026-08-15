#!/usr/bin/env python3
"""Stage 13 — Build FAISS index từ out/clip_b32.f32.npy (đã L2-normalize).

Idempotent/resumable như mọi stage khác: ghi ingest_manifest với sentinel
video_id "__GLOBAL__" + fingerprint = sha256 của file .npy nguồn. Chạy lại
sẽ tự skip nếu file .npy không đổi và index đã tồn tại; dùng --force để build lại.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from aic.cli import base_parser  # noqa: E402
from aic.db import connect, init_schema, is_done, mark_done, transaction  # noqa: E402
from aic.utils import setup_logging, sha256_file  # noqa: E402

STAGE = "13_faiss_index"
SENTINEL = "__GLOBAL__"


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument(
        "--src", type=Path, default=Path(__file__).resolve().parents[1] / "out"
    )
    parser.add_argument("--index-type", choices=["flat", "ivf"], default="flat")
    parser.add_argument("--nlist", type=int, default=100)
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    npy_path = args.src / "clip_b32.f32.npy"
    out_path = args.src / "faiss_clip.index"
    fingerprint = sha256_file(npy_path)

    conn = connect()
    init_schema(conn)

    if (
        not args.force
        and out_path.exists()
        and is_done(conn, STAGE, SENTINEL, fingerprint)
    ):
        log.info(
            "faiss_clip.index đã build với fingerprint hiện tại — bỏ qua (--force để build lại)"
        )
        conn.close()
        return 0

    matrix = np.load(npy_path).astype(np.float32)
    dim = matrix.shape[1]

    if args.index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(
            quantizer, dim, args.nlist, faiss.METRIC_INNER_PRODUCT
        )
        index.train(matrix)

    index.add(matrix)
    faiss.write_index(index, str(out_path))

    with transaction(conn):
        mark_done(
            conn,
            STAGE,
            SENTINEL,
            fingerprint,
            {"n_vectors": index.ntotal, "dim": dim, "index_type": args.index_type},
        )

    log.info(
        "đã build FAISS index: %s vector, dim=%s -> %s", index.ntotal, dim, out_path
    )
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
