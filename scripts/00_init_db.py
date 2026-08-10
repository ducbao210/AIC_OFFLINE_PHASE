#!/usr/bin/env python3
"""Stage 00 — Tạo schema SQLite và bucket MinIO."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema  # noqa: E402
from aic.storage import ensure_buckets, get_client  # noqa: E402
from aic.utils import has_ffprobe, setup_logging  # noqa: E402


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    log = setup_logging(args.verbose)

    log.info("data root  : %s (%s)", CONFIG.data_root,
             "OK" if CONFIG.data_root.exists() else "KHÔNG TỒN TẠI")
    log.info("sqlite     : %s", CONFIG.sqlite_path)
    log.info("minio      : %s (secure=%s)", CONFIG.endpoint, CONFIG.secure)

    if not has_ffprobe():
        log.warning("Không tìm thấy ffprobe — stage 02 sẽ bỏ qua phần probe video.")

    conn = connect()
    init_schema(conn)
    tables = [
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
        )
    ]
    conn.close()
    log.info("schema sẵn sàng: %s", ", ".join(tables))

    try:
        ensure_buckets(get_client())
        log.info("buckets sẵn sàng: %s", ", ".join(CONFIG.buckets))
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Không kết nối được MinIO (%s): %s — SQLite vẫn dùng được, "
            "chạy các stage với --no-upload hoặc bật MinIO rồi chạy lại.",
            CONFIG.endpoint, exc.__class__.__name__,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
