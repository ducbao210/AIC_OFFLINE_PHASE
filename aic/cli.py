"""Argument parser dùng chung cho mọi script trong scripts/."""

from __future__ import annotations

import argparse
import sqlite3
from typing import Sequence

from .config import CONFIG


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--videos", nargs="*", default=None,
        help="Chỉ xử lý các video_id này (mặc định: tất cả).",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Chỉ xử lý các nhóm L21 L22 ...",
    )
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số video.")
    parser.add_argument("--workers", type=int, default=CONFIG.workers)
    parser.add_argument("--force", action="store_true", help="Bỏ qua manifest, làm lại từ đầu.")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in ra, không ghi.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def select_videos(
    args: argparse.Namespace,
    candidates: Sequence[str],
    conn: sqlite3.Connection | None = None,
    stage: str | None = None,
) -> list[str]:
    """Lọc danh sách video theo --videos/--groups/--limit và manifest."""
    selected = sorted(set(candidates))

    if args.videos:
        wanted = set(args.videos)
        selected = [v for v in selected if v in wanted]
    if args.groups:
        groups = set(args.groups)
        selected = [v for v in selected if v.split("_", 1)[0] in groups]

    if conn is not None and stage and not args.force:
        from .db import done_videos

        done = done_videos(conn, stage)
        selected = [v for v in selected if v not in done]

    if args.limit:
        selected = selected[: args.limit]
    return selected
