#!/usr/bin/env python3
"""Stage 12 — Visual captioning (BLIP-2) trên keyframe.

Dùng BLIP-2 (hoặc BLIP) để sinh mô tả tiếng Anh/Việt cho từng keyframe.
Lưu vào bảng `captions`, liên kết qua keyframe_id.

Idempotent: kiểm tra ingest_manifest trước khi chạy lại.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser, select_videos  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema, mark_done, transaction  # noqa: E402
from aic.utils import find_keyframe_dirs, setup_logging, video_group  # noqa: E402

STAGE = "12_captions"


def load_caption_model(model_name: str, device: str):
    """Load BLIP-2 model. Trả về (model, processor) hoặc (None, None) nếu lỗi."""
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
    except ImportError:
        import logging

        logging.getLogger("aic").error(
            "transformers chưa được cài đặt. Chạy: pip install transformers torch"
        )
        return None, None

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, processor


def generate_caption(
    model,
    processor,
    image_path: Path,
    device: str,
    max_new_tokens: int,
    prompt: str | None = None,
) -> str | None:
    """Sinh caption cho 1 ảnh. Trả về text hoặc None nếu lỗi."""
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        if prompt:
            inputs = processor(image, text=prompt, return_tensors="pt").to(device)
        else:
            inputs = processor(image, return_tensors="pt").to(device)

        import torch

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        return caption
    except Exception as exc:
        import logging

        logging.getLogger("aic").warning("caption failed for %s: %s", image_path, exc)
        return None


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument(
        "--model",
        default=CONFIG.caption_model,
        help="Tên model BLIP-2 trên HuggingFace.",
    )
    parser.add_argument(
        "--device",
        default=CONFIG.caption_device,
        choices=["cpu", "cuda"],
        help="Thiết bị chạy model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=CONFIG.caption_max_new_tokens,
        help="Số token tối đa cho caption.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt prefix (vd: 'Question: What is in this image? Answer:').",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Chỉ caption N keyframe đầu tiên mỗi video (debug).",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=None,
        help="Giới hạn số keyframe caption mỗi video (phân bố đều).",
    )
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    kf_dirs = find_keyframe_dirs(CONFIG.data_root, args.shards)
    conn = connect()
    init_schema(conn)
    targets = select_videos(args, list(kf_dirs), conn, STAGE)
    log.info(
        "captioning: %d/%d video (model=%s, device=%s)",
        len(targets),
        len(kf_dirs),
        args.model,
        args.device,
    )

    if args.dry_run or not targets:
        conn.close()
        return 0

    log.info("Đang load model %s ...", args.model)
    model, processor = load_caption_model(args.model, args.device)
    if model is None:
        log.error("Không load được model captioning.")
        conn.close()
        return 1

    total_captions = 0
    for video_id in targets:
        images = sorted(
            p
            for p in kf_dirs[video_id].iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not images:
            log.warning("%s: không có ảnh", video_id)
            continue

        # Lấy danh sách keyframe từ DB để khớp
        kf_rows = conn.execute(
            "SELECT id, n, frame_idx, file_name FROM keyframes WHERE video_id = ? ORDER BY n",
            (video_id,),
        ).fetchall()

        # Map: file_name -> keyframe_id
        file_to_kf = {}
        for row in kf_rows:
            if row["file_name"]:
                file_to_kf[row["file_name"]] = (row["id"], row["frame_idx"])

        # Map: thứ tự n -> keyframe_id (fallback nếu không khớp tên file)
        n_to_kf = {row["n"]: (row["id"], row["frame_idx"]) for row in kf_rows}

        # Chọn ảnh cần caption
        if args.frames_per_video and len(images) > args.frames_per_video:
            step = max(1, len(images) // args.frames_per_video)
            selected = images[::step][: args.frames_per_video]
        else:
            selected = images
        if args.sample:
            selected = selected[: args.sample]

        rows: list[tuple] = []
        for pos, img_path in enumerate(selected):
            # Khớp keyframe
            target = file_to_kf.get(img_path.name)
            if target is None:
                n = pos + 1  # fallback theo thứ tự
                target = n_to_kf.get(n)
            if target is None:
                continue

            keyframe_id, frame_idx = target
            caption = generate_caption(
                model, processor, img_path, args.device, args.max_tokens, args.prompt
            )
            if caption:
                rows.append((keyframe_id, video_id, frame_idx, caption, args.model))

        if args.dry_run:
            log.info("[%s] %d/%d captions", video_id, len(rows), len(selected))
            continue

        with transaction(conn):
            conn.execute(
                "INSERT OR IGNORE INTO videos (video_id, group_id) VALUES (?, ?)",
                (video_id, video_group(video_id)),
            )
            conn.executemany(
                """
                INSERT INTO captions (keyframe_id, video_id, frame_idx, caption_text, model_name)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(keyframe_id) DO UPDATE SET
                    caption_text = excluded.caption_text,
                    model_name   = excluded.model_name
                """,
                rows,
            )
            mark_done(
                conn,
                STAGE,
                video_id,
                None,
                {"n_captions": len(rows), "model": args.model},
            )

        total_captions += len(rows)
        log.info("[%s] %d captions", video_id, len(rows))

    log.info("captioning hoàn tất: %d captions tổng cộng", total_captions)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
