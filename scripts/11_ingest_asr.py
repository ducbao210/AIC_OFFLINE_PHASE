#!/usr/bin/env python3
"""Stage 11 — ASR (Whisper) transcription từ audio video.

Trích xuất audio từ .mp4, chạy Whisper, lưu transcript vào bảng `transcripts`.
Mỗi dòng là 1 segment (có start_s, end_s, text).

Idempotent: kiểm tra ingest_manifest trước khi chạy lại.
"""

from __future__ import annotations

import sys
import tempfile
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.cli import base_parser, select_videos  # noqa: E402
from aic.config import CONFIG  # noqa: E402
from aic.db import connect, init_schema, mark_done, transaction  # noqa: E402
from aic.utils import find_video_files, setup_logging, video_group  # noqa: E402

STAGE = "11_asr"


def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 16000) -> bool:
    """Trích xuất audio 16kHz mono WAV từ video (dùng ffmpeg)."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-vn",                          # bỏ video stream
        "-acodec", "pcm_s16le",         # PCM 16-bit
        "-ar", str(sample_rate),        # 16kHz
        "-ac", "1",                     # mono
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        import logging
        logging.getLogger("aic").error("ffmpeg audio extraction failed: %s", exc.stderr.decode()[:200])
        return False


def run_whisper(audio_path: Path, model_name: str, language: str, device: str) -> list[dict] | None:
    """Chạy Whisper, trả về list segment dicts hoặc None nếu lỗi."""
    try:
        import whisper
    except ImportError:
        import logging
        logging.getLogger("aic").error(
            "Whisper chưa được cài đặt. Chạy: pip install openai-whisper"
        )
        return None

    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        str(audio_path),
        language=language,
        verbose=False,
    )
    segments = []
    for i, seg in enumerate(result.get("segments", [])):
        segments.append({
            "segment_index": i,
            "start_s": float(seg["start"]),
            "end_s": float(seg["end"]),
            "text": seg["text"].strip(),
        })
    return segments


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--model", default=CONFIG.asr_model,
                        help="Tên Whisper model (tiny, base, small, medium, large-v3).")
    parser.add_argument("--language", default=CONFIG.asr_language,
                        help="Mã ngôn ngữ (vi, en, ...).")
    parser.add_argument("--device", default=CONFIG.asr_device,
                        choices=["cpu", "cuda"], help="Thiết bị chạy model.")
    parser.add_argument("--no-ffmpeg", action="store_true",
                        help="Bỏ qua trích xuất audio (dùng khi đã có sẵn audio file).")
    args = parser.parse_args()
    log = setup_logging(args.verbose)

    videos = find_video_files(CONFIG.data_root)
    conn = connect()
    init_schema(conn)
    targets = select_videos(args, list(videos), conn, STAGE)
    log.info("ASR: %d/%d video (model=%s, lang=%s, device=%s)",
             len(targets), len(videos), args.model, args.language, args.device)

    if args.dry_run or not targets:
        conn.close()
        return 0

    total_segments = 0
    for video_id in targets:
        video_path = videos[video_id]
        log.info("[%s] trích xuất audio...", video_id)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = Path(tmp.name)

        try:
            if not args.no_ffmpeg:
                ok = extract_audio(video_path, audio_path)
                if not ok:
                    mark_done(conn, STAGE, video_id, None,
                              {"error": "ffmpeg extraction failed"}, status="error")
                    continue

            segments = run_whisper(audio_path, args.model, args.language, args.device)
            if segments is None:
                mark_done(conn, STAGE, video_id, None,
                          {"error": "Whisper failed"}, status="error")
                continue

            if args.dry_run:
                log.info("[%s] %d segments", video_id, len(segments))
                continue

            with transaction(conn):
                conn.execute(
                    "INSERT OR IGNORE INTO videos (video_id, group_id) VALUES (?, ?)",
                    (video_id, video_group(video_id)),
                )
                conn.execute("DELETE FROM transcripts WHERE video_id = ?", (video_id,))
                conn.executemany(
                    """
                    INSERT INTO transcripts (video_id, segment_index, start_s, end_s, text, model_name, language)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (video_id, seg["segment_index"], seg["start_s"], seg["end_s"],
                         seg["text"], args.model, args.language)
                        for seg in segments
                    ],
                )
                mark_done(conn, STAGE, video_id, None,
                          {"n_segments": len(segments), "model": args.model})

            total_segments += len(segments)
            log.info("[%s] %d segments", video_id, len(segments))

        finally:
            if audio_path.exists():
                audio_path.unlink()

    log.info("ASR hoàn tất: %d segments tổng cộng", total_segments)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())