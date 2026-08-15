"""Tiện ích dùng chung: log, checksum, ffprobe, parse video_id, quét cây thư mục."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

SHARD_RE = re.compile(
    r"^(?:Keyframes|Videos|objects|media-info|clip-features)_(L\d{2}_[a-z])$"
)


VIDEO_ID_RE = re.compile(r"^L\d{2}_V\d{3}$")

T = TypeVar("T")
R = TypeVar("R")


def setup_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("aic")


LOG = logging.getLogger("aic")


def dataset_shard(path: Path) -> str | None:
    """Keyframes_L26_a -> L26_a."""
    for part in path.parts:
        match = SHARD_RE.match(part)
        if match:
            return match.group(1)
    return None


def is_video_id(name: str) -> bool:
    return bool(VIDEO_ID_RE.match(name))


def video_group(video_id: str) -> str:
    """L21_V001 -> L21"""
    return video_id.split("_", 1)[0]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def nfc(text: str | None) -> str | None:
    if text is None:
        return None
    return unicodedata.normalize("NFC", text)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- discovery


def iter_dirs(root: Path, pattern: str) -> Iterator[Path]:
    if not root.exists():
        return iter(())
    return (p for p in sorted(root.glob(pattern)) if p.is_dir())


def find_video_files(data_root: Path) -> dict[str, Path]:
    """Videos_L*_?/video/*.mp4  ->  {video_id: path}"""
    found: dict[str, Path] = {}
    for shard in iter_dirs(data_root, "Videos_L*"):
        for mp4 in sorted(shard.rglob("*.mp4")):
            found.setdefault(mp4.stem, mp4)
    return found


def find_keyframe_dirs(
    data_root: Path,
    shards: Sequence[str] | None = None,
) -> dict[str, Path]:
    found: dict[str, Path] = {}
    allowed = set(shards or [])

    for shard in iter_dirs(data_root, "Keyframes_L*"):
        shard_id = dataset_shard(shard)

        if allowed and shard_id not in allowed:
            continue

        base = shard / "keyframes"
        base = base if base.is_dir() else shard

        for vdir in sorted(p for p in base.iterdir() if p.is_dir()):
            if is_video_id(vdir.name):
                found.setdefault(vdir.name, vdir)

    return found


def find_map_keyframes(data_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for csv in sorted(data_root.glob("map-keyframes*/**/*.csv")):
        if is_video_id(csv.stem):
            found.setdefault(csv.stem, csv)
    return found


def find_media_info(data_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for js in sorted(data_root.glob("media-info*/**/*.json")):
        if is_video_id(js.stem):
            found.setdefault(js.stem, js)
    return found


def find_object_dirs(data_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for shard in iter_dirs(data_root, "objects*"):
        base = shard / "objects"
        base = base if base.is_dir() else shard
        for vdir in sorted(p for p in base.iterdir() if p.is_dir()):
            if is_video_id(vdir.name):
                found.setdefault(vdir.name, vdir)
    return found


def find_clip_features(data_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for npy in sorted(data_root.glob("clip-features*/**/*.npy")):
        if is_video_id(npy.stem):
            found.setdefault(npy.stem, npy)
    return found


# ---------------------------------------------------------------- ffprobe


def has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def ffprobe(path: Path) -> dict:
    """Trả về dict thông tin video; fps giữ dạng phân số (fps_num/fps_den)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height,codec_name,duration",
        "-show_entries",
        "format=duration,size,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    raw = json.loads(out)
    stream = (raw.get("streams") or [{}])[0]
    fmt = raw.get("format") or {}

    rate = stream.get("r_frame_rate") or "0/1"
    num, _, den = rate.partition("/")
    fps_num, fps_den = int(num or 0), int(den or 1) or 1

    duration = stream.get("duration") or fmt.get("duration")
    nb_frames = stream.get("nb_frames")
    duration_s = float(duration) if duration else None
    if nb_frames:
        n_frames = int(nb_frames)
    elif duration_s and fps_num:
        n_frames = round(duration_s * fps_num / fps_den)
    else:
        n_frames = None

    # audio stream?
    acmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    astreams = json.loads(
        subprocess.run(acmd, capture_output=True, text=True, check=True).stdout
    ).get("streams") or [{}]
    audio = astreams[0]

    return {
        "fps_num": fps_num,
        "fps_den": fps_den,
        "n_frames": n_frames,
        "duration_s": duration_s,
        "width": int(stream["width"]) if stream.get("width") else None,
        "height": int(stream["height"]) if stream.get("height") else None,
        "codec": stream.get("codec_name"),
        "size_bytes": int(fmt["size"]) if fmt.get("size") else path.stat().st_size,
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": (
            int(audio["sample_rate"]) if audio.get("sample_rate") else None
        ),
        "audio_channels": int(audio["channels"]) if audio.get("channels") else None,
    }


# ---------------------------------------------------------------- parallel


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    workers: int = 8,
    desc: str = "",
) -> Iterator[R]:
    """Chạy song song, yield kết quả khi xong; lỗi được log và bỏ qua."""
    if workers <= 1:
        for item in _progress(items, desc):
            try:
                yield fn(item)
            except Exception as exc:  # noqa: BLE001
                LOG.error("%s failed: %s", item, exc)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for fut in _progress(as_completed(futures), desc, total=len(futures)):
            try:
                yield fut.result()
            except Exception as exc:  # noqa: BLE001
                LOG.error("%s failed: %s", futures[fut], exc)


def _progress(iterable: Iterable, desc: str, total: int | None = None):
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc or None, total=total, unit="it")
    except Exception:  # noqa: BLE001
        return iterable
