# AIC 2026 — Offline Ingestion Pipeline (SQLite + MinIO)

Chuyển toàn bộ bộ dữ liệu BTC (`AIC_26_DATA`) đang nằm trên Drive cá nhân thành:

- **MinIO (S3)** — lưu *bytes*: video `.mp4`, keyframe `.jpg`, file `.npy` CLIP features gốc.
- **SQLite** — lưu *con trỏ + metadata + index*: videos, keyframes, map-keyframes, objects,
  media-info, CLIP feature offsets, và bảng FTS5 để full-text search tiếng Việt.

Nguyên tắc: **mọi thứ quy chiếu về `frame_idx` tuyệt đối của video gốc** (lấy từ
`map-keyframes-aic25-b1/*.csv`). Mọi stage đều **idempotent + resumable** (ghi `ingest_manifest`,
chạy lại chỉ xử lý phần còn thiếu).

---

## 1. Cấu trúc dữ liệu nguồn

```
AIC_26_DATA/
├── Videos_L21_a/video/L21_V001.mp4 ...        # mp4 (có audio)
├── Keyframes_L21/keyframes/L21_V001/0001.jpg  # jpg
├── map-keyframes-aic25-b1/map-keyframes/L21_V001.csv
├── objects-aic25-b1/objects/L21_V001/0001.json
├── media-info-aic25-b1/media-info/L21_V001.json
└── clip-features-32-aic25-b1/clip-features-32/L21_V001.npy
```

Các thư mục `Keyframes_L26_a..e`, `Videos_L26_a..e` chỉ là shard — script tự gộp theo `video_id`.

## 2. Cài đặt

```bash
cd pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # sửa AIC_DATA_ROOT + thông tin MinIO
```

Chạy MinIO local:

```bash
docker compose -f docker-compose.minio.yml up -d
# console: http://localhost:9001  (minioadmin / minioadmin)
```

## 3. Chạy pipeline

```bash
bash run_all.sh
```

Hoặc từng bước (mỗi script chạy độc lập được, đều có `--help`):

| Script | Việc làm |
|---|---|
| `scripts/00_init_db.py` | Tạo schema SQLite + bucket MinIO |
| `scripts/01_scan_dataset.py` | Quét cây thư mục, dựng bảng `assets` (inventory mọi file) |
| `scripts/02_ingest_videos.py` | `ffprobe` (fps hữu tỉ, n_frames…) + sha256 + upload mp4 |
| `scripts/03_ingest_map_keyframes.py` | Nạp CSV → bảng `keyframes` (n, pts_time, fps, frame_idx) |
| `scripts/04_ingest_keyframes.py` | Upload ảnh jpg lên MinIO, ghi `s3_key`, kích thước, pHash |
| `scripts/05_ingest_objects.py` | Nạp JSON Faster R-CNN → `objects` (lọc theo score) |
| `scripts/06_ingest_clip_features.py` | Upload `.npy`, ghi `clip_features` (offset theo keyframe) |
| `scripts/07_ingest_media_info.py` | Nạp metadata YouTube → `media_info` + `media_keywords` |
| `scripts/08_build_fts.py` | Dựng `documents` (title+desc+keywords+objects) + FTS5 |
| `scripts/09_verify.py` | Kiểm tra toàn vẹn: đếm, khớp keyframe ↔ CSV ↔ npy, mẫu ngẫu nhiên |
| `scripts/10_export_index.py` | Xuất ma trận CLIP đã chuẩn hoá + `id_map.json` cho FAISS |

Ví dụ chạy lại chỉ 1 batch:

```bash
python scripts/04_ingest_keyframes.py --videos L21_V001 L21_V002 --workers 16
python scripts/05_ingest_objects.py --min-score 0.3 --limit-per-frame 20
```

## 4. Layout trên MinIO

```
aic-raw/     videos/{video_id}.mp4
aic-frames/  keyframes/{video_id}/{name}.jpg
aic-feats/   clip-features-32/{video_id}.npy
```

## 5. Truy vấn thử

```bash
sqlite3 data/aic.sqlite "
  SELECT k.video_id, k.frame_idx, o.entity, o.score
  FROM objects o JOIN keyframes k ON k.id = o.keyframe_id
  WHERE o.entity = 'Tomato' AND o.score > 0.7 LIMIT 10;"

sqlite3 data/aic.sqlite "
  SELECT video_id, snippet(documents_fts, 1, '[', ']', '…', 12)
  FROM documents_fts WHERE documents_fts MATCH 'năng lượng tích cực' LIMIT 5;"
```
