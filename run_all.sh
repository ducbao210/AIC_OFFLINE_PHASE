#!/usr/bin/env bash
# Chạy toàn bộ offline pipeline theo thứ tự. Mọi stage đều resumable.
set -euo pipefail

cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
ARGS=("$@")   # ví dụ: bash run_all.sh --groups L21 L22

run() {
  echo ""
  echo "=============================================================="
  echo ">>> $1"
  echo "=============================================================="
  "$PY" "scripts/$1" "${ARGS[@]}"
}

"$PY" scripts/00_init_db.py
run 01_scan_dataset.py
run 02_ingest_videos.py
run 03_ingest_map_keyframes.py
run 04_ingest_keyframes.py
run 05_ingest_objects.py
run 06_ingest_clip_features.py
run 07_ingest_media_info.py
run 11_ingest_asr.py
run 12_ingest_captions.py
run 08_build_fts.py
run 09_verify.py
run 10_export_index.py

echo ""
echo "Hoàn tất. SQLite: ${AIC_SQLITE_PATH:-./data/aic.sqlite}"