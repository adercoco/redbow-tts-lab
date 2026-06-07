#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/ader/Documents/App"
cd "$ROOT"

LOG_DIR="distillation/taiwan_mandarin_low_r/logs"
LOG_FILE="$LOG_DIR/golden_cosy_teacher_500_background.log"
mkdir -p "$LOG_DIR"

exec >>"$LOG_FILE" 2>&1

trap 'code=$?; echo "Golden Cosy teacher background run exiting with code=${code}: $(date)"' EXIT
trap 'echo "ERROR at line ${LINENO}: $(date)"' ERR

echo "============================================================"
echo "Golden Cosy teacher background run started: $(date)"
echo "cwd=$ROOT"
echo "============================================================"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

for limit in 350 400 450 500; do
  echo
  echo "---- target limit=${limit} started: $(date) ----"
  echo "running generate_cosy_teacher_corpus.py --limit ${limit}"
  nice -n 10 .venv-cosyvoice/bin/python tools/generate_cosy_teacher_corpus.py \
    --texts distillation/taiwan_mandarin_low_r/distill_texts_golden_daily_v1.jsonl \
    --limit "$limit" \
    --pack-id clear_best2_7s \
    --out distillation/taiwan_mandarin_low_r/teacher_cosy_clear_best2_golden_daily_500_v1 \
    --resume \
    --quiet-skips

  .venv-cosyvoice/bin/python tools/build_golden_distillation_progress_report.py
  .venv-cosyvoice/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("distillation/taiwan_mandarin_low_r/teacher_cosy_clear_best2_golden_daily_500_v1/manifest.json")
rows = json.loads(p.read_text(encoding="utf-8"))
ok = [r for r in rows if r.get("status") == "ok"]
print(f"progress_ok={len(ok)} last={ok[-1]['id'] if ok else 'none'}")
PY
  echo "---- target limit=${limit} finished: $(date) ----"
done

echo
echo "Golden Cosy teacher background run finished: $(date)"
