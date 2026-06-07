#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/ader/Documents/App"
ZIP_EGS="$ROOT/external/ZipVoice/egs/zipvoice"
LOG_DIR="$ROOT/distillation/taiwan_mandarin_low_r/logs"
LOG_FILE="$LOG_DIR/zipvoice_cosy_golden_daily_500_decoder_train.log"
EXP_DIR="exp/zipvoice_cosy_golden_daily_500_decoder_v1"

mkdir -p "$LOG_DIR"
cd "$ZIP_EGS"

exec >>"$LOG_FILE" 2>&1

trap 'code=$?; echo "ZipVoice golden daily fine-tune exiting with code=${code}: $(date)"' EXIT
trap 'echo "ERROR at line ${LINENO}: $(date)"' ERR

echo "============================================================"
echo "ZipVoice golden daily fine-tune started: $(date)"
echo "cwd=$ZIP_EGS"
echo "exp_dir=$EXP_DIR"
echo "mode=decoder-only fine-tune; max_duration=10; num_iters=600"
echo "============================================================"

export PYTHONPATH="../../:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export PYTORCH_ENABLE_MPS_FALLBACK=1

nice -n 10 "$ROOT/.venv-zipvoice/bin/python" -m zipvoice.bin.train_zipvoice \
  --world-size 1 \
  --use-fp16 0 \
  --finetune 1 \
  --base-lr 0.00005 \
  --num-iters 600 \
  --save-every-n 100 \
  --keep-last-k 6 \
  --average-period 25 \
  --valid-by-epoch false \
  --save-epoch-checkpoints false \
  --max-duration 10 \
  --max-len 10 \
  --trainable-module-prefixes fm_decoder \
  --model-config download/zipvoice/model.json \
  --checkpoint download/zipvoice/model.pt \
  --tokenizer emilia \
  --lang default \
  --token-file download/zipvoice/tokens.txt \
  --dataset custom \
  --train-manifest data/fbank/custom-cosy-golden-daily_cuts_train.jsonl.gz \
  --dev-manifest data/fbank/custom-cosy-golden-daily_cuts_dev.jsonl.gz \
  --num-buckets 4 \
  --num-workers 0 \
  --exp-dir "$EXP_DIR"

"$ROOT/.venv-zipvoice/bin/python" - <<'PY'
from pathlib import Path
import torch

exp = Path("exp/zipvoice_cosy_golden_daily_500_decoder_v1")
preferred = exp / "checkpoint-600.pt"
if preferred.exists():
    src = preferred
else:
    checkpoints = sorted(exp.glob("checkpoint-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    if not checkpoints:
        raise SystemExit("No checkpoint-*.pt files found")
    src = checkpoints[-1]

state = torch.load(src, map_location="cpu", weights_only=False)
out = exp / f"{src.stem}-model-only.pt"
torch.save({"model": state["model"]}, out)
print(f"model_only={out} size_mb={out.stat().st_size / 1024 / 1024:.1f}")
PY

echo "ZipVoice golden daily fine-tune finished: $(date)"
