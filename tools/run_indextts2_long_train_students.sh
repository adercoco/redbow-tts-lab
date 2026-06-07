#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/ader/Documents/App"
DISTILL="$ROOT/distillation/taiwan_mandarin_low_r"
LOG_DIR="$DISTILL/logs"
PIPER_OUT="$DISTILL/students/piper_indextts2_phoneme_ids_long_v4_step10000"
MATCHA_OUT="$DISTILL/students/matcha_indextts2_pinyin_long_v4_step10000"

run_piper() {
  cd "$ROOT/external/piper1-gpl"
  export PYTHONPATH="$ROOT/external/piper1-gpl/src"

  "$ROOT/.venv-piper/bin/python" -m piper.train fit \
    --config "$DISTILL/students/piper_vits_phoneme_ids_step1000/config.yaml" \
    --trainer.default_root_dir "$PIPER_OUT" \
    --trainer.max_steps 10000 \
    --trainer.log_every_n_steps 25 \
    --data.csv_path "$DISTILL/datasets/piper_phoneme_ids_indextts2_500_v1/metadata.csv" \
    --data.cache_dir "$PIPER_OUT/cache" \
    --data.config_path "$PIPER_OUT/zh_TW-indextts2-piper-phoneme_ids-long_v4-step10000-low.onnx.json" \
    --data.voice_name indextts2_piper_phoneme_ids_step10000 \
    --data.audio_dir "$DISTILL/datasets/piper_phoneme_ids_indextts2_500_v1/wavs" \
    --data.phonemes_path "$DISTILL/datasets/piper_phoneme_ids_indextts2_500_v1/phonemes.json" \
    --ckpt_path "$DISTILL/students/piper_vits_phoneme_ids_step1000/checkpoints/epoch=3-step=1000.ckpt"
}

run_matcha() {
  cd "$ROOT/external/CosyVoice/third_party/Matcha-TTS"
  export PYTHONPATH="$ROOT/external/CosyVoice/third_party/Matcha-TTS"
  export PROJECT_ROOT="$ROOT/external/CosyVoice/third_party/Matcha-TTS"

  "$ROOT/.venv-matcha/bin/python" matcha/train.py \
    run_name=indextts2_pinyin_step10000 \
    tags="[indextts2,pinyin,phone-small]" \
    test=false \
    logger=tensorboard \
    paths.output_dir="$MATCHA_OUT" \
    hydra.run.dir="$MATCHA_OUT" \
    data.train_filelist_path="$DISTILL/datasets/matcha_pinyin_indextts2_500_v1/train.txt" \
    data.valid_filelist_path="$DISTILL/datasets/matcha_pinyin_indextts2_500_v1/valid.txt" \
    data.batch_size=8 \
    data.num_workers=0 \
    data.pin_memory=false \
    data.cleaners="[]" \
    data.data_statistics.mel_mean=-6.055769920349121 \
    data.data_statistics.mel_std=2.259850025177002 \
    trainer.accelerator=cpu \
    trainer.devices=1 \
    trainer.precision=32 \
    +trainer.max_steps=10000 \
    +trainer.num_sanity_val_steps=0 \
    trainer.max_epochs=-1 \
    callbacks.model_checkpoint.dirpath="$MATCHA_OUT/checkpoints" \
    callbacks.model_checkpoint.save_last=true \
    callbacks.model_checkpoint.every_n_epochs=20 \
    callbacks.model_checkpoint.save_top_k=3
}

start_one() {
  local name="$1"
  local log="$LOG_DIR/${name}_train_r3.log"
  local pidfile="$LOG_DIR/${name}.pid"

  mkdir -p "$LOG_DIR"
  nohup /usr/bin/env bash "$0" "$name" >"$log" 2>&1 &
  echo "$!" >"$pidfile"
  printf '%s %s %s\n' "$name" "$(cat "$pidfile")" "$log"
}

start_caffeinate_guard() {
  local pidfile="$LOG_DIR/long_train_caffeinate_guard.pid"
  local log="$LOG_DIR/long_train_caffeinate_guard.log"

  if [[ -f "$pidfile" ]] && ps -p "$(cat "$pidfile")" >/dev/null 2>&1; then
    printf 'caffeinate_guard %s %s\n' "$(cat "$pidfile")" "$log"
    return
  fi

  nohup /usr/bin/caffeinate -dimsu /bin/sleep 86400 >"$log" 2>&1 &
  echo "$!" >"$pidfile"
  printf 'caffeinate_guard %s %s\n' "$(cat "$pidfile")" "$log"
}

case "${1:-}" in
  piper_indextts2_phoneme_ids_long_v4_step10000)
    run_piper
    status=$?
    echo "PIPER_EXIT:$status"
    exit "$status"
    ;;
  matcha_indextts2_pinyin_long_v4_step10000)
    run_matcha
    status=$?
    echo "MATCHA_EXIT:$status"
    exit "$status"
    ;;
  start)
    start_caffeinate_guard
    start_one piper_indextts2_phoneme_ids_long_v4_step10000
    start_one matcha_indextts2_pinyin_long_v4_step10000
    ;;
  *)
    echo "usage: $0 start|piper_indextts2_phoneme_ids_long_v4_step10000|matcha_indextts2_pinyin_long_v4_step10000" >&2
    exit 2
    ;;
esac
