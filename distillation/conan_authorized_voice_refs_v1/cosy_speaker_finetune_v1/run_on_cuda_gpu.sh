#!/usr/bin/env bash
set -euo pipefail

# Run from /Users/ader/Documents/App on a CUDA Linux host with CosyVoice deps.
export PYTHONPATH=external/CosyVoice

MODEL_DIR="external/CosyVoice/pretrained_models/CosyVoice2-0.5B"
EXP_DIR="distillation/conan_authorized_voice_refs_v1/cosy_speaker_finetune_v1/exp/haibara_sft_llm"
TRAIN_LIST="distillation/conan_authorized_voice_refs_v1/cosy_speaker_finetune_v1/parquet/train/data.list"
DEV_LIST="distillation/conan_authorized_voice_refs_v1/cosy_speaker_finetune_v1/parquet/dev/data.list"

mkdir -p "$EXP_DIR"

# Before serious training, copy cosyvoice2.yaml and change:
#   padding.use_spk_embedding: true
#   train_conf.max_epoch: 20-100 for this tiny smoke set
#   train_conf.log_interval: 1
#   train_conf.save_per_step: 20
CONFIG="distillation/conan_authorized_voice_refs_v1/cosy_speaker_finetune_v1/cosyvoice2_haibara_sft.yaml"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  external/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config "$CONFIG" \
  --train_data "$TRAIN_LIST" \
  --cv_data "$DEV_LIST" \
  --model llm \
  --checkpoint "$MODEL_DIR/llm.pt" \
  --model_dir "$EXP_DIR" \
  --tensorboard_dir "$EXP_DIR/tensorboard" \
  --ddp.dist_backend nccl \
  --num_workers 0 \
  --prefetch 10
