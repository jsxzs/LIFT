#!/bin/bash
# Stage 1 -- dense-layout teacher: supervised fine-tuning with the per-frame layout canvas on EVERY frame
# (--layout_drop_rate 0.0). Starts from the camera-control model; the trainer widens patch_embedding from
# 32 to 48 input channels on load (the 16 new layout channels start from a fresh init).
# Recipe of the released teacher: 2 GPU x batch 8 x accum 2 (global 32), lr 1e-4 constant after 100 warm-up
# steps, 4000 steps, 352x640 x 81 frames, checkpoint + validation every 1000 steps.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_NAME=${MODEL_NAME:-models/Wan2.1-Fun-V1.1-1.3B-Control-Camera}
INIT=${INIT:-$MODEL_NAME/diffusion_pytorch_model.safetensors}   # or a camera-control fine-tune of it
TRAIN_CSV=${TRAIN_CSV:?path to the training CSV}
VAL_CSV=${VAL_CSV:-}
DATA_ROOT=${DATA_ROOT:-}            # only if the CSV's relative paths are not relative to the CSV's own directory
OUT=${OUT:-output/stage1_dense_layout}
NUM_GPUS=${NUM_GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
STEPS=${STEPS:-4000}
CKPT_STEPS=${CKPT_STEPS:-1000}

accelerate launch --num_processes "$NUM_GPUS" --num_machines 1 --mixed_precision bf16 \
  scripts/train_wan21_camlayout.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --transformer_path "$INIT" \
  --train_data_meta "$TRAIN_CSV" ${VAL_CSV:+--validation_csv "$VAL_CSV" --validation_at_start} ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
  --video_sample_size 352 640 --video_sample_n_frames 81 --video_sample_stride 1 \
  --max_objs 5 --bbox_area_threshold 0.001 --layout_drop_rate 0.0 \
  --train_batch_size "$BATCH_SIZE" --gradient_accumulation_steps "$GRAD_ACCUM" --dataloader_num_workers 2 --dataloader_prefetch_factor 1 \
  --num_train_epochs 100 --max_train_steps "$STEPS" \
  --learning_rate 1e-4 --lr_scheduler constant_with_warmup --lr_warmup_steps 100 --lr_num_training_steps "$STEPS" --seed 42 \
  --output_dir "$OUT" \
  --checkpointing_steps "$CKPT_STEPS" --validation_steps "$CKPT_STEPS" --validation_epochs 1000000 --checkpoints_total_limit 10 \
  --gradient_checkpointing --mixed_precision bf16 --vae_mini_batch 32 \
  --max_grad_norm 0.05 --uniform_sampling --scheduler_shift 5.0 \
  --trainable_modules "." --report_to wandb --tracker_project_name LIFT --tracker_run_name dense_layout
