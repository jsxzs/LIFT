#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_NAME=${MODEL_NAME:-models/Wan2.1-Fun-V1.1-1.3B-Control-Camera}
TEACHER=${TEACHER:?stage-1 checkpoint: <ckpt>/transformer/diffusion_pytorch_model.safetensors}
TRAIN_CSV=${TRAIN_CSV:?path to the training CSV}
VAL_CSV=${VAL_CSV:-}
DATA_ROOT=${DATA_ROOT:-}            # only if the CSV's relative paths are not relative to the CSV's own directory
OUT=${OUT:-output/stage2_opsd}
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
STEPS=${STEPS:-500}
CKPT_STEPS=${CKPT_STEPS:-100}

accelerate launch --num_processes "$NUM_GPUS" --num_machines 1 --mixed_precision bf16 \
  scripts/train_wan21_camlayout_opd.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --transformer_path "$TEACHER" --teacher_transformer_path "$TEACHER" \
  --train_data_meta "$TRAIN_CSV" ${VAL_CSV:+--validation_csv "$VAL_CSV"} ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
  --video_sample_size 352 640 --video_sample_n_frames 81 --video_sample_stride 1 \
  --max_objs 5 --bbox_area_threshold 0.001 --layout_drop_rate 0.0 \
  --student_num_keep_frames 1 \
  --student_cond_mix_prob 0.7 \
  --distill_rollout_steps 10 --distill_num_sampled_states 10 \
  --distill_num_steps 50 --distill_loss_type x0 --sft_loss_weight 0.1 \
  --train_batch_size "$BATCH_SIZE" --gradient_accumulation_steps 1 --dataloader_num_workers 2 --dataloader_prefetch_factor 1 \
  --num_train_epochs 1 --max_train_steps "$STEPS" \
  --learning_rate 5e-5 --lr_scheduler constant_with_warmup --lr_warmup_steps 10 --seed 42 \
  --output_dir "$OUT" \
  --checkpointing_steps "$CKPT_STEPS" --validation_steps "$CKPT_STEPS" --validation_epochs 1 --checkpoints_total_limit 10 \
  --gradient_checkpointing --mixed_precision bf16 --vae_mini_batch 32 \
  --max_grad_norm 1.0 --initial_grad_norm_ratio 1 --abnormal_norm_clip_start 1000000000 --uniform_sampling \
  --trainable_modules "." --report_to wandb --tracker_project_name LIFT --tracker_run_name stage2_opsd
