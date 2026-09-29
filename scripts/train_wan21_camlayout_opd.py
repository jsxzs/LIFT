"""On-policy distillation (OPD) trainer for Wan2.1-Fun camera + LAYOUT control.

Ported from CamTraj-VideoGen's diffsynth-based ``train_camlayout_dense_rgb.py`` on top of the
validated VideoX-Fun ``train_wan21_camlayout.py`` data/model path. The goal is a *dense -> sparse*
layout distillation, re-implemented in this codebase to rule out a codebase-specific cause of the
blurring seen with the diffsynth implementation.

Setup (teacher and student share the SAME architecture; they differ ONLY in their weights and in the
layout canvas fed via ``y``):

  * TEACHER: a frozen copy conditioned on the DENSE per-frame bbox canvas (all frames carry layout).
  * STUDENT: the trainable model conditioned on the SPARSE canvas (only ``--student_num_keep_frames``
    frames, last frame always kept). Initialized from the teacher's weights by default.

Loss (on-policy, x0-matching): the student rolls out its own coarse trajectory under its SPARSE
conditioning; at randomly sampled visited noisy states we reconstruct x0 from both the frozen teacher
(dense) and the trainable student (sparse) and match them with MSE. An optional ``--sft_loss_weight``
anchors the student to ground-truth data with a plain flow-matching loss.

Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
from contextlib import nullcontext
import gc
import logging
import math
import os
import pickle
import shutil
import sys
import random

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data.bucket_sampler import RandomSampler
from videox_fun.data.dataset_camlayout import (WanFunCameraControlDataset,
                                               WanFunCamLayoutControlDataset)
from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                               WanTransformer3DModel)
# from videox_fun.pipeline import WanI2VPipeline, WanPipeline
from videox_fun.pipeline import WanCamLayoutPipeline
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.utils import (save_videos_grid)

if is_wandb_available():
    import wandb

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, walking backwards"
)


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def _expand_dit_patch_embedding_input_dim(dit, extra_in_dim):
    """Grow the DiT patch_embedding to accept ``extra_in_dim`` more input channels (the VAE-encoded
    bbox-canvas latent that is channel-concatenated to y). Pretrained channels are copied; the new
    channels keep the fresh Conv3d init. Ported from CamTraj-VideoGen's
    ``_expand_dit_patch_embedding_input_dim`` (examples/train_camlayout_rgb.py)."""
    if dit is None or extra_in_dim <= 0:
        return
    old = dit.patch_embedding
    old_in_dim = old.in_channels
    new_in_dim = old_in_dim + extra_in_dim
    if new_in_dim == old_in_dim:
        return
    new = torch.nn.Conv3d(
        in_channels=new_in_dim,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight[:, :old_in_dim].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    dit.patch_embedding = new
    dit.in_dim = new_in_dim
    # Keep the diffusers config in sync so save_pretrained/from_pretrained round-trips the new in_dim
    # (otherwise a resumed / reloaded checkpoint rebuilds a 32-channel conv and the 48-channel weight fails to load).
    try:
        dit.register_to_config(in_dim=new_in_dim)
    except Exception as e:
        print(f"[camlayout] warning: could not register_to_config(in_dim={new_in_dim}): {e}")
    print(f"[camlayout] expanded patch_embedding in_dim {old_in_dim} -> {new_in_dim}")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def _gather_object_to_rank0(local_obj, accelerator):
    """Gather a Python object from every rank ONLY to rank 0 (avoids the all-gather in
    ``accelerate.utils.gather_object`` duplicating the large per-rank video buffers on
    every process). Returns a list (one entry per rank) on rank 0, else None."""
    if accelerator.num_processes == 1:
        return [local_obj]
    import torch.distributed as dist
    out = [None for _ in range(accelerator.num_processes)] if accelerator.is_main_process else None
    dist.gather_object(local_obj, out, dst=0)
    return out


def _overlay_bbox_canvas(video_fchw, canvas_f3hw):
    """Draw the per-frame bbox outlines carried by ``canvas_f3hw`` (``[F, 3, H, W]`` in [-1, 1], colored
    rectangle edges on a black background — the DENSE layout canvas) onto ``video_fchw`` (``[F, C, H, W]``
    uint8 in [0, 255]). Every non-black canvas pixel (a box edge; PIL draws outlines with no anti-aliasing)
    replaces the underlying video pixel, so each frame shows its own boxes. Returns a new uint8 array."""
    canvas_u8 = (((canvas_f3hw.float().clamp(-1, 1) + 1) / 2) * 255).to(torch.uint8).cpu().numpy()  # [F, 3, H, W]
    out = np.ascontiguousarray(video_fchw)
    mask = canvas_u8.max(axis=1, keepdims=True) > 10           # [F, 1, H, W] — box-edge pixels only
    np.copyto(out, canvas_u8, where=np.broadcast_to(mask, out.shape))
    return out


# Base seed of the per-row layout draw in log_validation (row i -> random.seed(VAL_LAYOUT_SEED + i)).
VAL_LAYOUT_SEED = 0

def log_validation(vae, text_encoder, tokenizer, clip_image_encoder, transformer3d, val_dataset, args, config, accelerator, weight_dtype, global_step):
    """Run camera-control inference over the validation dataset, sharded across GPUs
    (mirrors CamTraj-VideoGen's run_validation), and log the videos to wandb from rank 0.

    The generated video is annotated with the DENSE per-frame bboxes (the full intended box trajectory,
    even though the student is conditioned on the SPARSE last-frame canvas). The ground-truth videos are
    NOT logged here — upload them once (separately) via camlayout/opd/upload_gt_videos.py."""
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running validation... ")
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
            )

            pipeline = WanCamLayoutPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d,
                scheduler=scheduler,
                clip_image_encoder=clip_image_encoder,
            )
            pipeline = pipeline.to(accelerator.device)
            # pipeline.to() does not reliably move the CLIP image encoder (and under
            # --low_vram it sits on CPU between steps); a CPU bf16 CLIP forward dies with
            # a mixed-dtype LayerNorm error -> move it explicitly, restored below.
            clip_image_encoder.to(accelerator.device)

            height, width = int(args.video_sample_size[0]), int(args.video_sample_size[1])

            # Assign each rank a round-robin subset of the validation set.
            total = len(val_dataset)
            if args.validation_num_samples is not None:
                total = min(total, args.validation_num_samples)
            num_processes = accelerator.num_processes
            process_index = accelerator.process_index
            rank_indices = list(range(total))[process_index::num_processes]
            max_local = (total + num_processes - 1) // num_processes  # same count on every rank -> collectives stay balanced

            pbar = tqdm(total=total, desc="Validation", disable=not accelerator.is_main_process)
            local_results = []
            for iter_idx in range(max_local):
                if iter_idx < len(rank_indices):
                    i = rank_indices[iter_idx]
                    # Fixed per-row layout draw: the dataset picks the object subset and the box
                    # colours (also written into the prompt) from the GLOBAL random, whose state
                    # differs at every validation. Seed by row index -- same scheme as
                    # infer_val_layout.py --layout_seed 0 -- and restore, so every validation of
                    # every run sees identical conditioning and training randomness is untouched.
                    _rand_state = random.getstate()
                    random.seed(VAL_LAYOUT_SEED + i)
                    data = val_dataset[i]
                    random.setstate(_rand_state)
                    pixel_values = data["pixel_values"]                     # [F, 3, H, W] in [-1, 1]
                    control_camera_values = data["control_camera_values"]   # [F, 6, H, W]
                    prompt = data["text"]
                    # augment the prompt with per-object color/phrase (same as training). Skipped when the
                    # student is trained without the bbox sentences, so validation matches its condition.
                    if not args.student_no_bbox_prompt:
                        for _phrase, _color in zip(data["bbox_phrases"], data["bbox_color_names"]):
                            prompt = prompt + f"\nIn the {_color} bounding box region: {_phrase}."

                    # first frame -> input_image (drives the start-image y and the CLIP feature)
                    ff = pixel_values[0].float().clamp(-1, 1)
                    first_frame_pil = Image.fromarray(((ff + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
                    control_camera_video = rearrange(control_camera_values, "f c h w -> 1 c f h w").contiguous()  # [1, 6, F, H, W]
                    # Validate the STUDENT under its intended SPARSE layout conditioning (only
                    # student_num_keep_frames frames kept, last always kept) so the eval reflects
                    # sparse-conditioned inference, not the dense teacher condition.
                    val_canvas = make_sparse_canvas(data["bbox_canvas"][None], args.student_num_keep_frames)[0]  # [F, 3, H, W]
                    control_layout_video = rearrange(val_canvas, "f c h w -> 1 c f h w").contiguous()             # [1, 3, F, H, W]

                    # fixed seed per sample -> validation is deterministic and independent of the GPU count
                    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    sample = pipeline(
                        prompt,
                        num_frames           = pixel_values.shape[0],
                        negative_prompt      = NEGATIVE_PROMPT,
                        height               = height,
                        width                = width,
                        generator            = generator,
                        input_image          = first_frame_pil,
                        control_camera_video = control_camera_video,
                        control_layout_video = control_layout_video,
                        num_inference_steps  = 50,
                        guidance_scale       = 6.0,
                    ).videos  # [1, C, F, H, W] in [0, 1]

                    # os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                    # save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-val{i:03d}.mp4"), fps=16)
                    video_np = (sample[0].permute(1, 0, 2, 3).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()  # [F, C, H, W]
                    # annotate the generated video with the DENSE per-frame bboxes (full intended trajectory)
                    gen_np = _overlay_bbox_canvas(video_np, data["bbox_canvas"])
                    local_results.append((i, gen_np, prompt))

                # every rank gathers each iteration so the collectives below never deadlock
                completed = torch.tensor(len(local_results), device=accelerator.device, dtype=torch.int64)
                gathered = accelerator.gather(completed)
                if accelerator.is_main_process:
                    pbar.n = int(gathered.sum().item()); pbar.refresh()

            # gather every rank's results to rank 0 and log in sample order
            if accelerator.num_processes > 1:
                gathered = _gather_object_to_rank0(local_results, accelerator)
                all_results = sorted([x for sub in gathered for x in sub], key=lambda t: t[0]) if accelerator.is_main_process else None
            else:
                all_results = sorted(local_results, key=lambda t: t[0])

            if accelerator.is_main_process and all_results and is_wandb_available():
                val_videos = [wandb.Video(gen, fps=16, format="mp4", caption=c) for _, gen, c in all_results]
                accelerator.log({"val/videos": val_videos}, step=global_step)
                del val_videos

            del pipeline, local_results, all_results
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
            text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
            clip_image_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
            accelerator.wait_for_everyone()
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")
        vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        clip_image_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        # Re-sync with the other ranks even if this rank hit an eval error (e.g. a rank-0-only
        # wandb failure), so a one-rank exception cannot desync the collectives and hang training.
        accelerator.wait_for_everyone()

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)


def make_sparse_canvas(dense_canvas, num_keep):
    """Derive the SPARSE student layout canvas from the DENSE teacher canvas by blanking every frame
    except ``num_keep`` evenly-spaced frames (the last frame is always kept). The empty canvas value is
    ``-1`` (a black RGB canvas after the dataset's ``/255*2-1`` normalization), matching the diffsynth
    ``_make_sparse_canvas`` (which keeps the last frame only, i.e. ``num_keep == 1``).

    dense_canvas: ``[B, F, 3, H, W]`` in [-1, 1]. Returns a tensor of the same shape.
    """
    num_frames = dense_canvas.shape[1]
    if num_keep is None or num_keep >= num_frames:
        return dense_canvas.clone()
    if num_keep <= 0:
        # NO layout at all: every frame blank -> a camera-control-only student.
        return torch.full_like(dense_canvas, -1.0)
    if num_keep == 1:
        keep_idx = {num_frames - 1}
    else:
        keep_idx = {int(round(i * (num_frames - 1) / (num_keep - 1))) for i in range(num_keep)}
        keep_idx.add(num_frames - 1)
    sparse = torch.full_like(dense_canvas, -1.0)
    for i in keep_idx:
        sparse[:, i] = dense_canvas[:, i]
    return sparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Directory that relative Video_Path / Annotation_Path entries in the CSVs are resolved against "
             "(default: the directory of each CSV, i.e. the released LIFT-Vista layout).",
    )
    parser.add_argument(
        "--validation_csv",
        type=str,
        default=None,
        help=("Validation metadata CSV (same format as --train_data_meta). Loaded with the same "
              "WanFunCameraControlDataset; every row is generated (sharded across GPUs) and logged to wandb."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    # parser.add_argument(
    #     "--scale_lr",
    #     action="store_true",
    #     default=False,
    #     help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    # )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--dataloader_prefetch_factor", type=int, default=2,
        help="Batches each worker keeps in flight (torch default 2). Host RAM scales as "
             "num_processes x num_workers x prefetch x batch bytes; a layout sample is ~835 MB fp32 "
             "(pixels + 6-ch Plucker + bbox canvas), which OOM-killed a 220 GB job at bs8 x 4 workers "
             "x prefetch 2 x 2 ranks. Lower it if the cgroup runs hot.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--no_grad_clip", default=False, action="store_true",
        help="Disable gradient clipping entirely (backward -> optimizer.step with no clip_grad_norm_), "
             "matching the diffsynth OPD trainer. Ablation knob to test whether the aggressive 0.05 clip "
             "is what keeps videox-fun OPD from blurring.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--no_checkpoint", default=False, action="store_true",
        help="Disable ALL checkpoint saving (periodic AND the final end-of-training save). Validation still "
             "runs. Useful for eval-only OPD experiments where only the wandb val videos matter.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_num_samples",
        type=int,
        default=None,
        help="Cap the number of validation rows generated per validation (e.g. for debug). None = all rows.",
    )
    parser.add_argument(
        "--validation_only", default=False, action="store_true",
        help="Load the model, run ONE validation pass at global_step=0 (i.e. the pre-training baseline), "
             "log val/videos, then exit WITHOUT training. Do not pass --resume_from_checkpoint so the "
             "freshly-loaded --transformer_path weights are validated as-is.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers (for wandb this is the "
            "wandb project). See https://huggingface.co/docs/accelerate/en/package_reference/accelerator"
        ),
    )
    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default=None,
        help="Run name for the tracker (wandb run name). If unset, the tracker auto-generates one.",
    )

    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    # parser.add_argument(
    #     "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    # )
    # parser.add_argument(
    #     "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    # )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        nargs=2,
        default=[352, 640],
        help='Fixed training resolution (H W) used for the "camera" train_mode.',
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=1,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=81,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--max_objs", type=int, default=5,
        help="Max number of layout objects (bboxes) drawn on the canvas per sample.",
    )
    parser.add_argument(
        "--bbox_area_threshold", type=float, default=0.02,
        help="Drop layout bboxes smaller than this fraction of the frame area.",
    )
    parser.add_argument(
        "--layout_drop_rate", type=float, default=0.0,
        help="Sparsify the layout by dropping this fraction of frames (0 = dense; last frame always kept). "
             "For OPD keep this 0.0: the dataset returns the DENSE canvas (teacher condition); the student's "
             "SPARSE canvas is derived in-code from it (see --student_num_keep_frames).",
    )
    # -------- On-policy distillation (dense-layout teacher -> sparse-layout student) --------
    parser.add_argument(
        "--teacher_transformer_path", type=str, default=None,
        help="Teacher transformer weights (dense per-frame layout). If unset, the teacher is a frozen copy "
             "of the student's initial weights (--transformer_path), i.e. student is initialized FROM the teacher.",
    )
    parser.add_argument(
        "--student_num_keep_frames", type=int, default=1,
        help="How many layout frames the SPARSE student is conditioned on (evenly spaced, last frame always "
             "kept). 1 = last-frame-only sparse layout (matches the diffsynth OPD default); "
             "0 = NO layout at all (every frame's canvas blanked -> camera-control-only student).",
    )
    parser.add_argument(
        "--student_cond_mix_prob", type=float, default=0.0,
        help="Train ONE shared-weight student under TWO conditions, resampled every optimizer step: "
             "with this probability the student gets LAST-FRAME layout + the per-object bbox sentences, "
             "otherwise it gets NO layout at all (blank canvas + bare caption). 0 = off (use the static "
             "--student_num_keep_frames / --student_no_bbox_prompt, which then also drive validation). "
             "The teacher always keeps the DENSE layout, so both modes distil from the same privileged "
             "target. The draw is derived from (seed, global_step), so it is identical on every rank and "
             "reproducible across resumes.",
    )
    parser.add_argument(
        "--student_no_bbox_prompt", action="store_true",
        help="Drop the per-object 'In the {color} bounding box region: {phrase}.' sentences from the STUDENT's "
             "prompt (the teacher keeps them). Pair with --student_num_keep_frames 0 so no layout information "
             "reaches the student through either the visual or the text channel.",
    )
    parser.add_argument(
        "--distill_num_steps", type=int, default=8,
        help="Number of coarse denoising steps for the on-policy student roll-out.",
    )
    parser.add_argument(
        "--distill_num_sampled_states", type=int, default=1,
        help="How many visited states to sample from one roll-out trajectory for the distillation loss.",
    )
    parser.add_argument(
        "--distill_rollout_steps", type=int, default=0,
        help="Truncate the on-policy roll-out to the FIRST N steps of the distill_num_steps schedule "
             "(0 = roll the full schedule). Sampled states are drawn from the rolled prefix only, so e.g. "
             "--distill_num_steps 50 --distill_rollout_steps 10 --distill_num_sampled_states 10 distills on "
             "exactly the 10 highest-noise states without paying for the remaining 40 roll-out forwards.",
    )
    parser.add_argument(
        "--distill_state_list", type=str, default=None,
        help="Explicit, deterministic roll-out states to distil on, as comma-separated 0-based indices into the "
             "distill_num_steps schedule (e.g. '40,41,...,49' = the last 10 of 50; '0,5,11,...,49' = 10 equally "
             "spaced). Every sample uses the same set. The roll-out stops after the last listed state. "
             "Overrides --distill_num_sampled_states / --distill_rollout_steps / --distill_stratified_sampling.",
    )
    parser.add_argument(
        "--distill_stratified_sampling", action="store_true",
        help="Stratified state sampling: split the rolled schedule (prefix) into distill_num_sampled_states "
             "contiguous non-overlapping bins and draw exactly ONE state uniformly from each bin (per sample). "
             "E.g. --distill_num_steps 50 --distill_num_sampled_states 10 -> bins [0-4],[5-9],...,[45-49], one "
             "state each, covering the full noise range with only 10 loss states.",
    )
    parser.add_argument(
        "--distill_micro_bs", type=int, default=0,
        help="Rows per student forward/backward when computing the distill (+anchor) loss. The k*B distill "
             "rows are split into micro-batches of this size and gradients are ACCUMULATED across them "
             "(DDP synced only on the last), so peak activation memory is bounded by one micro-batch instead "
             "of scaling with k. 0 = use train_batch_size (~one roll-out state per forward, diffsynth-style).",
    )
    parser.add_argument(
        "--distill_shift", type=float, default=5.0,
        help="Shift for the roll-out FlowMatch scheduler. Defaults to the config scheduler_kwargs.shift (5.0).",
    )
    parser.add_argument(
        "--sft_loss_weight", type=float, default=0.0,
        help="Weight of an auxiliary ground-truth flow-matching loss (student, sparse layout) that anchors the "
             "student to real data. 0 = pure on-policy distillation.",
    )
    parser.add_argument(
        "--distill_loss_type", type=str, default="x0", choices=["x0", "velocity"],
        help="Distillation target: 'x0' matches reconstructed x0 (== sigma^2-weighted velocity, faithful to "
             "the diffsynth OPD); 'velocity' matches the raw velocity (uniform weighting, less blur-prone at "
             "high noise).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/wan2.1/wan_civitai.yaml",
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    # parser.add_argument(
    #     "--vae_path",
    #     type=str,
    #     default=None,
    #     help=("If you want to load the weight from other vaes, input its path."),
    # )

    parser.add_argument(
        '--trainable_modules', 
        nargs='+', 
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    # parser.add_argument(
    #     "--train_mode",
    #     type=str,
    #     default="normal",
    #     help=(
    #         'The format of training data. Support `"normal"`'
    #         ' (default), `"i2v"`, `"camera"` (Wan2.1-Fun camera control on the CamLayout npz dataset).'
    #     ),
    # )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    
    # timestep sampling config
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # Validation lets rank 0 do slow I/O (online wandb video upload) while other ranks
        # wait at the post-validation barrier; raise the collective timeout past the 10-min
        # NCCL default so that does not trip the watchdog.
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=7200))],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    # Dedicated Python RNG for the on-policy roll-out state sampling. Seeded PER RANK so each process picks
    # its own kept states (set_seed above seeds the GLOBAL `random` identically on all ranks), and kept
    # separate from the global `random` so it never interferes with the dataset's worker-side sampling.
    distill_py_rng = random.Random((args.seed + accelerator.process_index) if args.seed is not None else None)
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()
    # Get Vae
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    )
    vae.eval()
    # Get Clip Image Encoder
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    )
    clip_image_encoder = clip_image_encoder.eval()

    # Get Transformer
    transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)

    # Layout: expand patch_embedding to also ingest the VAE-encoded bbox canvas (channel-concatenated
    # to y), growing in_dim 32 -> 48. For OPD, --transformer_path is an ALREADY-expanded (48-ch) camlayout
    # checkpoint (the dense-trained model), so we expand BEFORE loading it. (The base camera-only trainer
    # expands AFTER, because there transformer_path is a 32-ch camera checkpoint.)
    _expand_dit_patch_embedding_input_dim(transformer3d, vae.latent_channels)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    # -------- Build the frozen DENSE teacher (on-policy distillation) --------
    # The teacher shares the student's (now 48-ch) architecture and differs ONLY in its (frozen) weights
    # and in the layout canvas fed via y. By default it is a frozen copy of the student's INITIAL weights
    # (i.e. the student is initialized FROM the teacher); optionally load a separately-trained dense
    # teacher from --teacher_transformer_path. It is kept OUT of accelerator.prepare so DDP never tracks it.
    teacher_transformer3d = copy.deepcopy(transformer3d)
    if args.teacher_transformer_path is not None:
        print(f"Teacher from checkpoint: {args.teacher_transformer_path}")
        if args.teacher_transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            teacher_state_dict = load_file(args.teacher_transformer_path)
        else:
            teacher_state_dict = torch.load(args.teacher_transformer_path, map_location="cpu")
        teacher_state_dict = teacher_state_dict["state_dict"] if "state_dict" in teacher_state_dict else teacher_state_dict
        tm, tu = teacher_transformer3d.load_state_dict(teacher_state_dict, strict=False)
        print(f"[teacher] missing keys: {len(tm)}, unexpected keys: {len(tu)}")
        assert len(tu) == 0
    teacher_transformer3d.requires_grad_(False)
    teacher_transformer3d.eval()

    # A good trainable modules is showed below now.
    # For 3D Patch: trainable_modules = ['ff.net', 'pos_embed', 'attn2', 'proj_out', 'timepositionalencoding', 'h_position', 'w_position']
    # For 2D Patch: trainable_modules = ['ff.net', 'attn2', 'timepositionalencoding', 'h_position', 'w_position']
    transformer3d.train()
    if accelerator.is_main_process:
        accelerator.print(
            f"Trainable modules '{args.trainable_modules}'."
        )
    for name, param in transformer3d.named_parameters():
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                weights.pop()

                with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                    pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = WanTransformer3DModel.from_pretrained(
                    input_dir, subfolder="transformer"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

            pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    loaded_number, _ = pickle.load(file)
                    batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # if args.scale_lr:
    #     args.learning_rate = (
    #         args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    #     )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    # All matched trainable_modules share the same learning rate (no low-lr group).
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
    ]
    in_already = []
    for name, param in transformer3d.named_parameters():
        if name in in_already:
            continue
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                in_already.append(name)
                trainable_params_optim[0]['params'].append(param)
                # if accelerator.is_main_process:
                #     print(f"Set {name} to lr : {args.learning_rate}")
                break

    optimizer = optimizer_cls(
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the training dataset
    # Get the dataset
    # CamLayout camera + layout dataset: fixed resolution, crop-aware Plucker + bbox canvas.
    train_dataset = WanFunCamLayoutControlDataset(
        args.train_data_meta,
        video_sample_size=args.video_sample_size,
        video_sample_n_frames=args.video_sample_n_frames,
        video_sample_stride=args.video_sample_stride,
        text_drop_ratio=0.0,
        data_root=args.data_root,
        max_objs=args.max_objs,
        bbox_area_threshold=args.bbox_area_threshold,
        layout_drop_rate=args.layout_drop_rate,
    )

    # Validation dataset: same loader/format as training; indexed directly (sharded across GPUs) in log_validation.
    val_dataset = None
    if args.validation_csv is not None:
        val_dataset = WanFunCamLayoutControlDataset(
            args.validation_csv,
            video_sample_size=args.video_sample_size,
            video_sample_n_frames=args.video_sample_n_frames,
            video_sample_stride=args.video_sample_stride,
            text_drop_ratio=0.0,
            data_root=args.data_root,
            max_objs=args.max_objs,
            bbox_area_threshold=args.bbox_area_threshold,
            layout_drop_rate=args.layout_drop_rate,
        )

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn
    

    # Fixed-resolution camera+layout path: stack frames + Plucker + bbox canvas, derive first-frame
    # ref/clip, keep per-object phrases/colors for prompt augmentation. No bucketing.
    def camera_collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        control_camera_values = torch.stack([example["control_camera_values"] for example in examples])
        new_examples = {}
        new_examples["pixel_values"] = pixel_values
        new_examples["control_camera_values"] = control_camera_values
        new_examples["bbox_canvas"] = torch.stack([example["bbox_canvas"] for example in examples])
        new_examples["ref_pixel_values"] = pixel_values[:, 0:1].contiguous()
        new_examples["clip_pixel_values"] = (pixel_values[:, 0].permute(0, 2, 3, 1).contiguous() * 0.5 + 0.5) * 255
        new_examples["text"] = [example["text"] for example in examples]
        new_examples["bbox_phrases"] = [example["bbox_phrases"] for example in examples]
        new_examples["bbox_color_names"] = [example["bbox_color_names"] for example in examples]
        new_examples["data_type"] = [example["data_type"] for example in examples]
        return new_examples

    batch_sampler_generator = torch.Generator().manual_seed(args.seed)
    batch_sampler = torch.utils.data.BatchSampler(
        RandomSampler(train_dataset, generator=batch_sampler_generator),
        batch_size=args.train_batch_size, drop_last=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=camera_collate_fn,
        persistent_workers=True if args.dataloader_num_workers != 0 else False,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=(args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None),
        worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
    )

    # Scheduler and math around the number of training steps.
    # `len(train_dataloader)` is still the GLOBAL batch count here (the loader is sharded only later, in
    # `accelerator.prepare`). Estimate the PER-PROCESS sharded length up front so the lr scheduler gets a
    # correct step budget: with accelerate's default AcceleratedScheduler, each `.step()` advances the
    # inner scheduler `num_processes` times, so num_training_steps must be (per-process steps) * num_processes.
    overrode_max_train_steps = False
    len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
    num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    clip_image_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

    # The frozen teacher stays resident on the GPU: it is queried at every roll-out state and, unlike the
    # vae/text/clip encoders, is used throughout the whole distillation step (low_vram does not offload it).
    teacher_transformer3d.to(accelerator.device, dtype=weight_dtype)

    # Dedicated coarse FlowMatch schedule for the on-policy student roll-out. We keep it SEPARATE from
    # `noise_scheduler` (which stays on the 1000-step training schedule used by the optional SFT anchor).
    # `use_dynamic_shifting=false` in the config, so the shift is baked in at construction (not set_timesteps).
    rollout_scheduler_kwargs = filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    if args.distill_shift is not None:
        rollout_scheduler_kwargs['shift'] = args.distill_shift
    rollout_scheduler = FlowMatchEulerDiscreteScheduler(**rollout_scheduler_kwargs)

    # Recompute against the ACTUAL sharded per-process length now that prepare has run.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # The lr scheduler's step budget was fixed at creation from the per-process ESTIMATE above; if the
        # actual sharded length differs (e.g. even_batches padding), warn so the mismatch is visible.
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"lr scheduler num_training_steps ({num_training_steps_for_scheduler}) != recomputed "
                f"max_train_steps*num_processes ({args.max_train_steps * accelerator.num_processes}): the "
                f"estimated per-process dataloader length differed from the actual sharded length."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # Stringify the list-valued args we want to keep visible in the tracker config
        # (trackers only accept scalar hparams, so lists would otherwise be dropped below).
        tracker_config["video_sample_size"] = "x".join(map(str, args.video_sample_size))
        if args.trainable_modules is not None:
            tracker_config["trainable_modules"] = ",".join(args.trainable_modules)
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
            print(f"Removed tracker_config['{k}']")
        # wandb run name is set via init_kwargs (ignored by other trackers).
        init_kwargs = {"wandb": {"name": args.tracker_run_name}} if args.tracker_run_name is not None else {}
        accelerator.init_trackers(args.tracker_project_name, tracker_config, init_kwargs=init_kwargs)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            # Resume: derive first_epoch from global_step. Checkpoints are written at epoch
            # boundaries (global_step == k * num_update_steps_per_epoch), so this is exact and, unlike the
            # pkl's stored value (which is the job's START epoch and is never updated mid-run), it makes a
            # resumed job continue at the NEXT epoch instead of redoing the last one.
            first_epoch = global_step // num_update_steps_per_epoch
            print(f"[resume] global_step={global_step} -> first_epoch={first_epoch} (num_update_steps_per_epoch={num_update_steps_per_epoch})")

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            # Resume shuffle position. The load hook above already restored batch_sampler.sampler._pos_start
            # from the checkpoint. Two cases:
            #  - EPOCH BOUNDARY (global_step % num_update_steps_per_epoch == 0): the checkpoint sits at the
            #    end of an epoch. Start the next
            #    epoch's fresh (seed+epoch) shuffle from index 0.
            #  - MID-EPOCH (the run was stopped before finishing an epoch): KEEP the restored _pos_start so the deterministic (seed+first_epoch) shuffle
            #    resumes from where it stopped instead of re-training the epoch's first batches (which would
            #    make the model loop over only the first part of the epoch and never see the rest).
            if global_step % num_update_steps_per_epoch == 0:
                batch_sampler.sampler._pos_start = 0
            else:
                print(f"[resume] MID-epoch resume: keep _pos_start={batch_sampler.sampler._pos_start} "
                      f"(global_step={global_step}, steps/epoch={num_update_steps_per_epoch})")
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # Pre-training baseline: at global_step 0 (a fresh run, not a resumed one) everything above
    # (model load, patch_embedding expand, --transformer_path load, accelerator.prepare, on-device
    # VAE/text/CLIP, val_dataset, trackers) is exactly the state we want to validate from. Log one
    # validation pass at step 0 (SPARSE-conditioned like real val, DENSE bbox overlay) BEFORE any
    # optimization step. Resumed jobs (global_step>0) skip this and just keep training.
    if val_dataset is not None and global_step == 0:
        if accelerator.is_main_process:
            logger.info("Running step-0 (pre-training) validation before the first optimization step.")
        log_validation(
            vae, text_encoder, tokenizer, clip_image_encoder, transformer3d,
            val_dataset, args, config, accelerator, weight_dtype, 0,
        )

    # --validation_only: the baseline above is all we wanted -> stop here without training.
    if args.validation_only:
        if accelerator.is_main_process:
            logger.info("[validation_only] step-0 validation done; exiting without training.")
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer3d):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)

                # both i2v and camera use a CLIP image condition
                clip_pixel_values = batch["clip_pixel_values"].to(weight_dtype)
                # if args.train_mode == "camera":
                control_camera_values = batch["control_camera_values"].to(weight_dtype)
                ref_pixel_values = batch["ref_pixel_values"].to(weight_dtype)
                bbox_canvas = batch["bbox_canvas"].to(weight_dtype)

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    clip_image_encoder.to(accelerator.device)
                    text_encoder.to("cpu")

                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.mode()  # deterministic (mu), matches diffsynth training
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim = 0)
                    latents = _batch_encode_vae(pixel_values)

                    # start-image conditioning: first-frame latent placed at temporal position 0. Shared by
                    # the teacher and the student (they differ only in the bbox-latent block below).
                    ref_latents = _batch_encode_vae(ref_pixel_values)
                    camera_y_img = torch.zeros_like(latents)
                    camera_y_img[:, :, :1] = ref_latents
                    # --- OPD dense TEACHER y vs sparse STUDENT y ---
                    # The dataset returns the DENSE bbox canvas (all frames carry layout). y = [start-image
                    # latent(16), bbox-canvas latent(16)] (CamTraj RGB-channel-concat injection). The teacher
                    # sees the dense canvas; the student sees the SPARSE canvas (only student_num_keep_frames
                    # frames, last always kept) derived in-code so the two differ ONLY in layout density.
                    bbox_latents_dense = _batch_encode_vae(bbox_canvas)
                    y_teacher = torch.cat([camera_y_img, bbox_latents_dense], dim=1)  # [b, 32, f_lat, h, w]
                    # Two-condition student (--student_cond_mix_prob): one set of weights, the layout
                    # condition resampled per OPTIMIZER step as Bernoulli(--student_cond_mix_prob).
                    # Derived from (seed, global_step) so every rank draws the same mode and a
                    # resumed run replays the same schedule.
                    if args.student_cond_mix_prob > 0.0:
                        _mrng = random.Random((args.seed if args.seed is not None else 0) * 1000003 + global_step)
                        _use_lastframe = _mrng.random() < args.student_cond_mix_prob
                        eff_keep_frames = 1 if _use_lastframe else 0
                        eff_no_bbox_prompt = not _use_lastframe
                    else:
                        eff_keep_frames = args.student_num_keep_frames
                        eff_no_bbox_prompt = args.student_no_bbox_prompt
                    sparse_canvas = make_sparse_canvas(bbox_canvas, eff_keep_frames)
                    bbox_latents_sparse = _batch_encode_vae(sparse_canvas)
                    y_student = torch.cat([camera_y_img, bbox_latents_sparse], dim=1)  # [b, 32, f_lat, h, w]
                    # camera control: Plucker -> control_adapter input, temporally compressed by 4
                    # (matches the validated Wan2.1-Fun camera-control reshape -> [b, 24, f_lat, h, w])
                    control_camera_latents = rearrange(control_camera_values, "b f c h w -> b c f h w")
                    control_camera_latents = torch.concat(
                        [
                            torch.repeat_interleave(control_camera_latents[:, :, 0:1], repeats=4, dim=2),
                            control_camera_latents[:, :, 1:]
                        ], dim=2
                    ).transpose(1, 2).contiguous()
                    control_camera_latents = control_camera_latents.view(control_camera_latents.shape[0], control_camera_latents.shape[1] // 4, 4, control_camera_latents.shape[2], control_camera_latents.shape[3], control_camera_latents.shape[4])
                    control_camera_latents = control_camera_latents.transpose(2, 3).contiguous()
                    control_camera_latents = control_camera_latents.view(control_camera_latents.shape[0], control_camera_latents.shape[1], control_camera_latents.shape[2] * 4, control_camera_latents.shape[4], control_camera_latents.shape[5])
                    control_camera_latents = control_camera_latents.transpose(1, 2)

                    # CLIP image condition (both i2v and camera)
                    clip_context = []
                    for clip_pixel_value in clip_pixel_values:
                        clip_image = Image.fromarray(np.uint8(clip_pixel_value.float().cpu().numpy()))
                        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(clip_image_encoder.device, weight_dtype)
                        _clip_context = clip_image_encoder([clip_image[:, None, :, :]])
                        clip_context.append(_clip_context)
                    clip_context = torch.cat(clip_context)

                if args.low_vram:
                    vae.to('cpu')
                    clip_image_encoder.to('cpu')
                    torch.cuda.empty_cache()
                    text_encoder.to(accelerator.device)

                # Augment each prompt with per-object "In the <color> bounding box region: <phrase>."
                # (CamTraj WanVideoUnit_PromptEmbedder) so the model can tie colored boxes to objects.
                layout_texts = []
                for _t, _phrases, _colors in zip(batch['text'], batch['bbox_phrases'], batch['bbox_color_names']):
                    for _phrase, _color in zip(_phrases, _colors):
                        _t = _t + f"\nIn the {_color} bounding box region: {_phrase}."
                    layout_texts.append(_t)

                with torch.no_grad():
                    def _encode_texts(texts):
                        prompt_ids = tokenizer(
                            texts,
                            padding="max_length",
                            max_length=args.tokenizer_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids
                        prompt_attention_mask = prompt_ids.attention_mask

                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                        return [u[:v] for u, v in zip(embeds, seq_lens)]

                    prompt_embeds = _encode_texts(layout_texts)
                    # The student may be denied the bbox sentences (camera-only regime); the teacher keeps them.
                    prompt_embeds_student = (_encode_texts(list(batch['text']))
                                             if eff_no_bbox_prompt else prompt_embeds)
                    _seen_modes = getattr(args, "_cond_logged_modes", set())
                    if eff_keep_frames not in _seen_modes:
                        _seen_modes.add(eff_keep_frames)
                        args._cond_logged_modes = _seen_modes
                        print(f"[rank {accelerator.process_index}] student cond | keep_frames="
                              f"{eff_keep_frames} canvas_min/max="
                              f"{sparse_canvas.min().item():.3f}/{sparse_canvas.max().item():.3f} "
                              f"(min=max=-1 => no layout) | no_bbox_prompt={eff_no_bbox_prompt} | "
                              f"mix_prob={args.student_cond_mix_prob} | "
                              f"teacher_prompt(len {len(layout_texts[0])}, tail)=...{layout_texts[0][-90:]!r} | "
                              f"student_prompt(len {len(prompt_s0 := (batch['text'][0] if eff_no_bbox_prompt else layout_texts[0]))}, tail)=...{prompt_s0[-90:]!r}")

                # Everything pixel-space has been consumed by now (3 VAE encodes, the Plucker reshape
                # and the CLIP loop above; the banner is the last reader of sparse_canvas/batch).
                # Free it BEFORE the roll-out and the chunked distill backward, which is where the
                # peak lives -- measured 88.7 GB alloc / 92.2 GB reserved of 96 at micro_bs=8.
                # accelerate put the fp32 batch on the GPU and each .to(weight_dtype) above is a
                # separate bf16 copy, so at bs4 this is ~4.9 GB (pixels 0.8+0.4, canvas 0.8+0.4,
                # 6-ch Plucker 1.6+0.8, plus ref/clip) idling through the whole step.
                del pixel_values, clip_pixel_values, control_camera_values, ref_pixel_values
                del bbox_canvas, sparse_canvas
                batch = None

                if args.low_vram:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

                bsz, channel, num_frames, height, width = latents.size()

                # seq_len for the transformer positional packing (shared by teacher & student, constant
                # across roll-out states).
                target_shape = (vae.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                # y_camera (Plucker) and clip_fea are shared by teacher & student and constant across states.
                y_camera = control_camera_latents
                clip_fea = clip_context
                student_raw = accelerator.unwrap_model(transformer3d)  # raw module for the no-grad roll-out

                def _call_dit(model, x_in, t_in, y_in, context_in, y_camera_in, clip_in):
                    return model(
                        x=x_in, context=context_in, t=t_in, seq_len=seq_len,
                        y=y_in, y_camera=y_camera_in, clip_fea=clip_in,
                    )

                # ---------------- On-policy roll-out (student, SPARSE layout, no grad) ----------------
                # Roll the student forward from pure noise over a COARSE schedule under its OWN sparse
                # conditioning, keeping k randomly-chosen visited noisy states (x_t, t, sigma). Under no_grad
                # this stays memory-bounded; only the loss forwards below carry grad. Keep steps are sampled
                # PER SAMPLE (and per rank, via distill_py_rng) so each item in the batch is distilled at its
                # own noise levels; every sample still contributes exactly k rows -> k*B distill rows on every
                # rank, so the single DDP student forward below has a rank-consistent shape.
                with torch.no_grad():
                    rollout_scheduler.set_timesteps(args.distill_num_steps, device=accelerator.device)
                    roll_timesteps = rollout_scheduler.timesteps
                    roll_sigmas = rollout_scheduler.sigmas
                    num_states = len(roll_timesteps)
                    # Optional prefix truncation: roll only the first `distill_rollout_steps` steps of the
                    # schedule and sample states from that prefix (rows beyond the prefix are never used,
                    # so stopping early is exact — it just skips wasted roll-out forwards).
                    # State selection (shared by single- and dual-condition students): roll the first
                    # `distill_rollout_steps` steps of the schedule and take `distill_num_sampled_states` of them
                    # (random / stratified / explicit list). The released recipe rolls 10 and keeps all 10.
                    rollout_limit = getattr(args, "distill_rollout_steps", 0) or 0
                    rollout_limit = min(rollout_limit, num_states) if rollout_limit > 0 else num_states
                    k = max(1, min(args.distill_num_sampled_states, rollout_limit))
                    if getattr(args, "distill_state_list", None):
                        # explicit state indices: same set for every sample, roll-out truncated right
                        # after the last one (later steps could never enter the loss)
                        _sl = sorted({int(t) for t in args.distill_state_list.split(",") if t.strip()})
                        assert _sl and _sl[0] >= 0 and _sl[-1] < num_states, \
                            f"--distill_state_list {_sl} outside the {num_states}-step schedule"
                        rollout_limit, k = _sl[-1] + 1, len(_sl)
                        keep_per_sample = [set(_sl) for _ in range(bsz)]
                        if not getattr(args, "_list_logged", False):
                            args._list_logged = True
                            print(f"[rank {accelerator.process_index}] state scheme | explicit list {_sl} | "
                                  f"rows/sample={k} rollout_limit={rollout_limit}")
                    elif getattr(args, "distill_stratified_sampling", False):
                        # k contiguous non-overlapping bins over the rolled prefix, ONE state per bin
                        # (bins are disjoint so each sample still contributes exactly k rows).
                        edges = [round(j * rollout_limit / k) for j in range(k + 1)]
                        keep_per_sample = [
                            set(distill_py_rng.randrange(edges[j], edges[j + 1]) for j in range(k))
                            for _ in range(bsz)
                        ]
                        if not getattr(args, "_strat_logged", False):
                            args._strat_logged = True
                            print(f"[rank {accelerator.process_index}] stratified state sampling: bins={edges}, "
                                  f"first-batch keeps={[sorted(s) for s in keep_per_sample]}")
                    else:
                        keep_per_sample = [set(distill_py_rng.sample(range(rollout_limit), k=k)) for _ in range(bsz)]
                        if not getattr(args, "_rand_logged", False):
                            args._rand_logged = True
                            print(f"[rank {accelerator.process_index}] state scheme | random {k} of prefix 0..{rollout_limit - 1} | "
                                  f"first-batch keeps={[sorted(x) for x in keep_per_sample]}")

                    x = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)
                    kept_rows = []  # (x_row [1, C, F, H, W], t_scalar, sigma_scalar, sample_idx)
                    for i in range(rollout_limit):
                        t_i = roll_timesteps[i]
                        for b in range(bsz):
                            if i in keep_per_sample[b]:
                                kept_rows.append((x[b:b + 1].detach().clone(), t_i, roll_sigmas[i], b))
                        t_b = t_i.to(device=x.device).expand(x.shape[0])
                        with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                            v_roll = _call_dit(student_raw, x, t_b, y_student, prompt_embeds_student, y_camera, clip_fea)
                        x = rollout_scheduler.step(v_roll, t_i, x, return_dict=False)[0].to(weight_dtype)

                # ---- Chunked distill (+ optional SFT anchor) loss: memory-safe, DDP-correct ----
                # Stacking all k*B distill rows into ONE grad forward makes peak activation memory scale with
                # k (e.g. k=50, B=2 -> 100 rows in a single backward -> OOM). Instead we split the rows into
                # micro-batches of `distill_micro_bs` rows and do a forward+backward per micro-batch,
                # ACCUMULATING gradients. Peak activation memory is then bounded by one micro-batch,
                # independent of k. The accumulated gradient is IDENTICAL to the stacked version's (sum of
                # per-chunk mean-losses weighted by row fraction == mean over all rows). DDP correctness: we
                # sync gradients only on the LAST micro-batch (no_sync on all earlier ones) -> exactly one
                # all-reduce per optimizer step, and no "marked ready twice" reducer error.
                micro = args.distill_micro_bs if args.distill_micro_bs and args.distill_micro_bs > 0 else bsz

                distill_rows = kept_rows                      # (x_row [1,...], t_scalar, sigma_scalar, sample_idx)
                n_distill = len(distill_rows)

                # optional SFT-anchor rows (GT-noised states, sparse layout, velocity target = noise-latents)
                anchor_rows = []
                if args.sft_loss_weight > 0.0:
                    a_indices = idx_sampling(bsz, generator=torch_rng, device=latents.device).long().cpu()
                    t_anchor = noise_scheduler.timesteps[a_indices].to(device=latents.device)
                    sigma_anchor = get_sigmas(t_anchor, n_dim=latents.ndim, dtype=latents.dtype)
                    noise_anchor = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)
                    x_anchor = (1.0 - sigma_anchor) * latents + sigma_anchor * noise_anchor
                    tgt_anchor = noise_anchor - latents
                    anchor_rows = [(x_anchor[b:b + 1], t_anchor[b], tgt_anchor[b:b + 1], b) for b in range(bsz)]

                distill_chunks = [distill_rows[i:i + micro] for i in range(0, n_distill, micro)]
                anchor_chunks = [anchor_rows[i:i + micro] for i in range(0, len(anchor_rows), micro)]
                total_chunks = len(distill_chunks) + len(anchor_chunks)

                def _assemble(rows):
                    idx = [r[-1] for r in rows]
                    x_c = torch.cat([r[0] for r in rows], dim=0)
                    t_c = torch.cat([r[1].to(device=latents.device).view(1) for r in rows], dim=0)
                    ctx_c = [prompt_embeds[b] for b in idx]                  # teacher prompt
                    ctx_s_c = [prompt_embeds_student[b] for b in idx]        # student prompt (may drop bbox text)
                    yc_c = torch.cat([y_camera[b:b + 1] for b in idx], dim=0)
                    cl_c = torch.cat([clip_fea[b:b + 1] for b in idx], dim=0) if clip_fea is not None else None
                    return idx, x_c, t_c, ctx_c, ctx_s_c, yc_c, cl_c

                chunk_i, distill_loss_val, anchor_loss_val = 0, 0.0, 0.0
                for rows in distill_chunks:
                    idx, x_c, t_c, ctx_c, ctx_s_c, yc_c, cl_c = _assemble(rows)
                    sigma_c = torch.stack([r[2].to(device=latents.device, dtype=torch.float32) for r in rows]) \
                                    .view(-1, *([1] * (latents.ndim - 1)))
                    ys_c = torch.cat([y_student[b:b + 1] for b in idx], dim=0)
                    yt_c = torch.cat([y_teacher[b:b + 1] for b in idx], dim=0)
                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                        v_t = _call_dit(teacher_transformer3d, x_c, t_c, yt_c, ctx_c, yc_c, cl_c)
                    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                        v_s = _call_dit(transformer3d, x_c, t_c, ys_c, ctx_s_c, yc_c, cl_c)
                    if args.distill_loss_type == "x0":
                        x0_t = x_c.float() - sigma_c * v_t.float()
                        x0_s = x_c.float() - sigma_c * v_s.float()
                        l = F.mse_loss(x0_s, x0_t.detach())
                    else:  # "velocity"
                        l = F.mse_loss(v_s.float(), v_t.float().detach())
                    l_scaled = l * (len(rows) / n_distill)     # sum over chunks == mean over all distill rows
                    with (nullcontext() if chunk_i == total_chunks - 1 else accelerator.no_sync(transformer3d)):
                        accelerator.backward(l_scaled)
                    distill_loss_val += float(l_scaled.detach())
                    chunk_i += 1

                for rows in anchor_chunks:
                    idx, x_c, t_c, ctx_c, ctx_s_c, yc_c, cl_c = _assemble(rows)
                    ys_c = torch.cat([y_student[b:b + 1] for b in idx], dim=0)
                    tgt_c = torch.cat([r[2] for r in rows], dim=0)
                    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                        v_s = _call_dit(transformer3d, x_c, t_c, ys_c, ctx_s_c, yc_c, cl_c)
                    l = F.mse_loss(v_s.float(), tgt_c.float()) * (len(rows) / bsz)   # sum over chunks == mean over anchor rows
                    with (nullcontext() if chunk_i == total_chunks - 1 else accelerator.no_sync(transformer3d)):
                        accelerator.backward(args.sft_loss_weight * l)
                    anchor_loss_val += float(l.detach())
                    chunk_i += 1

                # Scalar tensors for logging (grads were already accumulated per-chunk above; no more backward).
                distill_loss = torch.tensor(distill_loss_val, device=accelerator.device)
                anchor_loss = torch.tensor(anchor_loss_val, device=accelerator.device) if anchor_rows else None
                loss = torch.tensor(distill_loss_val + args.sft_loss_weight * anchor_loss_val, device=accelerator.device)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                if accelerator.sync_gradients:
                    trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                    trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                    max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                    if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                        actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                    else:
                        actual_max_grad_norm = max_grad_norm

                    if args.report_model_info and accelerator.is_main_process:
                        # stdout copy of the PRE-clip norm and the threshold in force, so a
                        # WANDB_MODE=disabled probe can read them (off in every real run)
                        print(f"[gradnorm] step={global_step} mode={'lastframe' if eff_keep_frames == 1 else 'camonly'} "
                              f"pre_clip={float(trainable_params_total_norm):.4f} clip_at={float(actual_max_grad_norm):.4f}",
                              flush=True)

                    if args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            accelerator.log(
                                {
                                    f'gradients/before_clip_norm/{name}': param.grad.norm().item()
                                    for name, param in transformer3d.named_parameters()
                                    if param.requires_grad
                                },
                                step=global_step,
                            )

                    if not args.no_grad_clip:
                        norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                        # This list references every .grad tensor. Without dropping it,
                        # optimizer.zero_grad(set_to_none=True) leaves last step's grads (1.3B fp32
                        # = 4.8 GB) alive through the whole next step, until the list is reassigned.
                        del trainable_params_grads
                        if args.report_model_info and accelerator.is_main_process:
                            accelerator.log(
                                {
                                    'gradients/norm_sum': float(norm_sum),
                                    'gradients/actual_max_grad_norm': float(actual_max_grad_norm),
                                },
                                step=global_step,
                            )
                    elif accelerator.is_main_process:
                        # --no_grad_clip ablation: no clipping at all (like diffsynth). Log the raw (unclipped)
                        # grad norm at this same step so we can watch it grow vs the clipped runs.
                        accelerator.log({'gradients/unclipped_norm': float(trainable_params_total_norm)}, step=global_step)
                        
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                if torch.cuda.is_available():
                    # peak since the last reset -> the true per-step high-water mark, which the
                    # micro-batch size drives. Reset each step so one early spike does not mask
                    # the steady state.
                    _pk = torch.cuda.max_memory_allocated() / 2**30
                    _rs = torch.cuda.max_memory_reserved() / 2**30
                    print(f"[rank {accelerator.process_index}] step {global_step} "
                          f"gpu_peak_alloc={_pk:.1f}G reserved={_rs:.1f}G "
                          f"micro_bs={args.distill_micro_bs or args.train_batch_size}", flush=True)
                    torch.cuda.reset_peak_memory_stats()
                log_dict = {
                    "train_loss": train_loss,
                    "distill_loss": float(distill_loss.detach().item()),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if anchor_loss is not None:
                    log_dict["anchor_loss"] = float(anchor_loss.detach().item())
                accelerator.log(log_dict, step=global_step)
                train_loss = 0.0

                if not args.no_checkpoint and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if val_dataset is not None and global_step % args.validation_steps == 0:
                    log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        clip_image_encoder,
                        transformer3d,
                        val_dataset,
                        args,
                        config,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.student_cond_mix_prob > 0.0:
                # 1 = last-frame layout this step, 0 = camera-only; its running mean must track mix_prob
                logs["student_lastframe"] = float(eff_keep_frames == 1)
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if val_dataset is not None and epoch % args.validation_epochs == 0:
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                clip_image_encoder,
                transformer3d,
                val_dataset,
                args,
                config,
                accelerator,
                weight_dtype,
                global_step,
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and not args.no_checkpoint:
        transformer3d = unwrap_model(transformer3d)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
