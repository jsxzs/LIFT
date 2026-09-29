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

from lr_schedules import add_lr_schedule_args, build_lr_scheduler
from diffusers.training_utils import (compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data.bucket_sampler import RandomSampler
from videox_fun.data.dataset_camlayout import (COLOR_LIST, WanFunCameraControlDataset,
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


def _draw_bboxes_on_video(video_np, bboxes, color_names):
    """Draw the layout boxes onto a generated video. video_np is [F, C, H, W] uint8.

    Ported from CamTraj-VideoGen's run_validation, with one deliberate change: the outline colour
    comes from the sample's own bbox_color_names via COLOR_LIST, not from a separate hard-coded
    palette indexed by object order. The dataset picks colours with random.sample and the prompt
    says "In the <colour> bounding box region: ...", so a palette lookup by index would draw a box
    in a colour the prompt never mentions.

    Returns (annotated [F, C, H, W] uint8, last annotated frame as PIL) -- or (video_np, None) when
    the sample carries no boxes.
    """
    if not bboxes:
        return video_np, None
    frames = []
    for fi in range(video_np.shape[0]):
        img = Image.fromarray(video_np[fi].transpose(1, 2, 0))
        draw = ImageDraw.Draw(img)
        # large boxes first, so a small box nested inside stays visible (same order as the canvas)
        items = []
        for obj_bboxes, cname in zip(bboxes, color_names):
            if not isinstance(obj_bboxes, list) or fi >= len(obj_bboxes) or obj_bboxes[fi] is None:
                continue
            items.append((obj_bboxes[fi], cname))
        items.sort(key=lambda it: max(0.0, it[0][2] - it[0][0]) * max(0.0, it[0][3] - it[0][1]),
                   reverse=True)
        for bbox, cname in items:
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]],
                           outline=COLOR_LIST.get(cname, (255, 255, 255)), width=2)
        frames.append(np.array(img))
    ann = np.stack(frames).transpose(0, 3, 1, 2)
    return ann, Image.fromarray(frames[-1])


# Base seed of the per-row layout draw in log_validation (row i -> random.seed(VAL_LAYOUT_SEED + i)).
VAL_LAYOUT_SEED = 0

def log_validation(vae, text_encoder, tokenizer, clip_image_encoder, transformer3d, val_dataset, args, config, accelerator, weight_dtype, global_step):
    """Run camera-control inference over the validation dataset, sharded across GPUs
    (mirrors CamTraj-VideoGen's run_validation), and log the videos to wandb from rank 0."""
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running validation... ")
            _val_sched_kwargs = OmegaConf.to_container(config['scheduler_kwargs'])
            if args.scheduler_shift is not None:
                _val_sched_kwargs['shift'] = args.scheduler_shift   # validation denoising uses the same shift as training
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, _val_sched_kwargs)
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
                    _rand_state = random.getstate()
                    random.seed(VAL_LAYOUT_SEED + i)
                    data = val_dataset[i]
                    random.setstate(_rand_state)
                    pixel_values = data["pixel_values"]                     # [F, 3, H, W] in [-1, 1]
                    control_camera_values = data["control_camera_values"]   # [F, 6, H, W]
                    prompt = data["text"]
                    # augment the prompt with per-object color/phrase (same as training)
                    for _phrase, _color in zip(data["bbox_phrases"], data["bbox_color_names"]):
                        prompt = prompt + f"\nIn the {_color} bounding box region: {_phrase}."

                    # first frame -> input_image (drives the start-image y and the CLIP feature)
                    ff = pixel_values[0].float().clamp(-1, 1)
                    first_frame_pil = Image.fromarray(((ff + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
                    control_camera_video = rearrange(control_camera_values, "f c h w -> 1 c f h w").contiguous()  # [1, 6, F, H, W]
                    control_layout_video = rearrange(data["bbox_canvas"], "f c h w -> 1 c f h w").contiguous()    # [1, 3, F, H, W]

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
                    # The logged video carries the conditioning boxes drawn on every frame, so the
                    # layout can be judged directly in the wandb panel; the last frame is also
                    # logged as a still for quick scanning.
                    video_np, last_frame_pil = _draw_bboxes_on_video(
                        video_np, data.get("bboxes"), data["bbox_color_names"])
                    if args.validation_save_local:
                        _sdir = os.path.join(args.output_dir, "sample")
                        os.makedirs(_sdir, exist_ok=True)
                        _v = torch.from_numpy(video_np).float().div(255).permute(1, 0, 2, 3).unsqueeze(0)
                        save_videos_grid(_v, os.path.join(_sdir, f"sample-{global_step}-val{i:03d}.mp4"), fps=16)
                    local_results.append((i, video_np, prompt, last_frame_pil))

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
                log_dict = {"val/videos_layout": [wandb.Video(v, fps=16, format="mp4", caption=c)
                                                  for _, v, c, _ in all_results]}
                imgs = [wandb.Image(f, caption=c) for _, _, c, f in all_results if f is not None]
                if imgs:
                    log_dict["val/lastframe_layout"] = imgs
                accelerator.log(log_dict, step=global_step)
                del log_dict, imgs

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

def _parse_keep_frame_list(spec):
    """'10,20,81' -> [10, 20, 81]; None/'' -> None."""
    if spec is None or not str(spec).strip():
        return None
    return [int(tok) for tok in str(spec).replace(" ", "").split(",") if tok]


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
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
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
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
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
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
            ' "constant", "constant_with_warmup", "cosine_with_min_lr"]. Only the last one decays to'
            ' a non-zero floor (--lr_min); diffusers\' "cosine" always ends at 0.'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    add_lr_schedule_args(parser)
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
        "--dataloader_prefetch_factor",
        type=int,
        default=2,
        help="Batches each worker keeps in flight (torch default 2). Host RAM scales as "
             "num_processes x num_workers x prefetch x batch bytes; a layout sample is ~835 MB fp32 "
             "(pixels + 6-ch Plucker + bbox canvas), so bs8 x 4 workers x prefetch 2 x 2 ranks is "
             "~104 GB in flight and hit the 220 GB cgroup limit. 1 halves that.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
        "--validation_at_start",
        action="store_true",
        help="Also run one validation BEFORE the first training step (captures the init / resumed model at step 0).",
    )
    parser.add_argument(
        "--scheduler_shift",
        type=float,
        default=None,
        help="Override the noise scheduler's shift (scheduler_kwargs, default 5) for BOTH training (noise "
             "distribution -> biased to high noise) AND in-training validation (denoising schedule). Larger = "
             "more high-noise emphasis.",
    )
    parser.add_argument(
        "--validation_save_local",
        action="store_true",
        help="Also write each validation video (and its bbox overlay) as mp4 under <output_dir>/sample/.",
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
        help="Sparsify the layout by dropping this fraction of frames (0 = dense; last frame always kept).",
    )
    parser.add_argument(
        "--layout_keep_frame_list", type=str, default=None,
        help="Comma-separated 1-indexed frame positions whose layout is kept, e.g. '10,20,30,40,50,60,70,81'. "
             "Overrides --layout_drop_rate for the TRAIN set (the random per-window draw cannot express a "
             "fixed schedule). Positions past the clip length collapse onto the last frame.",
    )
    parser.add_argument(
        "--validation_layout_drop_rate", type=float, default=0.0,
        help="Layout sparsity of the VALIDATION set (default 0 = dense, the historical behaviour). "
             "Set 1.0 to validate in last-frame-layout mode, which is what the sparse runs deploy.",
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
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )

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
        log_with=(None if args.report_to in (None, "none", "None") else args.report_to),
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
    _sched_kwargs = OmegaConf.to_container(config['scheduler_kwargs'])
    if args.scheduler_shift is not None:
        print(f"[scheduler] overriding shift {_sched_kwargs.get('shift')} -> {args.scheduler_shift} "
              f"for training noise AND in-training validation")
        _sched_kwargs['shift'] = args.scheduler_shift
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, _sched_kwargs)
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

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        # A 32-ch camera checkpoint is loaded FIRST and the patch_embedding widened after, so the
        # pretrained channels survive and the 16 layout channels start fresh. A checkpoint that is
        # ALREADY 48-ch (any camlayout run, e.g. the dense stage-2 model) needs the opposite order,
        # or load_state_dict rejects it on shape. Decide from the checkpoint itself.
        _ckpt_in_dim = state_dict["patch_embedding.weight"].shape[1]
        _model_in_dim = transformer3d.patch_embedding.weight.shape[1]
        if _ckpt_in_dim != _model_in_dim:
            print(f"patch_embedding: checkpoint in_dim={_ckpt_in_dim} vs model {_model_in_dim}"
                  f" -> expanding BEFORE load")
            _expand_dit_patch_embedding_input_dim(transformer3d, vae.latent_channels)
            _expanded_before_load = True
        else:
            _expanded_before_load = False

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        assert transformer3d.patch_embedding.weight.shape[1] == _ckpt_in_dim
    else:
        _expanded_before_load = False

    # Layout: expand patch_embedding to also ingest the VAE-encoded bbox canvas (channel-concatenated
    # to y). Done AFTER loading the camera weights so the pretrained 32 channels are preserved; the new
    # 16 layout channels start from a fresh Conv3d init and are trained from scratch.
    if not _expanded_before_load:
        _expand_dit_patch_embedding_input_dim(transformer3d, vae.latent_channels)

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
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio

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
        layout_keep_frames=_parse_keep_frame_list(args.layout_keep_frame_list),
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
            layout_drop_rate=args.validation_layout_drop_rate,
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
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = build_lr_scheduler(args, optimizer, accelerator.num_processes)

    # Prepare everything with our `accelerator`.
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    clip_image_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
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

            pkl_path = os.path.join(os.path.join(args.output_dir, path), "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
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

    # Validation BEFORE any training step -> shows the init (or resumed) model at step 0.
    if args.validation_at_start and val_dataset is not None:
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
            initial_global_step,
        )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)
                clip_pixel_values = batch['clip_pixel_values'].cpu()
                for idx, (clip_pixel_value, text) in enumerate(zip(clip_pixel_values, texts)):
                    name = gif_name[:10] if text != '' else f'{global_step}-{idx}'
                    Image.fromarray(np.uint8(clip_pixel_value)).save(f"{args.output_dir}/sanity_check/clip_{name}.png")
                bbox_canvas_sc = rearrange(batch['bbox_canvas'].cpu(), "b f c h w -> b c f h w")
                for idx, (bc, text) in enumerate(zip(bbox_canvas_sc, texts)):
                    name = gif_name[:10] if text != '' else f'{global_step}-{idx}'
                    save_videos_grid(bc[None], f"{args.output_dir}/sanity_check/bbox_{name}.gif", rescale=True)

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

                    # if args.train_mode == "camera":
                    # start-image conditioning y: first-frame latent placed at temporal position 0
                    ref_latents = _batch_encode_vae(ref_pixel_values)
                    camera_y = torch.zeros_like(latents)
                    camera_y[:, :, :1] = ref_latents
                    # layout conditioning: VAE-encode the colored bbox canvas and channel-concat to y
                    bbox_latents = _batch_encode_vae(bbox_canvas)
                    camera_y = torch.cat([camera_y, bbox_latents], dim=1)  # [b, 32, f_lat, h, w]
                    # camera control: Plucker -> control_adapter input, temporally compressed by 4
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

                _texts, _phr, _cols = batch['text'], batch['bbox_phrases'], batch['bbox_color_names']
                del pixel_values, bbox_canvas, control_camera_values, ref_pixel_values, clip_pixel_values
                batch = None

                # Augment each prompt with per-object "In the <color> bounding box region: <phrase>."
                # (CamTraj WanVideoUnit_PromptEmbedder) so the model can tie colored boxes to objects.
                layout_texts = []
                for _t, _phrases, _colors in zip(_texts, _phr, _cols):
                    for _phrase, _color in zip(_phrases, _colors):
                        _t = _t + f"\nIn the {_color} bounding box region: {_phrase}."
                    layout_texts.append(_t)

                with torch.no_grad():
                    prompt_ids = tokenizer(
                        layout_texts,
                        padding="max_length",
                        max_length=args.tokenizer_max_length, 
                        truncation=True, 
                        add_special_tokens=True, 
                        return_tensors="pt"
                    )
                    text_input_ids = prompt_ids.input_ids
                    prompt_attention_mask = prompt_ids.attention_mask

                    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                    prompt_embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                if args.low_vram:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Sample a random timestep for each image
                    # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    # timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                    indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Add noise
                target = noise - latents
                
                target_shape = (vae.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )

                # Predict the noise residual
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    # if args.train_mode == "camera":
                    y = camera_y
                    y_camera = control_camera_latents
                    # elif args.train_mode == "i2v":
                    #     y = inpaint_latents
                    #     y_camera = None
                    # else:
                    #     y = None
                    #     y_camera = None
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=timesteps,
                        seq_len=seq_len,
                        y=y,
                        y_camera=y_camera,
                        clip_fea=clip_context,
                    )
                
                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
                    mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * mask
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                    final_loss = masked_loss.mean()
                    return final_loss
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                    trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                    max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                    if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                        actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                    else:
                        actual_max_grad_norm = max_grad_norm

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

                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    # This list holds references to every .grad tensor. Without dropping it,
                    # optimizer.zero_grad(set_to_none=True) leaves ~2.4 GB of last step's grads alive
                    # through the whole next step, until the list is reassigned above.
                    del trainable_params_grads
                    if args.report_model_info and accelerator.is_main_process:
                        accelerator.log(
                            {
                                'gradients/norm_sum': float(norm_sum),
                                'gradients/actual_max_grad_norm': float(actual_max_grad_norm),
                            },
                            step=global_step,
                        )
                        
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "train_lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
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
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if val_dataset is not None and (epoch + 1) % args.validation_epochs == 0:
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
    if accelerator.is_main_process:
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
