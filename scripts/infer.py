#!/usr/bin/env python3
"""Generate one video with LIFT from a first frame, a camera trajectory and (optionally) a last-frame layout.

    python scripts/infer.py \
        --model_name  /path/to/Wan2.1-Fun-V1.1-1.3B-Control-Camera \
        --transformer_dir /path/to/LIFT/checkpoint/transformer \
        --image   examples/example1/first_frame.png \
        --camera  examples/example1/camera.npz \
        --layout  examples/example1/layout_lastframe.json \
        --prompt_file examples/example1/caption.txt \
        --output  outputs/demo.mp4

Inputs
  --image    first frame, any resolution. It is resize-to-cover + centre-cropped to --resolution, and
             the SAME geometry is applied to the camera intrinsics and the layout boxes, so everything
             stays registered with the pixels (see videox_fun/data/utils.py::cover_crop_geometry).
  --camera   .npz with `extrinsic` (N,3,4) or (N,4,4) world-to-camera matrices and `intrinsic` (N,3,3)
             pixel-unit camera matrices *at the resolution of --image*; one entry per output frame
             (N == --num_frames; otherwise poses are resampled uniformly).
  --layout   JSON `{"instances": [{"id": 0, "category": "couch", "bbox": [x0, y0, x1, y1], "caption": "..."}, ...]}`
             with boxes in the pixel coordinates of --image, describing where each object should be in
             the LAST frame. Omit it (or pass --camera_only) for camera-only generation.
  --layout_dense  dense per-frame layout, for the dense-layout teacher checkpoint: JSON
             `{"instances": [{"id", "category", "caption", "bboxes": {"<frame>": [x0, y0, x1, y1], ...}}, ...]}`
             with a box on every output frame (0-based index) in which the object is visible; an object
             that is out of view in a frame simply has no box there.
  --prompt   global caption (or --prompt_file). Per-object sentences "In the <colour> bounding box
             region: <caption>." are appended automatically, exactly as in training.

The layout is rendered as a coloured-outline canvas on black, blank on every frame except the last (a dense
layout is drawn on every frame instead), and fed to the model through the 16 extra input channels of the LIFT transformer (`in_dim=48`). This is the
"dual-condition" mode of the paper; --camera_only is the "single-condition" mode of the same weights.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn.functional as nnf
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from videox_fun.data.dataset_camlayout import COLOR_LIST
from videox_fun.data.utils import cover_crop_geometry, process_pose_params, read_camera_npz
from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel, WanTransformer3DModel)
from videox_fun.pipeline import WanCamLayoutPipeline
from videox_fun.utils.utils import filter_kwargs, save_videos_grid

MAX_OBJS = 5            # the released model was trained with at most 5 boxes per clip
MIN_BOX_AREA = 0.001    # boxes smaller than this fraction of the frame are dropped (same as training)

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, walking backwards"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", default="models/Wan2.1-Fun-V1.1-1.3B-Control-Camera", help="base Wan2.1-Fun-V1.1-1.3B-Control-Camera dir (VAE / T5 / CLIP / tokenizer)")
    p.add_argument("--transformer_dir", default=None, help="LIFT checkpoint: <ckpt>/transformer (config.json + safetensors, in_dim 48). Omit = base model, camera-only")
    p.add_argument("--config_path", default=os.path.join(ROOT, "config", "wan2.1", "wan_civitai.yaml"))
    p.add_argument("--image", required=True, help="first frame (png/jpg)")
    p.add_argument("--camera", required=True, help="camera .npz (extrinsic + intrinsic at the image resolution)")
    p.add_argument("--layout", default=None, help="last-frame layout json; omit for camera-only")
    p.add_argument("--layout_dense", default=None, help="dense per-frame layout json (a box on every frame the object is visible); use with the dense-layout teacher checkpoint")
    p.add_argument("--camera_only", action="store_true", help="ignore --layout: blank canvas, plain caption")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt_file", default=None)
    p.add_argument("--output", required=True, help="output .mp4")
    p.add_argument("--resolution", type=int, nargs=2, default=[352, 640], metavar=("H", "W"), help="the released model was trained at 352x640")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=48)
    return p.parse_args()


def build_pipeline(args, device, dtype=torch.bfloat16):
    config = OmegaConf.load(args.config_path)
    base = args.model_name
    tf_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    tf_dir = args.transformer_dir or os.path.join(base, tf_kwargs.get("transformer_subpath", "./"))
    transformer = WanTransformer3DModel.from_pretrained(
        tf_dir, transformer_additional_kwargs=tf_kwargs, low_cpu_mem_usage=True, torch_dtype=dtype)
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(base, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"])).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(base, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")))
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(base, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=dtype).eval()
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(base, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder")),
    ).to(device=device, dtype=dtype).eval()
    sk = OmegaConf.to_container(config["scheduler_kwargs"])
    sk["shift"] = args.shift
    scheduler = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler, sk))
    pipe = WanCamLayoutPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
                                transformer=transformer, scheduler=scheduler,
                                clip_image_encoder=clip_image_encoder)
    return pipe.to(device=device), vae


def cover_crop_image(img, H, W):
    """PIL image -> PIL image of exactly HxW: resize so the frame covers HxW, then centre-crop.

    Uses the SAME ops as the training dataset (WanFunCameraControlDataset._cover_crop: bilinear
    F.interpolate without antialiasing, [-1, 1] range, uint8 truncation), so the conditioning frame
    is preprocessed exactly like the frames the model was trained and evaluated on."""
    arr = np.array(img.convert("RGB"))
    _, rh, rw, top, left = cover_crop_geometry(arr.shape[:2], (H, W))
    x = torch.from_numpy(arr).permute(2, 0, 1).float()[None] / 255.0
    x = nnf.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)[0, :, top:top + H, left:left + W]
    x = (x * 2.0 - 1.0).clamp(-1, 1)
    return Image.fromarray(((x + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).numpy())


def resize_bbox(bbox, orig_hw, target_hw):
    """Same arithmetic as WanFunCamLayoutControlDataset.resize_bbox: scale + integer crop offset, clip, area filter."""
    x0, y0, x1, y1 = bbox
    scale, _, _, top, left = cover_crop_geometry(orig_hw, target_hw)
    x0, y0, x1, y1 = round(x0 * scale) - left, round(y0 * scale) - top, round(x1 * scale) - left, round(y1 * scale) - top
    th, tw = target_hw
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, tw), min(y1, th)
    if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) / (tw * th) < MIN_BOX_AREA:
        return None
    return x0, y0, x1, y1


def layout_items(instances, orig_hw, target_hw):
    """instances: [{"id", "bbox": [x0,y0,x1,y1] (pixels of the orig_hw image), "caption"|"category"}, ...]
    -> list of (id, caption, colour_name, bbox_in_target_pixels), colours in COLOR_LIST order."""
    boxes = []
    for o in instances:
        bb = resize_bbox(o["bbox"], orig_hw, target_hw)
        if bb is None:
            print(f"[layout] instance {o.get('id')} ({o.get('category')}) dropped: outside the crop or too small")
            continue
        boxes.append((o.get("id"), o.get("caption") or o.get("category", ""), bb))
    if len(boxes) > len(COLOR_LIST):
        raise ValueError(f"{len(boxes)} boxes but only {len(COLOR_LIST)} colours")
    if len(boxes) > MAX_OBJS:
        print(f"[layout] {len(boxes)} boxes; the released model was trained with at most {MAX_OBJS}, results may degrade")
    colours = list(COLOR_LIST.keys())
    return [(i, cap, colours[k], bb) for k, (i, cap, bb) in enumerate(boxes)]


def load_layout(path, orig_hw, target_hw):
    """layout_lastframe.json -> layout_items(...)."""
    return layout_items(json.load(open(path))["instances"], orig_hw, target_hw)


def _draw_frame(H, W, boxes):
    """boxes: [(colour_name, (x0, y0, x1, y1)), ...] -> [3, H, W] in [-1, 1]; large boxes first so small ones stay on top."""
    img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    for colour, (x0, y0, x1, y1) in sorted(boxes, key=lambda it: -(it[1][2] - it[1][0]) * (it[1][3] - it[1][1])):
        draw.rectangle([x0, y0, x1, y1], outline=COLOR_LIST[colour], width=4)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0


def draw_canvas(H, W, num_frames, items):
    """[F, 3, H, W] in [-1, 1]: black everywhere, the coloured outlines only on the LAST frame."""
    canvas = torch.full((num_frames, 3, H, W), -1.0)
    canvas[-1] = _draw_frame(H, W, [(colour, bb) for _, _, colour, bb in items])
    return canvas


def dense_layout_items(instances, orig_hw, target_hw):
    """Dense per-frame variant of layout_items: instances carry "bboxes": {"<frame>": [x0,y0,x1,y1]} ->
    list of (id, caption, colour_name, {frame: bbox_in_target_pixels}); frames whose box falls outside
    the crop are dropped, objects with no box left are dropped."""
    objs = []
    for o in instances:
        per_frame = {}
        for f, bb in (o.get("bboxes") or {}).items():
            rb = resize_bbox(bb, orig_hw, target_hw)
            if rb is not None:
                per_frame[int(f)] = rb
        if per_frame:
            objs.append((o.get("id"), o.get("caption") or o.get("category", ""), per_frame))
        else:
            print(f"[layout] instance {o.get('id')} ({o.get('category')}) dropped: no box inside the crop")
    if len(objs) > len(COLOR_LIST):
        raise ValueError(f"{len(objs)} objects but only {len(COLOR_LIST)} colours")
    if len(objs) > MAX_OBJS:
        print(f"[layout] {len(objs)} objects; the released models were trained with at most {MAX_OBJS}, results may degrade")
    colours = list(COLOR_LIST.keys())
    return [(i, cap, colours[k], pf) for k, (i, cap, pf) in enumerate(objs)]


def draw_canvas_dense(H, W, num_frames, dense_items):
    """[F, 3, H, W] in [-1, 1]: each object's box drawn on every frame it has one (dense per-frame layout)."""
    canvas = torch.full((num_frames, 3, H, W), -1.0)
    for f in range(num_frames):
        boxes = [(colour, pf[f]) for _, _, colour, pf in dense_items if f in pf]
        if boxes:
            canvas[f] = _draw_frame(H, W, boxes)
    return canvas


def num_frames_4k1(n):
    """The VAE compresses time by 4, so the frame count must be 4k + 1."""
    return int((int(n) - 1) // 4 * 4) + 1


def camera_condition(camera_path, orig_hw, H, W, F):
    """camera .npz -> Plucker embedding [1, 6, F, H, W]: intrinsics rescaled for the cover-crop of an
    orig_hw image to HxW, poses resampled uniformly to F frames."""
    coords = read_camera_npz(camera_path, orig_size=orig_hw, target_size=(H, W))
    idx = np.linspace(0, len(coords) - 1, F).round().astype(int)
    plucker = process_pose_params([coords[i] for i in idx], width=W, height=H, apply_aspect_fit=False)  # [F, H, W, 6]
    return rearrange(plucker.permute(0, 3, 1, 2).contiguous(), "f c h w -> 1 c f h w")


def generate(pipe, image, camera_path, prompt, instances=None, H=352, W=640, num_frames=81, seed=42,
             num_inference_steps=50, guidance_scale=6.0, shift=5.0, callback_on_step_end=None, dense_instances=None):
    """One LIFT generation. image: PIL first frame (any resolution); camera_path: .npz at that image's
    resolution; instances: last-frame layout instances in the image's pixels, or None / [] for
    camera-only; dense_instances: a dense per-frame layout instead (a box on every frame the object is
    visible, for the dense-layout teacher). Returns the pipeline's videos tensor [1, 3, F, H, W]. The CLI (main) and
    the UI backend (UI/generate_single_video.py) both go through here, so they stay identical."""
    device = pipe.transformer.device
    F = num_frames_4k1(num_frames)
    orig_hw = (image.height, image.width)
    first = cover_crop_image(image, H, W)
    cam = camera_condition(camera_path, orig_hw, H, W, F)

    in_dim = pipe.transformer.patch_embedding.in_channels
    if (instances or dense_instances) and in_dim != 48:
        print(f"[layout] transformer has in_dim={in_dim} (no layout channels); running camera-only")
    items, dense_items = [], []
    if in_dim == 48 and dense_instances:
        dense_items = dense_layout_items(dense_instances, orig_hw, (H, W))
        lay = rearrange(draw_canvas_dense(H, W, F, dense_items), "f c h w -> 1 c f h w")
        for _, cap, colour, _ in dense_items:
            prompt = prompt + f"\nIn the {colour} bounding box region: {cap}."
    elif in_dim == 48 and instances:
        items = layout_items(instances, orig_hw, (H, W))
        lay = rearrange(draw_canvas(H, W, F, items), "f c h w -> 1 c f h w")
        for _, cap, colour, _ in items:
            prompt = prompt + f"\nIn the {colour} bounding box region: {cap}."
    else:
        lay = rearrange(torch.full((F, 3, H, W), -1.0), "f c h w -> 1 c f h w") if in_dim == 48 else None
    print(f"[prompt]\n{prompt}\n")
    for i, cap, colour, bb in items:
        print(f"[layout] {colour:8s} id={i} box={list(bb)}  {cap}")
    for i, cap, colour, pf in dense_items:
        print(f"[layout] {colour:8s} id={i} frames {min(pf)}..{max(pf)} ({len(pf)} boxes)  {cap}")

    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        return pipe(prompt, num_frames=F, negative_prompt=NEGATIVE_PROMPT, height=H, width=W,
                    generator=generator, input_image=first, control_camera_video=cam,
                    control_layout_video=lay, num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, shift=shift, callback_on_step_end=callback_on_step_end).videos


def save_video(videos, path, fps=16):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    save_videos_grid(videos, path, fps=fps)


def main():
    args = parse_args()
    H, W = args.resolution
    if num_frames_4k1(args.num_frames) != args.num_frames:
        print(f"[frames] {args.num_frames} -> {num_frames_4k1(args.num_frames)} (must be 4k+1)")
    prompt = args.prompt if args.prompt is not None else open(args.prompt_file).read().strip()
    instances = dense_instances = None
    if args.layout_dense and not args.camera_only:
        dense_instances = json.load(open(args.layout_dense))["instances"]
    elif args.layout and not args.camera_only:
        instances = json.load(open(args.layout))["instances"]

    pipe, _ = build_pipeline(args, torch.device("cuda"))
    out = generate(pipe, Image.open(args.image), args.camera, prompt, instances, H=H, W=W,
                   num_frames=args.num_frames, seed=args.seed, num_inference_steps=args.num_inference_steps,
                   guidance_scale=args.guidance_scale, shift=args.shift, dense_instances=dense_instances)
    save_video(out, args.output, fps=args.fps)
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
