"""In-process video generation backend for the LIFT editor (viewer/serve.py).

A thin adapter: it only turns the editor's clip directory into the arguments of
``scripts/infer.py::generate`` and calls it, so a video generated from the UI is exactly what
``python scripts/infer.py`` would produce for the same first frame, camera and last-frame boxes.

Interface expected by the viewer:

    g = VideoGenerator(ckpt_path=<LIFT transformer dir or None>)
    g.load(log=print)                                   # once, in a background thread
    g.generate(clip_dir=..., out_path=..., seed=..., prompt=..., num_frames=81, height=352,
               width=640, fps=16, num_inference_steps=50, cfg_scale=6.0, progress_cb=None)

Files read from the clip directory the editor writes:
    input_image.png | first_frame.png | *.mp4   first frame (a png wins; else frame 0 of the video)
    camera_da3_edited.npz | camera_da3.npz | camera.npz   camera trajectory (the edited one wins)
    layout_edited.json               the editor's boxes; only the LAST annotated frame is used
                                     (LIFT conditions on a last-frame layout)
    caption.txt                      default prompt when the request carries none

Environment:
    WANGEN_CKPT       LIFT transformer dir (default <repo>/models/LIFT/transformer)
    LIFT_BASE_MODEL   Wan2.1-Fun-V1.1-1.3B-Control-Camera dir
                      (default <repo>/models/Wan2.1-Fun-V1.1-1.3B-Control-Camera)
"""
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import infer as I  # noqa: E402  (scripts/infer.py)


class _Args:
    pass


def first_frame(clip_dir):
    for name in ("input_image.png", "first_frame.png"):
        if (clip_dir / name).exists():
            return Image.open(clip_dir / name).convert("RGB")
    import cv2
    cands = [clip_dir / "0.mp4"] if (clip_dir / "0.mp4").exists() else sorted(clip_dir.glob("*.mp4"))
    if not cands:
        raise FileNotFoundError(f"no input_image.png / first_frame.png / *.mp4 in {clip_dir}")
    cap = cv2.VideoCapture(str(cands[0]))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame 0 of {cands[0]}")
    return Image.fromarray(bgr[:, :, ::-1])


def camera_file(clip_dir):
    for name in ("camera_da3_edited.npz", "camera_da3.npz", "camera.npz"):
        if (clip_dir / name).exists():
            return clip_dir / name
    raise FileNotFoundError(f"no camera .npz in {clip_dir}")


def layout_instances(clip_dir, image_hw):
    """layout_edited.json (the editor's format) -> the `instances` list scripts/infer.py takes:
    the boxes of the LAST annotated frame, in the pixels of the first frame."""
    p = clip_dir / "layout_edited.json"
    if not p.exists():
        return []
    lay = json.loads(p.read_text())
    frames = lay.get("frames") or {}
    if not frames:
        return []
    ow, oh = lay.get("image_resolution") or [image_hw[1], image_hw[0]]
    sx, sy = image_hw[1] / float(ow), image_hw[0] / float(oh)      # editor pixels -> image pixels
    names = {o["id"]: (o.get("name") or f"object {o['id']}") for o in lay.get("objects", [])}
    out = []
    for b in frames[max(frames, key=lambda k: int(k))]:
        x0, y0, x1, y1 = b["bbox_2d"]
        out.append({"id": b["obj_id"], "bbox": [x0 * sx, y0 * sy, x1 * sx, y1 * sy],
                    "caption": b.get("name") or names.get(b["obj_id"], f"object {b['obj_id']}")})
    return out


class VideoGenerator:
    def __init__(self, ckpt_path=None, base_model=None, config_path=None):
        self.transformer_dir = str(ckpt_path or os.environ.get("WANGEN_CKPT") or REPO / "models" / "LIFT" / "transformer")
        self.base_model = str(base_model or os.environ.get("LIFT_BASE_MODEL")
                              or REPO / "models" / "Wan2.1-Fun-V1.1-1.3B-Control-Camera")
        self.config_path = str(config_path or REPO / "config" / "wan2.1" / "wan_civitai.yaml")
        self.pipe = None

    def load(self, log=print):
        if self.pipe is not None:
            return
        for p, what in ((self.base_model, "base model"), (self.transformer_dir, "LIFT transformer")):
            if not os.path.isdir(p):
                raise FileNotFoundError(f"{what} dir not found: {p}")
        log(f"[gen] loading LIFT pipeline: base={self.base_model} transformer={self.transformer_dir}")
        args = _Args()
        args.model_name, args.transformer_dir, args.config_path, args.shift = self.base_model, self.transformer_dir, self.config_path, 5.0
        self.pipe, _ = I.build_pipeline(args, torch.device("cuda"))
        log(f"[gen] pipeline ready (transformer in_dim={self.pipe.transformer.patch_embedding.in_channels})")

    def generate(self, clip_dir, out_path, seed=42, prompt=None, num_frames=81, height=352, width=640,
                 fps=16, num_inference_steps=50, cfg_scale=6.0, progress_cb=None, **_ignored):
        if self.pipe is None:
            self.load()
        clip_dir = Path(clip_dir)
        cb = progress_cb or (lambda frac, msg: None)
        cb(0.02, "loading inputs…")
        image = first_frame(clip_dir)
        cam = camera_file(clip_dir)
        prompt = (prompt or "").strip()
        if not prompt and (clip_dir / "caption.txt").exists():
            prompt = (clip_dir / "caption.txt").read_text().strip()
        instances = layout_instances(clip_dir, (image.height, image.width))
        total = int(num_inference_steps)

        def _on_step(pipe, step, timestep, kw):
            cb(0.05 + 0.9 * (step + 1) / total, f"sampling {step + 1}/{total}")
            return kw

        print(f"[gen] seed={seed} steps={total} cfg={cfg_scale} boxes={len(instances)} camera={cam.name}")
        cb(0.05, "sampling…")
        out = I.generate(self.pipe, image, str(cam), prompt, instances, H=int(height), W=int(width),
                         num_frames=int(num_frames), seed=int(seed), num_inference_steps=total,
                         guidance_scale=float(cfg_scale), shift=5.0, callback_on_step_end=_on_step)
        cb(0.97, "encoding video…")
        I.save_video(out, str(out_path), fps=int(fps))
        cb(1.0, "done")
        return str(out_path)
