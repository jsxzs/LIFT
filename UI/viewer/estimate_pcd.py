"""Run MoGe-2 on the first frame of a clip and save a world-space point cloud.

Output: <clip>/pcd_moge.npz with keys
    points  (M, 3)  float32  — XYZ in world coords
    colors  (M, 3)  uint8    — RGB
    info    json blob with model, intrinsic, scale used

Usage:
    python estimate_pcd.py --clip /path/to/<uid>
    python estimate_pcd.py --clip ...  --convention w2c  --model Ruicheng/moge-2-vitl
"""

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from moge.model.v2 import MoGeModel


def compute_c2w(extr_3x4: np.ndarray, convention: str) -> np.ndarray:
    M = np.eye(4)
    M[:3, :4] = extr_3x4
    if convention == "c2w":
        return M
    return np.linalg.inv(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, type=Path)
    ap.add_argument("--model", default="Ruicheng/moge-2-vitl")
    ap.add_argument("--convention", choices=["c2w", "w2c"], default="w2c")
    ap.add_argument("--frame", default=0, type=int, help="which video frame to lift")
    ap.add_argument("--max-points", default=300_000, type=int,
                    help="random-subsample if more valid points than this")
    ap.add_argument("--use-clip-intrinsic", action="store_true",
                    help="pass the camera_da3 intrinsic as fov_x hint to MoGe.infer()")
    args = ap.parse_args()

    clip_dir = args.clip.resolve()
    print(f"[pcd] clip = {clip_dir}")

    cam_file = next((clip_dir / n for n in ("camera_da3.npz", "camera.npz") if (clip_dir / n).exists()), None)
    if cam_file is None:
        raise FileNotFoundError(f"no camera_da3.npz / camera.npz in {clip_dir}")
    npz = np.load(cam_file)
    extr = np.asarray(npz["extrinsic"], dtype=np.float64)
    intr = np.asarray(npz["intrinsic"], dtype=np.float64)
    mp4 = clip_dir / "0.mp4"
    if not mp4.exists():
        cands = sorted(clip_dir.glob("*.mp4"))
        if not cands:
            raise FileNotFoundError(f"no .mp4 found in {clip_dir}")
        mp4 = cands[0]
    frames = iio.imread(mp4)
    N, H, W = frames.shape[:3]
    assert 0 <= args.frame < N, f"frame {args.frame} out of range (N={N})"

    img_rgb = frames[args.frame]
    print(f"[pcd] image {W}x{H}, lifting frame {args.frame}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pcd] loading {args.model} on {device}...")
    model = MoGeModel.from_pretrained(args.model).to(device).eval()

    img_t = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    img_t = img_t.to(device)

    fov_x_deg = None
    if args.use_clip_intrinsic:
        K0 = intr[args.frame]
        fov_x_deg = float(np.degrees(2.0 * np.arctan2(W / 2.0, K0[0, 0])))
        print(f"[pcd] using fov_x hint = {fov_x_deg:.2f}°")

    with torch.no_grad():
        kwargs = {}
        if fov_x_deg is not None:
            kwargs["fov_x"] = fov_x_deg
        out = model.infer(img_t, **kwargs)

    # out["points"]: (H, W, 3) in OpenCV camera coords, metric for moge-2
    points_cam = out["points"].detach().cpu().numpy()  # (H, W, 3)
    mask = out["mask"].detach().cpu().numpy().astype(bool)  # (H, W)
    print(f"[pcd] valid pixels: {mask.sum()} / {mask.size}")

    # Flatten and filter
    pts = points_cam[mask]           # (M, 3) in camera coords
    rgb = img_rgb[mask]              # (M, 3) uint8

    # Transform to world coords using frame's extrinsic
    c2w = compute_c2w(extr[args.frame], args.convention)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    pts_world = (c2w @ pts_h.T).T[:, :3]
    print(f"[pcd] point bbox (world): "
          f"min={pts_world.min(axis=0)}, max={pts_world.max(axis=0)}")

    # Optional subsample for transport
    if pts_world.shape[0] > args.max_points:
        sel = np.random.default_rng(0).choice(
            pts_world.shape[0], size=args.max_points, replace=False
        )
        pts_world = pts_world[sel]
        rgb = rgb[sel]
        print(f"[pcd] subsampled to {len(pts_world)} points")

    out_path = clip_dir / "pcd_moge.npz"
    info = {
        "model": args.model,
        "frame": args.frame,
        "convention": args.convention,
        "fov_x_deg_hint": fov_x_deg,
        "moge_intrinsics_norm": out["intrinsics"].detach().cpu().numpy().tolist(),
        "extrinsic_used": extr[args.frame].tolist(),
        "image_hw": [int(H), int(W)],
    }
    np.savez(
        out_path,
        points=pts_world.astype(np.float32),
        colors=rgb.astype(np.uint8),
        info=np.array(json.dumps(info), dtype=object),
    )
    print(f"[pcd] saved {out_path}  ({len(pts_world)} pts, "
          f"{out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
