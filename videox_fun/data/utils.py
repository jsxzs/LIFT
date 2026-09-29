import csv
import gc
import io
import json
import math
import os
import random
from contextlib import contextmanager
from random import shuffle
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
try:
    from decord import VideoReader
    _HAS_DECORD = True
except Exception:
    # decord has no linux-aarch64 wheel (GH200); fall back to a cv2-based reader below.
    VideoReader = None
    _HAS_DECORD = False
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

VIDEO_READER_TIMEOUT = 20

def get_random_mask(shape, image_start_only=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]) 
        else:
            mask_index = np.random.choice([0, 1, 7, 8], p = [0.2, 0.7, 0.05, 0.05])
        if mask_index == 0:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)
            mask[:, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 1:
            mask[:, :, :, :] = 1
        elif mask_index == 2:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 3:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
        elif mask_index == 4:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)

            mask_frame_before = np.random.randint(0, f // 2)
            mask_frame_after = np.random.randint(f // 2, f)
            mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 5:
            mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
        elif mask_index == 6:
            num_frames_to_mask = random.randint(1, max(f // 2, 1))
            frames_to_mask = random.sample(range(f), num_frames_to_mask)

            for i in frames_to_mask:
                block_height = random.randint(1, h // 4)
                block_width = random.randint(1, w // 4)
                top_left_y = random.randint(0, h - block_height)
                top_left_x = random.randint(0, w - block_width)
                mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
        elif mask_index == 7:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
            b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

            for i in range(h):
                for j in range(w):
                    if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                        mask[:, :, i, j] = 1
        elif mask_index == 8:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
            for i in range(h):
                for j in range(w):
                    if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                        mask[:, :, i, j] = 1
        elif mask_index == 9:
            for idx in range(f):
                if np.random.rand() > 0.5:
                    mask[idx, :, :, :] = 1
        else:
            raise ValueError(f"The mask_index {mask_index} is not define")
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    return mask

class _CV2Batch:
    """Mimics decord's get_batch(...) return (has .asnumpy())."""
    def __init__(self, arr):
        self._arr = arr
    def asnumpy(self):
        return self._arr


class _CV2VideoReader:
    """decord.VideoReader drop-in backed by cv2 (for aarch64 where decord has no wheel).
    Supports len(vr) and vr.get_batch(indices).asnumpy() -> [N, H, W, 3] uint8 RGB."""
    def __init__(self, uri, *args, **kwargs):
        self.uri = uri
        self.cap = cv2.VideoCapture(uri)
        self._n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self._n

    def get_batch(self, indices):
        idxs = [int(i) for i in indices]
        want = set(idxs)
        max_i = max(idxs)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # idempotent: read from the start
        cache, pos = {}, 0
        while pos <= max_i:
            ret, frame = self.cap.read()
            if not ret:
                break
            if pos in want:
                cache[pos] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pos += 1
        last = None
        frames = []
        for i in idxs:
            f = cache.get(i, last)
            if f is None:
                raise RuntimeError(f"cv2 reader could not read frame {i} of {self.uri}")
            frames.append(f); last = f
        return _CV2Batch(np.stack(frames))

    def __del__(self):
        try:
            self.cap.release()
        except Exception:
            pass


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs) if _HAS_DECORD else _CV2VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames

def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)
    
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

def padding_image(images, new_width, new_height):
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    aspect_ratio = images.width / images.height
    if new_width / new_height > 1:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)
    else:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)

    resized_img = images.resize((new_img_width, new_img_height))

    paste_x = (new_width - new_img_width) // 2
    paste_y = (new_height - new_img_height) // 2

    new_image.paste(resized_img, (paste_x, paste_y))

    return new_image

def resize_image_with_target_area(img: Image.Image, target_area: int = 1024 * 1024) -> Image.Image:
    """
    将 PIL 图像缩放到接近指定像素面积（target_area），保持原始宽高比，
    并确保新宽度和高度均为 32 的整数倍。

    参数:
        img (PIL.Image.Image): 输入图像
        target_area (int): 目标像素总面积，例如 1024*1024 = 1048576

    返回:
        PIL.Image.Image: Resize 后的图像
    """
    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        raise ValueError("Input image has zero width or height.")

    ratio = orig_w / orig_h
    ideal_width = math.sqrt(target_area * ratio)
    ideal_height = ideal_width / ratio

    new_width = round(ideal_width / 32) * 32
    new_height = round(ideal_height / 32) * 32

    new_width = max(32, new_width)
    new_height = max(32, new_height)

    new_width = int(new_width)
    new_height = int(new_height)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    return resized_img

class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def custom_meshgrid(*args):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def get_relative_pose(cam_params):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    # dim=-1 explicitly: bare torch.cross uses "the first dimension of size 3", which lands on the
    # last axis here only because B/V/HW happen not to be 3. A batch of 3 would silently cross the
    # wrong axis. (o x d is the moment; the pair is stored as [m | d], the CameraCtrl convention.)
    rays_dxo = torch.linalg.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

def process_pose_file(pose_file_path, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False):
    """Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    with open(pose_file_path, 'r') as f:
        poses = f.readlines()

    poses = [pose.strip().split(' ') for pose in poses[1:]]
    cam_params = [[float(x) for x in pose] for pose in poses]
    if return_poses:
        return cam_params
    else:
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        sample_wh_ratio = width / height
        pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / width
        else:
            resized_ori_h = width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / height

        intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                                for cam_param in cam_params], dtype=np.float32)

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding

def cover_crop_geometry(orig_size, target_size):
    """Geometry of the resize-to-cover + centre-crop used everywhere in this codebase.

    ``orig_size`` / ``target_size`` are ``(H, W)``. Returns ``(scale, resized_h, resized_w, top, left)``.

    ``top`` / ``left`` are INTEGERS, because that is what an actual pixel crop can do -- the frame
    tensor is sliced ``x[..., top:top+H, left:left+W]``. Anything that has to stay registered with
    those pixels (camera intrinsics, bbox tracks, point tracks) must use the same integer offset:
    computing it as a float ``/2.0`` instead silently shifts that signal half a pixel against the
    frames whenever ``resized - target`` is odd. Keep every consumer on this one function so the
    three copies of this arithmetic cannot drift apart again.
    """
    orig_h, orig_w = orig_size
    tgt_h, tgt_w = target_size
    scale = max(tgt_w / orig_w, tgt_h / orig_h)
    resized_h, resized_w = round(orig_h * scale), round(orig_w * scale)
    top = (resized_h - tgt_h) // 2
    left = (resized_w - tgt_w) // 2
    return scale, resized_h, resized_w, top, left


def read_camera_npz(npz_path, orig_size, target_size, normalized_intrinsics=False):
    """Load a CamLayout camera ``.npz`` (``extrinsic``: (N,3,4)|(N,4,4) world-to-camera,
    ``intrinsic``: (N,3,3) in pixels at the original resolution) and convert it to the
    19-element pose list expected by :class:`Camera` / :func:`process_pose_params`.

    The intrinsics are rescaled to match a **resize-to-cover + center-crop** of the
    video (``scale = max(tgt_w/orig_w, tgt_h/orig_h)`` followed by a centered crop) --
    i.e. exactly the geometric transform applied to the frames, so the camera rays stay
    consistent with the pixels. This mirrors ``read_camera_traj`` in the diffsynth-based
    CamTraj-VideoGen project. ``orig_size`` / ``target_size`` are ``(H, W)`` tuples.

    Returns a list of ``[0, fx, fy, cx, cy, 0, 0, *w2c_3x4]`` where fx/fy/cx/cy are
    normalised by the *target* width/height (so ``process_pose_params(..., apply_aspect_fit=False)``
    reproduces the pixel intrinsics of the cropped frame).
    """
    cam = np.load(npz_path)
    extrinsics = cam["extrinsic"]                          # (N, 3|4, 4), world-to-camera
    intrinsics = cam["intrinsic"].astype(np.float64).copy()  # (N, 3, 3), pixel units
    assert extrinsics.shape[0] == intrinsics.shape[0]

    orig_h, orig_w = orig_size
    tgt_h, tgt_w = target_size

    # Some npz variants store fx/fy/cx/cy already normalised to [0, 1]; scale back to pixels.
    if normalized_intrinsics:
        intrinsics[:, 0, 0] *= orig_w
        intrinsics[:, 1, 1] *= orig_h
        intrinsics[:, 0, 2] *= orig_w
        intrinsics[:, 1, 2] *= orig_h

    # Resize-to-cover then center-crop -- the SAME integer offsets the frames are cropped with.
    scale, resized_h, resized_w, crop_top, crop_left = cover_crop_geometry((orig_h, orig_w), (tgt_h, tgt_w))

    coordinates = []
    for i in range(extrinsics.shape[0]):
        fx = intrinsics[i, 0, 0] * scale / tgt_w
        fy = intrinsics[i, 1, 1] * scale / tgt_h
        cx = (intrinsics[i, 0, 2] * scale - crop_left) / tgt_w
        cy = (intrinsics[i, 1, 2] * scale - crop_top) / tgt_h
        coordinates.append([0, fx, fy, cx, cy, 0, 0] + extrinsics[i][:3, :].flatten().tolist())
    return coordinates


def process_pose_params(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', apply_aspect_fit=True):
    """Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py

    ``apply_aspect_fit``: when True (default, original behaviour) the fx/fy are adjusted
    for the sample-vs-pose aspect ratio, assuming the intrinsics were normalised against
    ``original_pose_width/height`` (the CameraCtrl .txt convention). Set it to False when
    the intrinsics are already resolved for the exact target crop -- e.g. coordinates
    produced by :func:`read_camera_npz` -- so no extra aspect correction is applied.
    """
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    if apply_aspect_fit:
        sample_wh_ratio = width / height
        pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / width
        else:
            resized_ori_h = width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / height

    intrinsic = np.asarray([[cam_param.fx * width,
                            cam_param.fy * height,
                            cam_param.cx * width,
                            cam_param.cy * height]
                            for cam_param in cam_params], dtype=np.float32)

    K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
    c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
    plucker_embedding = plucker_embedding[None]
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
    return plucker_embedding