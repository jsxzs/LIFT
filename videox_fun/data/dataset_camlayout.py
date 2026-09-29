
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from func_timeout import FunctionTimedOut, func_timeout
from PIL import Image, ImageDraw
from torch.utils.data.dataset import Dataset

from .utils import (VIDEO_READER_TIMEOUT, Camera, VideoReader_contextmanager,
                    cover_crop_geometry, get_video_reader_batch, process_pose_params,
                    read_camera_npz,)

COLOR_LIST = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "orange": (255, 128, 0),
    "purple": (128, 0, 255),
    "pink": (255, 20, 147),
}


class WanFunCameraControlDataset(Dataset):
    """Camera-control training dataset.
      * frames are resize then center-cropped to ``(H, W)`` (``scale = max(..)``);
      * the camera trajectory is read from the per-clip ``.npz`` and its intrinsics are
        rescaled by the SAME cover+crop (see :func:`read_camera_npz`), then turned into a
        Plucker embedding with ``process_pose_params(..., apply_aspect_fit=False)``.

    Each item returns ``pixel_values`` ``[f, 3, H, W]`` in [-1, 1] and ``control_camera_values``
    ``[f, 6, H, W]`` -- the format the ``control_camera_ref`` training path consumes.
    """

    def __init__(
        self,
        ann_path,
        video_sample_size=(352, 640),
        video_sample_n_frames=81,
        video_sample_stride=1,
        text_drop_ratio=0.0,
        normalized_intrinsics=False,
        data_root=None,
    ):
        print(f"loading annotations from {ann_path} ...")
        with open(ann_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        if isinstance(video_sample_size, int):
            video_sample_size = (video_sample_size, video_sample_size)
        self.height, self.width = int(video_sample_size[0]), int(video_sample_size[1])
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_stride = video_sample_stride
        self.text_drop_ratio = text_drop_ratio
        # True if the camera .npz stores intrinsics normalised by the original frame size
        # (fx/W, fy/H, cx/W, cy/H) instead of in pixels; see read_camera_npz.
        self.normalized_intrinsics = normalized_intrinsics
        # Relative Video_Path / Annotation_Path entries are resolved against data_root, which defaults
        # to the directory holding the CSV (the released LIFT-Vista layout: <root>/{videos,annotations}).
        self.data_root = data_root if data_root is not None else os.path.dirname(os.path.abspath(ann_path))

    def _resolve(self, p):
        return p if (not p or os.path.isabs(p)) else os.path.join(self.data_root, p)

    def __len__(self):
        return self.length

    def _cover_crop(self, frames):
        """frames: np.uint8 [f, h0, w0, 3] -> tensor [f, 3, H, W] in [-1, 1] (resize-to-cover + center-crop)."""
        f, h0, w0, _ = frames.shape
        _, new_h, new_w, top, left = cover_crop_geometry((h0, w0), (self.height, self.width))
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # [f, 3, h0, w0]
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        x = x[:, :, top:top + self.height, left:left + self.width]
        return x * 2.0 - 1.0  # -> [-1, 1]

    def get_batch(self, idx):
        data_info = self.dataset[idx % self.length]
        video_path = self._resolve(data_info["Video_Path"])
        caption = (data_info.get("caption") or "").strip()

        # original resolution written as "WxH" (e.g. "1280x720")
        ow, oh = data_info["resolution"].lower().split("x")
        orig_w, orig_h = int(ow), int(oh)

        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            n = len(video_reader)
            if n == 0:
                raise ValueError(f"Empty video: {video_path}")
            n_sample = self.video_sample_n_frames
            clip_length = min(n, (n_sample - 1) * self.video_sample_stride + 1)
            assert clip_length == 81
            # start_idx = random.randint(0, n - clip_length) if n > clip_length else 0
            start_idx = 0
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, n_sample, dtype=int)

            frames = func_timeout(
                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(video_reader, batch_index)
            )
        pixel_values = self._cover_crop(np.array(frames))  # [f, 3, H, W]

        # camera trajectory (crop-aware intrinsics matching the frame cover+crop)
        cam_file = os.path.join(self._resolve(data_info["Annotation_Path"]), data_info["CameraFile"])
        coordinates = read_camera_npz(
            cam_file,
            orig_size=(orig_h, orig_w),
            target_size=(self.height, self.width),
            normalized_intrinsics=self.normalized_intrinsics,
        )
        # align the per-frame camera poses to the sampled frame indices
        n_cam = len(coordinates)
        coordinates = [coordinates[min(int(i), n_cam - 1)] for i in batch_index]
        plucker = process_pose_params(
            coordinates, width=self.width, height=self.height, apply_aspect_fit=False
        )  # [f, H, W, 6]
        control_camera_values = plucker.permute(0, 3, 1, 2).contiguous()  # [f, 6, H, W]

        # if random.random() < self.text_drop_ratio:
        #     caption = ""
        return pixel_values, control_camera_values, caption

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, control_camera_values, text = self.get_batch(idx)
                return {
                    "pixel_values": pixel_values,
                    "control_camera_values": control_camera_values,
                    "text": text,
                    "data_type": "video",
                    "idx": idx,
                }
            except Exception as e:
                print(f"Error loading data {self.dataset[idx % self.length].get('UID', idx)}: {e}")
                idx = random.randint(0, self.length - 1)

class WanFunCamLayoutControlDataset(WanFunCameraControlDataset):
    """Camera + layout control training dataset.

      * per-object bounding-box tracks are read from ``Annotation_Path/sam3_track_file`` (a SAM3
        track JSON) and rescaled to the target crop with the SAME resize-to-cover + center-crop
        as the frames (:meth:`resize_bbox`);
      * each object assigned a random color, and their boxes are drawn (outline only) on a black canvas per frame
        -> ``bbox_canvas`` ``[f, 3, H, W]`` in [-1, 1].
      * ``bbox_phrases`` / ``bbox_color_names`` are returned
    """

    def __init__(
        self,
        ann_path,
        video_sample_size=(352, 640),
        video_sample_n_frames=81,
        video_sample_stride=1,
        text_drop_ratio=0.0,
        normalized_intrinsics=False,
        max_objs=5,
        bbox_area_threshold=0.02,
        layout_drop_rate=0.0,
        layout_keep_frames=None,
        path_remap=None,
        data_root=None,
    ):
        super().__init__(
            ann_path,
            video_sample_size=video_sample_size,
            video_sample_n_frames=video_sample_n_frames,
            video_sample_stride=video_sample_stride,
            text_drop_ratio=text_drop_ratio,
            normalized_intrinsics=normalized_intrinsics,
            data_root=data_root,
        )
        self.max_objs = max_objs
        self.bbox_area_threshold = bbox_area_threshold
        self.layout_drop_rate = layout_drop_rate
        # Explicit 1-indexed frame positions whose layout survives; overrides layout_drop_rate.
        self.layout_keep_frames = tuple(sorted({int(x) for x in layout_keep_frames})) if layout_keep_frames else None
        self.path_remap = path_remap

    def _remap(self, p):
        if self.path_remap and p:
            return p.replace(self.path_remap[0], self.path_remap[1])
        return p

    def resize_bbox(self, bbox, orig_size, target_size):
        """Rescale a pixel-space bbox from ``orig_size`` (H, W) to ``target_size`` (H, W) under the
        same resize-to-cover + center-crop as the frames, dropping boxes that fall outside or are
        smaller than ``bbox_area_threshold``. Returns ``(x_min, y_min, x_max, y_max)`` or ``None``."""
        x_min, y_min, x_max, y_max = bbox
        orig_h, orig_w = orig_size
        target_h, target_w = target_size

        scale, _, _, crop_y, crop_x = cover_crop_geometry((orig_h, orig_w), (target_h, target_w))
        x_min, y_min, x_max, y_max = round(x_min * scale), round(y_min * scale), round(x_max * scale), round(y_max * scale)
        x_min -= crop_x; y_min -= crop_y; x_max -= crop_x; y_max -= crop_y

        inter_x1 = max(x_min, 0); inter_y1 = max(y_min, 0)
        inter_x2 = min(x_max, target_w); inter_y2 = min(y_max, target_h)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return None
        x_min, y_min, x_max, y_max = inter_x1, inter_y1, inter_x2, inter_y2

        area = (x_max - x_min) * (y_max - y_min)
        if area / (target_w * target_h) < self.bbox_area_threshold:
            return None
        return x_min, y_min, x_max, y_max

    def sample_layout_keep_frames(self, num_video_frames):
        """Which frames keep their layout. ``layout_keep_frames`` (an explicit 1-indexed list) wins
        when set; otherwise ``layout_drop_rate==0`` keeps all frames (dense) and higher rates
        sparsify by keeping ~one frame per local window, always keeping the last frame."""
        if num_video_frames <= 0:
            return set()
        if self.layout_keep_frames is not None:
            # 1-indexed positions inside the sampled clip -> 0-indexed, clipped to the clip length.
            return {min(max(int(p), 1), num_video_frames) - 1 for p in self.layout_keep_frames}
        if self.layout_drop_rate <= 0.0:
            return set(range(num_video_frames))
        if self.layout_drop_rate >= 1.0:
            return set([num_video_frames - 1])
        keep_rate = max(1.0 - float(self.layout_drop_rate), 1e-6)
        local_window = max(1, int(round(1.0 / keep_rate)))
        local_window = min(local_window, num_video_frames)
        keep = set()
        for start in range(0, num_video_frames, local_window):
            end = min(start + local_window, num_video_frames)
            keep.add(random.randint(start, end - 1))
        keep.add(num_video_frames - 1)  # always keep the last frame's layout
        return keep

    def _build_layout(self, data_info, annotation_path, batch_index):
        """Build the colored bbox canvas ``[f, 3, H, W]`` in [-1, 1] plus per-object phrases/colors."""
        track_file_rel = data_info.get("sam3_track_file")
        if not track_file_rel:
            raise ValueError(f"no sam3_track_file for {data_info.get('UID')}")
        with open(os.path.join(annotation_path, track_file_rel), "r") as f:
            track_data = json.load(f)
        tw, th = data_info["sam3_track_resolution"].lower().split("x")
        track_width, track_height = int(tw), int(th)

        num_frames = len(batch_index)
        keep = self.sample_layout_keep_frames(num_frames)

        bboxes_all, phrases_all, ids_all = [], [], []
        for track_key, track in track_data.get("tracks", {}).items():
            frame_to_bbox = track.get("bboxes", {})
            per_frame, has_bbox = [], False
            for i in range(num_frames):
                if i not in keep:
                    bbox = None
                else:
                    bbox = frame_to_bbox.get(str(int(batch_index[i])), None)
                    if bbox is not None:
                        bbox = self.resize_bbox(
                            bbox, orig_size=(track_height, track_width), target_size=(self.height, self.width)
                        )
                if bbox is not None:
                    has_bbox = True
                per_frame.append(bbox)
            if has_bbox:
                bboxes_all.append(per_frame)
                phrases_all.append(track.get("caption", ""))
                ids_all.append(str(track.get("id", track_key)))

        if len(bboxes_all) == 0:
            raise ValueError(f"no valid bboxes for {data_info.get('UID')}")
        if len(bboxes_all) > self.max_objs:
            sel = random.sample(range(len(bboxes_all)), self.max_objs)
            bboxes_all = [bboxes_all[i] for i in sel]
            phrases_all = [phrases_all[i] for i in sel]
            ids_all = [ids_all[i] for i in sel]
        if len(bboxes_all) > len(COLOR_LIST):
            raise ValueError(f"more objects ({len(bboxes_all)}) than colors ({len(COLOR_LIST)})")
        color_names = random.sample(list(COLOR_LIST.keys()), len(bboxes_all))

        # Draw the boxes per frame; large boxes first so small ones stay visible on top.
        canvas_frames = []
        for i in range(num_frames):
            img = Image.fromarray(np.zeros((self.height, self.width, 3), dtype=np.uint8))
            draw = ImageDraw.Draw(img)
            items = [(obj[i], c) for obj, c in zip(bboxes_all, color_names) if obj[i] is not None]
            items.sort(key=lambda it: max(0.0, it[0][2] - it[0][0]) * max(0.0, it[0][3] - it[0][1]), reverse=True)
            for bbox, color in items:
                draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=COLOR_LIST[color], width=4)
            canvas_frames.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))  # [3, H, W]
        bbox_canvas = torch.stack(canvas_frames, dim=0).float() / 255.0 * 2.0 - 1.0  # [f, 3, H, W] in [-1, 1]
        # bboxes_all is returned too (not just the rendered canvas) so validation can draw the
        # boxes onto the GENERATED frames -- the canvas alone cannot be overlaid without keying
        # out its black background. Shape: [n_obj][n_frame] of [x0, y0, x1, y1] or None.
        return bbox_canvas, phrases_all, color_names, bboxes_all, ids_all

    def get_batch(self, idx):
        data_info = self.dataset[idx % self.length]
        video_path = self._resolve(self._remap(data_info["Video_Path"]))
        annotation_path = self._resolve(self._remap(data_info["Annotation_Path"]))
        caption = (data_info.get("caption") or "").strip()

        ow, oh = data_info["resolution"].lower().split("x")
        orig_w, orig_h = int(ow), int(oh)

        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            n = len(video_reader)
            if n == 0:
                raise ValueError(f"Empty video: {video_path}")
            n_sample = self.video_sample_n_frames
            clip_length = min(n, (n_sample - 1) * self.video_sample_stride + 1)
            assert clip_length == 81
            start_idx = 0
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, n_sample, dtype=int)
            frames = func_timeout(
                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(video_reader, batch_index)
            )
        pixel_values = self._cover_crop(np.array(frames))  # [f, 3, H, W]

        # camera trajectory (crop-aware intrinsics matching the frame cover+crop)
        cam_file = os.path.join(annotation_path, data_info["CameraFile"])
        coordinates = read_camera_npz(
            cam_file, orig_size=(orig_h, orig_w), target_size=(self.height, self.width),
            normalized_intrinsics=self.normalized_intrinsics,
        )
        n_cam = len(coordinates)
        coordinates = [coordinates[min(int(i), n_cam - 1)] for i in batch_index]
        plucker = process_pose_params(coordinates, width=self.width, height=self.height, apply_aspect_fit=False)
        control_camera_values = plucker.permute(0, 3, 1, 2).contiguous()  # [f, 6, H, W]

        # layout (colored bbox canvas + per-object phrases/colors)
        bbox_canvas, bbox_phrases, bbox_color_names, bboxes, bbox_track_ids = self._build_layout(data_info, annotation_path, batch_index)

        return pixel_values, control_camera_values, bbox_canvas, bbox_phrases, bbox_color_names, bboxes, bbox_track_ids, caption

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, control_camera_values, bbox_canvas, bbox_phrases, bbox_color_names, bboxes, bbox_track_ids, text = self.get_batch(idx)
                return {
                    "pixel_values": pixel_values,
                    "control_camera_values": control_camera_values,
                    "bbox_canvas": bbox_canvas,
                    "bbox_phrases": bbox_phrases,
                    "bbox_color_names": bbox_color_names,
                    # ragged list -- camera_collate_fn picks keys explicitly, so it is ignored
                    # during training and only read by log_validation (which indexes the dataset).
                    "bboxes": bboxes,
                    "bbox_track_ids": bbox_track_ids,   # str ids, aligned with bboxes/phrases/colors
                    "text": text,
                    "data_type": "video",
                    "idx": idx,
                }
            except Exception as e:
                print(f"Error loading data {self.dataset[idx % self.length].get('UID', idx)}: {e}")
                idx = random.randint(0, self.length - 1)