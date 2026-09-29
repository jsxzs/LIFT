"""MoGe-2 runner for user mode: a single image -> world-space point cloud +
pixel intrinsics.

Loaded resident in serve.py (user mode) alongside the Wan pipeline. Mirrors the
lifting logic in estimate_pcd.py, but for an in-memory image with an identity
camera: the uploaded image defines frame 0, so world == camera coordinates and
no extrinsic transform is needed.
"""
import numpy as np

DEFAULT_MOGE_ID = "Ruicheng/moge-2-vitl"


class MoGeRunner:
    def __init__(self, model_id: str = DEFAULT_MOGE_ID, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self, log=print):
        if self.model is not None:
            return
        import torch  # noqa: F401
        from moge.model.v2 import MoGeModel
        log(f"[moge] loading {self.model_id} on {self.device} ...")
        self.model = MoGeModel.from_pretrained(self.model_id).to(self.device).eval()
        log("[moge] ready")

    def infer_image(self, img_rgb: np.ndarray, max_points: int = 120_000,
                    fov_x_deg: float = None):
        """img_rgb: (H, W, 3) uint8 RGB.

        Returns (points_world f32 (M,3), colors u8 (M,3), intr_px f64 (3,3)).
        Camera is identity, so points_world are just MoGe's camera-space points.
        """
        import torch
        if self.model is None:
            raise RuntimeError("MoGe not loaded; call load() first")
        H, W = img_rgb.shape[:2]
        img_t = (torch.from_numpy(np.ascontiguousarray(img_rgb)).float()
                 .permute(2, 0, 1) / 255.0).to(self.device)
        with torch.no_grad():
            kwargs = {}
            if fov_x_deg is not None:
                kwargs["fov_x"] = float(fov_x_deg)
            out = self.model.infer(img_t, **kwargs)
        points_cam = out["points"].detach().cpu().numpy()          # (H, W, 3)
        mask = out["mask"].detach().cpu().numpy().astype(bool)     # (H, W)
        intr_norm = out["intrinsics"].detach().cpu().numpy()       # (3, 3) normalized

        pts = points_cam[mask]           # world == camera (identity extrinsic)
        rgb = img_rgb[mask]
        if pts.shape[0] > max_points:
            sel = np.random.default_rng(0).choice(pts.shape[0], max_points,
                                                   replace=False)
            pts = pts[sel]
            rgb = rgb[sel]

        # MoGe returns intrinsics normalized by image size -> convert to pixels.
        K = intr_norm.astype(np.float64).copy()
        K[0, 0] *= W
        K[0, 2] *= W
        K[1, 1] *= H
        K[1, 2] *= H
        return pts.astype(np.float32), rgb.astype(np.uint8), K
