from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
import torch


class OSNetExtractor:
    def __init__(self, weights: str | None = None, device: str = "cuda",
                 model_name: str = "osnet_x1_0"):
        self.device = device
        self._ext = None
        self.ok = False
        if weights:
            self._init(Path(weights), model_name)

    def _init(self, w: Path, model_name: str):
        try:
            from torchreid.utils import FeatureExtractor  # type: ignore
        except Exception as e:
            print(f"[OSNet] torchreid missing: {e}. Embeddings disabled.")
            return

        try:
            # Force weights_only=False to allow loading full PyTorch checkpoint in PyTorch 2.6+
            # Note: Only do this for trusted official weights
            self._ext = FeatureExtractor(model_name=model_name,
                                         model_path=str(w),
                                         device=self.device)
            # Hack: load the checkpoint manually if needed
            import torch
            state = torch.load(str(w), map_location=self.device, weights_only=False)
            # Normally FeatureExtractor handles this internally, so this is optional
            self.ok = True
        except Exception as e:
            print(f"[OSNet] init failed: {e}")
            self._ext = None
            self.ok = False

    def is_ready(self) -> bool:
        return self.ok and self._ext is not None

    def _crop_rgb(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> List[np.ndarray]:
        H, W = frame_bgr.shape[:2]
        rgb_list = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                rgb_list.append(np.zeros((16, 8, 3), dtype=np.uint8)); continue
            crop = frame_bgr[y1:y2, x1:x2]
            rgb_list.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return rgb_list

    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> list[Optional[np.ndarray]]:
        if boxes_xyxy is None or boxes_xyxy.size == 0:
            return []
        if not self.is_ready():
            return [None] * len(boxes_xyxy)

        rgb_list = self._crop_rgb(frame_bgr, boxes_xyxy)
        feats = self._ext(rgb_list)
        out: list[Optional[np.ndarray]] = []

        for f in feats:
            if f is None:
                out.append(None)
                continue
            try:
                v = np.asarray(f, dtype=np.float32).reshape(-1)
                n = float(np.linalg.norm(v) + 1e-12)
                out.append(v / n)
            except Exception:
                out.append(None)
        return out
