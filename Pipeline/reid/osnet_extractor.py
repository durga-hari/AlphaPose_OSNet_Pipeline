#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

# ------------------------------------------------------------------
# PyTorch 2.6 safety for older checkpoints (optional but safe)
# ------------------------------------------------------------------
try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
except Exception:
    pass


class OSNetExtractor:
    """
    OSNet Re-ID feature extractor using Torchreid backbone.
    Loads MSMT17 OSNet checkpoint and discards classifier head.
    API matches the AlphaPose pipeline expectation:
      - .is_ready()
      - __call__(frame_bgr, boxes_xyxy) -> list[np.ndarray]
    """

    def __init__(self,
                 weights: str | None = None,
                 device: str = "cuda",
                 model_name: str = "osnet_x1_0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.ok = False
        if weights:
            self._init(Path(weights), model_name)

    # ------------------------------------------------------------------
    def _init(self, w: Path, model_name: str):
        """Build OSNet model, load weights (with state_dict unwrap), strip classifier."""
        try:
            from torchreid import models
        except Exception as e:
            print(f"[OSNet] Torchreid not available: {e}")
            return

        if not w.exists():
            print(f"[OSNet] Weights file does not exist: {w}")
            return

        try:
            # Match checkpoint’s classifier shape (1041 for MSMT17)
            model = models.build_model(
                name=model_name,
                num_classes=1041,
                pretrained=False
            )

            # Robust torch.load
            try:
                ckpt = torch.load(str(w), map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(str(w), map_location=self.device)
            except UnicodeDecodeError:
                print("[OSNet] UnicodeDecodeError; retrying with latin1 encoding…")
                try:
                    ckpt = torch.load(str(w), map_location=self.device,
                                      encoding="latin1", weights_only=False)
                except TypeError:
                    ckpt = torch.load(str(w), map_location=self.device,
                                      encoding="latin1")

            # Torchreid style: possibly wrapped as {"state_dict": ...}
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"[OSNet] Loaded OSNet checkpoint from {w}")
            if missing or unexpected:
                print(f"[OSNet] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

            # Remove classifier; we only care about feature embeddings
            if hasattr(model, "classifier"):
                model.classifier = torch.nn.Identity()

            model.eval().to(self.device)
            self.model = model
            self.ok = True
        except Exception as e:
            print(f"[OSNet] Initialization failed: {e}")
            self.ok = False
            self.model = None

    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return self.ok and self.model is not None

    # ------------------------------------------------------------------
    def _crop_rgb(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> List[np.ndarray]:
        """Crop and resize person boxes to 256×128 RGB images."""
        H, W = frame_bgr.shape[:2]
        rgb_list: List[np.ndarray] = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                rgb_list.append(np.zeros((256, 128, 3), dtype=np.uint8))
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (128, 256))
            rgb_list.append(crop)
        return rgb_list

    # ------------------------------------------------------------------
    def _preprocess(self, rgb_list: List[np.ndarray]) -> torch.Tensor:
        """Convert list of RGB crops to normalized torch.Tensor (N, 3, H, W)."""
        if not rgb_list:
            return torch.empty((0, 3, 256, 128), device=self.device)

        tensor_list = []
        for img in rgb_list:
            # HWC uint8 [0,255] → CHW float32 [0,1]
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            # Normalization (same as ImageNet)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t = (t - mean) / std
            tensor_list.append(t)

        return torch.stack(tensor_list, dim=0).to(self.device)

    # ------------------------------------------------------------------
    def __call__(self,
                 frame_bgr: np.ndarray,
                 boxes_xyxy: np.ndarray) -> list[Optional[np.ndarray]]:
        """
        Return list of (D,)-dim L2-normalized embeddings for each box.
        """
        if boxes_xyxy is None or getattr(boxes_xyxy, "size", 0) == 0:
            return []
        if not self.is_ready():
            # Keep alignment: return N None entries if model is not ready
            return [None] * len(boxes_xyxy)

        rgb_list = self._crop_rgb(frame_bgr, boxes_xyxy)
        x = self._preprocess(rgb_list)

        with torch.no_grad():
            x = x.to(self.device, dtype=torch.float32)
            # Torchreid OSNet forward: featuremaps → global pooling → fc → normalize
            fmaps = self.model.featuremaps(x)
            global_feat = self.model.global_avgpool(fmaps)
            global_feat = global_feat.view(global_feat.size(0), -1)
            feat = self.model.fc(global_feat)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)

        feats_np = feat.detach().cpu().numpy().astype(np.float32)
        return [feats_np[i] for i in range(feats_np.shape[0])]


# ------------------------------------------------------------------
if __name__ == "__main__":
    # Simple sanity check
    img = np.ones((128, 64, 3), np.uint8) * 127
    model_path = "/home/arun_remote/DaRA_Thesis/AlphaPose_OSNet_Pipeline/Pipeline/pretrained/osnet_x1_0_msmt17.pth"
    ext = OSNetExtractor(model_path)
    feats = ext(img, np.array([[0, 0, 64, 128]]))
    if feats and feats[0] is not None:
        v = feats[0]
        print("mean:", v.mean(), "std:", v.std(),
              "min:", v.min(), "max:", v.max(), "norm:", np.linalg.norm(v))
