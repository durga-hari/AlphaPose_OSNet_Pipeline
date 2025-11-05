# from __future__ import annotations
# from pathlib import Path
# from typing import List, Optional
# import numpy as np
# import cv2
# import torch


# class OSNetExtractor:
#     def __init__(self, weights: str | None = None, device: str = "cuda",
#                  model_name: str = "osnet_x1_0"):
#         self.device = device
#         self.ok = False
#         self._ext = None
#         if weights:
#             self._init(Path(weights), model_name)

#     def _init(self, w: Path, model_name: str):
#         try:
#             from torchreid.utils import FeatureExtractor  # type: ignore
#         except Exception as e:
#             print(f"[OSNet] torchreid not available: {e}")
#             return

#         try:
#             self._ext = FeatureExtractor(
#                 model_name=model_name,
#                 model_path=str(w),
#                 device=self.device,
#                 verbose=True,
#             )
#             # Force load to ensure weights applied
#             if not self._ext.model:
#                 ckpt = torch.load(str(w), map_location=self.device)
#                 self._ext.model.load_state_dict(ckpt, strict=False)

#             self.ok = True
#             print(f"[OSNet] Loaded weights from {w}")
#         except Exception as e:
#             print(f"[OSNet] Initialization failed: {e}")
#             self._ext = None
#             self.ok = False

#     def is_ready(self) -> bool:
#         return self.ok and self._ext is not None

#     def _crop_rgb(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> List[np.ndarray]:
#         H, W = frame_bgr.shape[:2]
#         rgb_list = []
#         for b in boxes_xyxy:
#             x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
#             x1 = max(0, min(W - 1, x1))
#             x2 = max(0, min(W - 1, x2))
#             y1 = max(0, min(H - 1, y1))
#             y2 = max(0, min(H - 1, y2))
#             if x2 <= x1 or y2 <= y1:
#                 rgb_list.append(np.zeros((16, 8, 3), dtype=np.uint8))
#                 continue
#             crop = frame_bgr[y1:y2, x1:x2]
#             rgb_list.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#         return rgb_list

#     def __call__(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> list[Optional[np.ndarray]]:
#         if boxes_xyxy is None or boxes_xyxy.size == 0:
#             return []
#         if not self.is_ready():
#             return [None] * len(boxes_xyxy)

#         rgb_list = self._crop_rgb(frame_bgr, boxes_xyxy)
#         feats = self._ext(rgb_list)
#         out: list[Optional[np.ndarray]] = []

#         for f in feats:
#             if f is None:
#                 out.append(None)
#                 continue
#             if isinstance(f, torch.Tensor):
#                 f = f.detach().cpu().numpy()        # <-- convert from CUDA to NumPy safely
#             v = np.asarray(f, dtype=np.float32).reshape(-1)
#             n = float(np.linalg.norm(v) + 1e-12)
#             out.append(v / n)
#         return out


#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
import torch


class OSNetExtractor:
    """
    OSNet Re-ID feature extractor using Torchreid backbone (true feature head).
    Loads full MSMT17 weights and discards classifier safely.
    """

    def __init__(self,
                 weights: str | None = None,
                 device: str = "cuda",
                 model_name: str = "osnet_x1_0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.ok = False
        if weights:
            self._init(Path(weights), model_name)

    # ------------------------------------------------------------------
    def _init(self, w: Path, model_name: str):
        """Build OSNet model, load weights, strip classifier."""
        try:
            from torchreid import models
        except Exception as e:
            print(f"[OSNet] Torchreid not available: {e}")
            return

        try:
            # Match checkpoint’s classifier shape (1041 for MSMT17)
            model = models.build_model(
                name=model_name,
                num_classes=1041,
                pretrained=False
            )
            ckpt = torch.load(str(w), map_location=self.device)

            # Load everything except classifier.* parameters
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"[OSNet] Ignored keys → missing: {missing}, unexpected: {unexpected}")

            # Remove classifier head; we only use features
            if hasattr(model, "classifier"):
                model.classifier = torch.nn.Identity()

            model.eval().to(self.device)
            self.model = model
            self.ok = True
            print(f"[OSNet] Loaded feature model from {w}")
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
    def _preprocess(self, imgs: List[np.ndarray]) -> torch.Tensor:
        """Normalize and convert RGB numpy images to Torch tensors."""
        tensor_list = []
        for img in imgs:
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32)

            tensor_list.append(img)
        return torch.cat(tensor_list, dim=0).to(self.device)

    # ------------------------------------------------------------------
    def __call__(self,
                 frame_bgr: np.ndarray,
                 boxes_xyxy: np.ndarray) -> list[Optional[np.ndarray]]:
        """Return list of (512,)-dim L2-normalized embeddings."""
        if boxes_xyxy is None or boxes_xyxy.size == 0:
            return []
        if not self.is_ready():
            return [None] * len(boxes_xyxy)

        rgb_list = self._crop_rgb(frame_bgr, boxes_xyxy)
        x = self._preprocess(rgb_list)

        with torch.no_grad():
            # Forward backbone → global pooling → fc head → normalize
            x = x.to(self.device, dtype=torch.float32)
            fmaps = self.model.featuremaps(x)
            global_feat = self.model.global_avgpool(fmaps)
            global_feat = global_feat.view(global_feat.size(0), -1)
            feat = self.model.fc(global_feat)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)

        feats_np = feat.detach().cpu().numpy().astype(np.float32)
        return [feats_np[i] for i in range(feats_np.shape[0])]


# ------------------------------------------------------------------
if __name__ == "__main__":
    import cv2
    img = np.ones((128, 64, 3), np.uint8) * 127
    model_path = "/home/arun_remote/DaRA_Thesis/AlphaPose_OSNet_Pipeline/Pipeline/pretrained/osnet_x1_0_msmt17.pth"
    ext = OSNetExtractor(model_path)
    feats = ext(img, np.array([[0, 0, 64, 128]]))
    if feats and feats[0] is not None:
        v = feats[0]
        print("mean:", v.mean(), "std:", v.std(),
              "min:", v.min(), "max:", v.max(), "norm:", np.linalg.norm(v))



