#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, List

import numpy as np

from common import Registry
from .resnet_extractor import ResNetExtractor
from .osnet_extractor import OSNetExtractor

REID = Registry()


class _DummyReID:
    """No-op ReID used when type is 'none'."""

    def is_ready(self) -> bool:
        return False

    def __call__(self, frame_bgr, boxes_xyxy) -> List[np.ndarray]:
        return []


class ResNetReID:
    """
    Wrapper around ResNetExtractor for the SequentialPipeline.

    API expected by pipeline:
      - .is_ready() -> bool
      - __call__(frame_bgr, boxes_xyxy) -> list[np.ndarray] (each L2-normalized)
    """

    def __init__(self, backbone: str = "resnet50",
                 device: str = "cuda",
                 weights: str | None = None):
        self.extractor = ResNetExtractor(
            backbone=backbone,
            device=device,
            weights=weights,
        )
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def __call__(self, frame_bgr, boxes_xyxy) -> List[np.ndarray]:
        """
        frame_bgr: HxWx3 uint8
        boxes_xyxy: (N,4) [x1,y1,x2,y2]
        returns: list of L2-normalized embeddings (np.ndarray, shape (D,))
        """
        if boxes_xyxy is None:
            return []

        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        H, W = frame_bgr.shape[:2]

        crops = []
        for b in boxes:
            x1, y1, x2, y2 = b
            x1 = int(max(0, min(W - 1, np.floor(x1))))
            y1 = int(max(0, min(H - 1, np.floor(y1))))
            x2 = int(max(0, min(W,     np.ceil(x2))))
            y2 = int(max(0, min(H,     np.ceil(y2))))

            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((256, 128, 3), dtype=np.uint8))
                continue

            crops.append(frame_bgr[y1:y2, x1:x2].copy())

        feats = self.extractor(crops)  # torch.Tensor (N, D), already L2-normalized

        if hasattr(feats, "detach"):
            feats_np = feats.detach().cpu().numpy().astype(np.float32)
        else:
            feats_np = np.asarray(feats, dtype=np.float32)

        norms = np.linalg.norm(feats_np, axis=1, keepdims=True) + 1e-12
        feats_np = feats_np / norms

        return [feats_np[i] for i in range(feats_np.shape[0])]


# ---- Registry bindings ----

REID.register("resnet")(ResNetReID)
REID.register("resnet50")(ResNetReID)

# OSNet uses OSNetExtractor directly (already matches pipeline API)
REID.register("osnet")(OSNetExtractor)


def build_reid(name: str, **kwargs) -> Any:
    """
    Factory used by ap_pipeline.SequentialPipeline.
    name: e.g., "resnet", "resnet50", "osnet", or "none"
    kwargs: backbone, weights, device, etc.
    """
    if name is None:
        return _DummyReID()
    name = str(name).lower().strip()
    if name in {"none", "off", ""}:
        return _DummyReID()
    return REID.build(name, **kwargs)
