#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
osnet_wrapper.py

Robust OSNet feature extractor.

- Prefers torchreid FeatureExtractor (model_name="osnet_x1_0") with a given .pth.
- If torchreid is missing or weight path is None/invalid, runs in "disabled" mode
  and returns [None] features (the pipeline still runs).
- Always converts features to CPU NumPy arrays and L2-normalizes them.

Usage:
  from .osnet_wrapper import OSNetExtractor
  osn = OSNetExtractor(weight=Path("Pipeline/pretrained/osnet_x1_0_msmt17.pth"), device="cuda")
  feats = osn.extract(frame_bgr, boxes)  # -> list of np.ndarray (D,) or None
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import logging
import numpy as np
import cv2

log = logging.getLogger(__name__)


class OSNetExtractor:
    def __init__(
        self,
        weight: Optional[Path],
        device: str = "cuda",
        require_torchreid: bool = False,
        model_name: str = "osnet_x1_0",
    ):
        """
        Parameters
        ----------
        weight : Path | None
            Path to OSNet checkpoint (.pth). If None or missing, extractor is disabled.
        device : str
            "cuda" or "cpu"
        require_torchreid : bool
            If True, raise ImportError when torchreid is missing. Otherwise log & disable.
        model_name : str
            torchreid model name. Default "osnet_x1_0".
        """
        self.device = "cuda" if str(device).lower() == "cuda" else "cpu"
        self._ext = None
        self._ok = False

        if weight is None:
            log.warning("OSNetExtractor: no weight provided; embeddings disabled.")
            return

        w = Path(weight).resolve()
        if not w.exists():
            log.warning("OSNetExtractor: weight not found at %s; embeddings disabled.", w)
            return

        # Try to import torchreid
        try:
            from torchreid.utils import FeatureExtractor  # type: ignore
        except Exception as e:
            msg = (
                "OSNetExtractor: 'torchreid' not installed. "
                "Install with: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git "
                f"(detail: {e})"
            )
            if require_torchreid:
                raise ImportError(msg)
            log.warning(msg)
            return

        # Initialize feature extractor
        try:
            self._ext = FeatureExtractor(
                model_name=model_name,
                model_path=str(w),
                device=self.device,
            )
            # Some torchreid versions print model stats on init; keep our own log too.
            log.info("OSNetExtractor loaded: %s (device=%s)", w, self.device)
            self._ok = True
        except Exception as e:
            log.warning("OSNetExtractor failed to initialize with %s: %s", w, e)
            self._ext = None
            self._ok = False

    def is_ready(self) -> bool:
        return bool(self._ok and self._ext is not None)

    def _crop_rgb_list(self, frame_bgr, boxes_xyxy: List[List[float]]) -> List[np.ndarray]:
        H, W = frame_bgr.shape[:2]
        crops: List[np.ndarray] = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                # use a tiny dummy crop to keep alignment (torchreid can handle arbitrary sizes)
                crops.append(np.zeros((16, 8, 3), dtype=np.uint8))
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return crops

    def extract(self, frame_bgr, boxes_xyxy: List[List[float]]) -> List[Optional[np.ndarray]]:
        """
        Returns a list of feature vectors (np.ndarray, L2-normalized) or None per box.
        """
        n = len(boxes_xyxy)
        if n == 0:
            return []
        if not self.is_ready():
            return [None] * n

        crops = self._crop_rgb_list(frame_bgr, boxes_xyxy)

        try:
            feats = self._ext(crops)  # may return a list/ndarray of tensors or numpy arrays
        except Exception as e:
            log.warning("OSNetExtractor: feature extraction failed (%s). Returning None features.", e)
            return [None] * n

        out: List[Optional[np.ndarray]] = []

        # We’ll support both torch.Tensor and np.ndarray results
        # without importing torch globally (avoid hard dep if disabled).
        torch = None  # lazy import only if needed
        for f in feats:
            if f is None:
                out.append(None)
                continue

            # Handle torch.Tensor
            if "torch" in type(f).__module__:
                if torch is None:
                    try:
                        import torch as _torch  # type: ignore
                        torch = _torch
                    except Exception:
                        torch = None
                if torch is not None and isinstance(f, torch.Tensor):
                    try:
                        v = f.detach().cpu().numpy().astype(np.float32).reshape(-1)
                    except Exception as e:
                        log.warning("OSNetExtractor: tensor -> numpy failed (%s).", e)
                        out.append(None)
                        continue
                else:
                    # Unexpected type
                    out.append(None)
                    continue
            else:
                # Assume it's already a numpy-like
                try:
                    v = np.asarray(f, dtype=np.float32).reshape(-1)
                except Exception as e:
                    log.warning("OSNetExtractor: array cast failed (%s).", e)
                    out.append(None)
                    continue

            # L2 normalize
            norm = float(np.linalg.norm(v) + 1e-12)
            v = v / norm
            out.append(v)

        # Keep length aligned with input boxes
        if len(out) != n:
            if len(out) < n:
                out.extend([None] * (n - len(out)))
            else:
                out = out[:n]
        return out
