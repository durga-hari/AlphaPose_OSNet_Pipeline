#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import logging
import numpy as np

log = logging.getLogger(__name__)

class YOLOv8Detector:
    def __init__(self,
                 model: str = "yolov8s",
                 weights: Optional[Path] = None,
                 device: str = "cuda",
                 conf: float = 0.25,
                 iou: float = 0.45):
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = "cuda" if str(device).lower() == "cuda" else "cpu"
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise ImportError(
                "Ultralytics is not installed. Install with: pip install ultralytics"
            ) from e

        w = str(weights) if weights else model
        self._mdl = YOLO(w)
        self._device_override = self.device
        log.info("YOLOv8Detector ready: %s (device=%s, conf=%.2f, iou=%.2f)", w, self.device, self.conf, self.iou)

    def detect(self, frame_bgr) -> List[List[float]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        try:
            res = self._mdl.predict(source=frame_bgr, verbose=False,
                                    conf=self.conf, iou=self.iou, device=self._device_override)
        except TypeError:
            res = self._mdl.predict(source=frame_bgr, verbose=False, conf=self.conf, iou=self.iou)

        out = []
        if not res: return out
        r0 = res[0]
        if getattr(r0, "boxes", None) is None: return out
        try:
            xyxy = r0.boxes.xyxy.cpu().numpy()
            conf = r0.boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, conf):
                x1, y1, x2, y2 = [float(v) for v in b]
                out.append([x1, y1, x2, y2, float(c)])
        except Exception:
            for b in r0.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                score = float(b.conf[0])
                out.append([x1, y1, x2, y2, score])
        return out
