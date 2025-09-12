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
                 iou: float = 0.45,
                 imgsz: int = 640,
                 half: Optional[bool] = None):
        """
        imgsz: input size for YOLO (long side). 640 is a good default.
        half : if None -> auto (True when device=cuda), else use provided bool.
        """
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = "cuda" if str(device).lower() == "cuda" else "cpu"
        self._half = (self.device == "cuda") if half is None else bool(half)

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise ImportError(
                "Ultralytics is not installed. Install with: pip install ultralytics"
            ) from e

        w = str(weights) if weights else model
        self._mdl = YOLO(w)
        self._device_override = self.device
        log.info("YOLOv8Detector ready: %s (device=%s, conf=%.2f, iou=%.2f, imgsz=%d, half=%s)",
                 w, self.device, self.conf, self.iou, self.imgsz, self._half)

    def detect(self, frame_bgr) -> np.ndarray:
        """
        Returns np.ndarray (N,5): [x1,y1,x2,y2,score] for 'person' class only.
        Downstream _to_np_boxes will slice [:, :4] as needed.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.empty((0, 5), dtype=np.float32)

        try:
            res = self._mdl.predict(
                source=frame_bgr, verbose=False,
                conf=self.conf, iou=self.iou,
                device=self._device_override,
                classes=[0], imgsz=self.imgsz, half=self._half
            )
        except TypeError:
            res = self._mdl.predict(source=frame_bgr, verbose=False,
                                    conf=self.conf, iou=self.iou, classes=[0])

        if not res or getattr(res[0], "boxes", None) is None:
            return np.empty((0, 5), dtype=np.float32)

        out = []
        r0 = res[0]
        try:
            xyxy = r0.boxes.xyxy.cpu().numpy()
            conf = r0.boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, conf):
                x1, y1, x2, y2 = b.tolist()
                out.append([x1, y1, x2, y2, float(c)])
        except Exception:
            for b in r0.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                score = float(b.conf[0])
                out.append([x1, y1, x2, y2, score])

        return np.asarray(out, dtype=np.float32).reshape(-1, 5)

