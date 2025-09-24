from __future__ import annotations
from typing import Optional
import numpy as np




class YOLOv8Detector:
    """Thin wrapper over ultralytics.YOLO returning person boxes as (N,4)."""
    def __init__(self,
    model: str = "yolov8s",
    weights: Optional[str] = None,
    device: str = "cuda",
    conf: float = 0.50,
    iou: float = 0.60,
    imgsz: int = 960,
    half: bool = False):
        from importlib import import_module
        self.ul = import_module("ultralytics")
        YOLO = self.ul.YOLO
        self.yolo = YOLO(weights or model)
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.half = bool(half)


    def detect(self, frame_bgr: "np.ndarray") -> "np.ndarray":
            """Run detection on a single frame and return person boxes (N,4)."""
            res = self.yolo.predict(source=frame_bgr[..., ::-1], imgsz=self.imgsz,
                                conf=self.conf, iou=self.iou, device=self.device,
                                verbose=False)
            boxes = []
            for r in res:
                if getattr(r, "boxes", None) is None:
                    continue
                xb = r.boxes.xyxy.cpu().numpy()
                xc = r.boxes.cls.cpu().numpy() if getattr(r.boxes, "cls", None) is not None else None
                if xc is None:
                    boxes.extend([b[:4] for b in xb])
                else:
                    for b, c in zip(xb, xc):
                        if int(c) == 0: # person class
                            boxes.append(b[:4])
            if not boxes:
                return np.empty((0, 4), dtype=np.float32)
            return np.asarray(boxes, dtype=np.float32)

    def __call__(self, frame_bgr: "np.ndarray") -> "np.ndarray":
        """Make the detector callable for pipeline compatibility."""
        return self.detect(frame_bgr)