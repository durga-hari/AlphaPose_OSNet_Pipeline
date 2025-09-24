from __future__ import annotations
from typing import Any
from .yolov8_detector import YOLOv8Detector
from common import Registry


DETECTORS = Registry()
DETECTORS.register("yolov8")(YOLOv8Detector)




def build_detector(name: str, **kwargs) -> Any:
    return DETECTORS.build(name, **kwargs)