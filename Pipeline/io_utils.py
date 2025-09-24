from __future__ import annotations
from typing import Iterator, Tuple, Optional
from pathlib import Path
import cv2
import json
import numpy as np



class VideoReader:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    def __iter__(self) -> Iterator[Tuple[int, "np.ndarray"]]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print(f"[VideoReader] Failed to read frame {idx}")
                break
        idx += 1
        yield idx, frame


    def close(self):
        self.cap.release()


class VideoWriter:
    def __init__(self, path: Path | str, fps: float, size_hw: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        W, H = size_hw
        self.vw = cv2.VideoWriter(str(path), fourcc, float(fps), (W, H))
        if not self.vw.isOpened():
            raise RuntimeError(f"Could not open VideoWriter at {path}")

    def write(self, frame: "np.ndarray"):
        self.vw.write(frame)

    def close(self):
        self.vw.release()




class JSONLWriter:
    def __init__(self, path: Path | str):
        self.f = open(path, "w", encoding="utf-8")


    def write(self, obj: dict):
        self.f.write(json.dumps(obj) + "\n")


    def close(self):
        try:
            self.f.close()
        except Exception:
            pass