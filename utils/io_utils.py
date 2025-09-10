# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Tuple, IO
import cv2
import re

def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def video_writer(out_path: Path, fps: float, frame_size: Tuple[int, int]):
    ensure_dir(out_path.parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, float(fps), frame_size, True)

def jsonl_writer(out_path: Path) -> IO[str]:
    ensure_dir(out_path.parent)
    return open(out_path, "w", encoding="utf-8", newline="\n")

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
def safe_stem(p: Path) -> str:
    stem = p.stem
    stem = _SAFE.sub("_", stem)
    return stem[:128]
