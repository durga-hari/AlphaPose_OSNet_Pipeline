# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path
import cv2
import io
import codecs

def ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def safe_stem(p: Path) -> str:
    # filename (without extension) sanitized for filesystem use
    s = Path(p).stem
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)

def video_writer(out_path: Path, fps: float, size_hw: tuple[int, int]):
    """
    Create an MP4 writer with a sane default codec. Tries 'avc1' then 'mp4v'.
    size_hw: (width, height)
    """
    ensure_dir(out_path.parent)
    w, h = size_hw
    # Try avc1 (H.264). If not available, fall back to mp4v (MPEG-4).
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if vw.isOpened():
            return vw
    # Last resort: XVID
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    return vw

def jsonl_writer(out_path: Path):
    """
    Open a UTF-8 JSONL writer. Returns a simple file-like object with .write(string).
    """
    ensure_dir(out_path.parent)
    # Use buffering; ensure UTF-8; avoid Windows newline translation
    return codecs.open(str(out_path), mode="w", encoding="utf-8")
