# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple, List
import cv2

# COCO-17 keypoints (order used by HRNet COCO configs)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Simple COCO links for drawing
COCO_LINKS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8,10),       # arms
    (11,12), (11,13), (13,15), (12,14), (14,16),  # legs
    (5,11), (6,12),                               # torso
    (0,1), (0,2), (1,3), (2,4)                    # head
]

def _color(i: int) -> Tuple[int,int,int]:
    rnd = (37 * (i + 1)) % 255
    return (int((rnd*3) % 255), int((rnd*7) % 255), int((rnd*11) % 255))

def draw_bbox_and_id(img, bbox, pid: Optional[int] = None, color=None):
    x1, y1, x2, y2 = map(int, bbox)
    color = color or (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if pid is not None:
        label = f"ID {pid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

def draw_skeleton_coco(img, kpts: List[Tuple[float,float,float]], pid: Optional[int] = None):
    # joints
    for (x, y, s) in kpts:
        if s > 0.05:
            cv2.circle(img, (int(x), int(y)), 3, _color(pid or 0), -1)
    # edges
    for a, b in COCO_LINKS:
        if a < len(kpts) and b < len(kpts):
            x1, y1, s1 = kpts[a]
            x2, y2, s2 = kpts[b]
            if s1 > 0.05 and s2 > 0.05:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), _color(pid or 0), 2)
