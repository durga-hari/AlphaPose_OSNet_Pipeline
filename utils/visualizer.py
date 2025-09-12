# # -*- coding: utf-8 -*-
# from __future__ import annotations
# from typing import Optional, Tuple, List
# import cv2
# import numpy as np

# # COCO-17 keypoints (HRNet COCO order)
# COCO_KEYPOINT_NAMES = [
#     "nose", "left_eye", "right_eye", "left_ear", "right_ear",
#     "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
#     "left_wrist", "right_wrist", "left_hip", "right_hip",
#     "left_knee", "right_knee", "left_ankle", "right_ankle",
# ]

# COCO_LINKS = [
#     (5, 6), (5, 7), (7, 9), (6, 8), (8,10),       # arms
#     (11,12), (11,13), (13,15), (12,14), (14,16),  # legs
#     (5,11), (6,12),                               # torso
#     (0,1), (0,2), (1,3), (2,4)                    # head
# ]

# # lower threshold so limbs render (scores now ~[0,1] after sigmoid)
# DRAW_THR = 0.01

# def _color(i: int) -> Tuple[int,int,int]:
#     rnd = (37 * (i + 1)) % 255
#     return (int((rnd*3) % 255), int((rnd*7) % 255), int((rnd*11) % 255))

# def draw_bbox_and_id(img, bbox, pid: Optional[int] = None, color=None):
#     x1, y1, x2, y2 = map(int, bbox)
#     color = color or (0, 255, 0)
#     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#     if pid is not None:
#         label = f"ID {pid}"
#         (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#         cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
#         cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

# def draw_skeleton_coco(img, kpts, kpt_thresh=0.2, radius=2, thickness=2, color=(0,255,0)):
    
#     kpts = np.asarray(kpts, dtype=np.float32)
#     for (x, y, c) in kpts:
#         if c >= kpt_thresh and 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
#             cv2.circle(img, (int(x), int(y)), radius, color, -1)

#     for i,j in COCO_LINKS:
#         xi, yi, ci = kpts[i]
#         xj, yj, cj = kpts[j]
#         if ci>=kpt_thresh and cj>=kpt_thresh:
#             cv2.line(img, (int(xi),int(yi)), (int(xj),int(yj)), color, thickness)
# -*- coding: utf-8 -*-
"""
vis_utils.py — robust bbox + skeleton drawing for COCO/HALPE outputs.
Handles:
- keypoints in shapes: (17|26|136,3), (N,17|26|136,3), flat length 51/78/408 lists
- dicts with keys: 'keypoints', 'kpts', 'kps', 'pose', 'landmarks'
- auto-rescale from model/input size back to original frame size
- configurable thresholds
"""

from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2


MARGIN = 200

# -------------------------
# Skeleton definitions
# -------------------------
COCO17_PAIRS: List[Tuple[int, int]] = [
    (0,1),(1,2),(2,3),(3,4),        # nose -> eyes -> ears (approx)
    (1,5),(5,7),(7,9),              # left arm
    (1,6),(6,8),(8,10),             # right arm
    (5,6),(5,11),(6,12),            # shoulders -> hips
    (11,12),(11,13),(13,15),        # left leg
    (12,14),(14,16)                 # right leg
]
# This COCO chain expects order: 0:nose,1:left_eye,2:right_eye,3:left_ear,4:right_ear,
# 5:left_shoulder,6:right_shoulder,7:left_elbow,8:right_elbow,9:left_wrist,10:right_wrist,
# 11:left_hip,12:right_hip,13:left_knee,14:right_knee,15:left_ankle,16:right_ankle.

# HALPE has 26/136 layouts. We’ll just connect neighbor indices to make it visible
# if you pass HALPE; for pretty lines you can customize this table later.
def _auto_pairs(n_kpts: int) -> List[Tuple[int, int]]:
    if n_kpts == 17:
        return COCO17_PAIRS
    pairs = []
    for i in range(n_kpts - 1):
        pairs.append((i, i + 1))
    return pairs

# -------------------------
# Public thresholds (tweak here or override per call)
# -------------------------
DEFAULT_KPT_CONF = 0.0 #0.1          # per-joint min confidence
DEFAULT_POSE_CONF = 0.0         # overall pose score gate (if provided)
DEFAULT_BBOX_CONF = 0.0         # bbox score gate (if provided)

# -------------------------
# Utilities
# -------------------------
def _first_present(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None

def _to_numpy(a: Any) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    return np.array(a, dtype=np.float32)

def _guess_layout(k: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Return (kpts[N, J, 3], J) no matter the incoming layout.
    Accepts flat arrays length 3*J or already [J,3] or [N,J,3].
    """
    arr = _to_numpy(k)
    if arr.ndim == 1:
        # flat vector length 3*J
        if arr.size % 3 != 0:
            raise ValueError(f"Flat keypoints length {arr.size} not divisible by 3")
        J = arr.size // 3
        return arr.reshape(1, J, 3), J
    elif arr.ndim == 2:
        # [J,3]
        if arr.shape[1] != 3:
            raise ValueError(f"Expected shape [J,3], got {arr.shape}")
        return arr[np.newaxis, ...], arr.shape[0]
    elif arr.ndim == 3:
        # [N,J,3]
        if arr.shape[2] != 3:
            raise ValueError(f"Expected shape [N,J,3], got {arr.shape}")
        return arr, arr.shape[1]
    else:
        raise ValueError(f"Unsupported keypoints ndim={arr.ndim}")

def _rescale_points(kpts: np.ndarray,
                    src_wh: Tuple[int, int],
                    dst_wh: Tuple[int, int],
                    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None,
                    was_cropped: bool = False) -> np.ndarray:
    """
    Rescale keypoints from model/input space to original frame size.

    If the pose model ran on the full frame resized to (src_wh) with possible letterbox,
    you must undo letterbox. If it ran on per-person crops (bbox), pass bbox and set was_cropped=True.
    """
    src_w, src_h = src_wh
    dst_w, dst_h = dst_wh

    out = kpts.copy()  # [N,J,3], x,y in src space
    if was_cropped and bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        bw = max(x2 - x1, 1e-3)
        bh = max(y2 - y1, 1e-3)
        # Here we assume src_wh was the crop size (pose input) mapped from bbox.
        # So convert from normalized crop pixels back to original coord:
        sx = bw / float(src_w)
        sy = bh / float(src_h)
        out[..., 0] = out[..., 0] * sx + x1
        out[..., 1] = out[..., 1] * sy + y1
    else:
        # Simple scale from src to dst (no letterbox handling here; add if needed)
        sx = float(dst_w) / float(src_w)
        sy = float(dst_h) / float(src_h)
        out[..., 0] = out[..., 0] * sx
        out[..., 1] = out[..., 1] * sy

    return out

# -------------------------
# Drawers
# -------------------------
def draw_bbox_and_id(
    frame_bgr: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    track_id: Optional[Union[int, str]] = None,
    score: Optional[float] = None,
    color: Optional[Tuple[int, int, int]] = None,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    if color is None:
        color = (0, 255, 0)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    label = []
    if track_id is not None:
        label.append(f"ID {track_id}")
    if score is not None:
        label.append(f"{score:.2f}")
    if label:
        txt = "  ".join(label)
        (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame_bgr, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

def draw_skeletons(
    frame_bgr: np.ndarray,
    keypoints_any: Any,
    *,
    frame_wh: Optional[Tuple[int, int]] = None,
    input_wh: Optional[Tuple[int, int]] = None,
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None,
    dataset: str = "coco",              # "coco" | "halpe" | "auto"
    kpt_thresh: float = DEFAULT_KPT_CONF,
    pose_score: Optional[float] = None,
    pose_score_thresh: float = DEFAULT_POSE_CONF,
    was_cropped: bool = False,
    color_pts: Tuple[int, int, int] = (0, 255, 255),
    color_lines: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
    thickness: int = 2,
) -> Dict[str, Any]:
    """
    Draws skeleton(s) on the frame. Returns diagnostics.
    - keypoints_any can be one pose [J,3] or many [N,J,3] or flat 3*J.
    - frame_wh required if rescaling (pass frame.shape[1], frame.shape[0]).
    - input_wh required if rescaling from the model/input size.
    """
    diag = {"n_poses": 0, "n_drawn": 0, "J": 0}

    kpts, J = _guess_layout(keypoints_any)  # [N,J,3]
    diag["n_poses"], diag["J"] = kpts.shape[0], J

    if pose_score is not None and pose_score < pose_score_thresh:
        return diag  # filtered at pose level

    if dataset == "auto":
        pairs = _auto_pairs(J)
    else:
        pairs = COCO17_PAIRS if dataset.lower().startswith("coco") and J == 17 else _auto_pairs(J)

    # Optional rescale
    if frame_wh is not None and input_wh is not None:
        kpts = _rescale_points(kpts, src_wh=input_wh, dst_wh=frame_wh,
                               bbox_xyxy=bbox_xyxy, was_cropped=was_cropped)

    H, W = frame_bgr.shape[:2]

    for n in range(kpts.shape[0]):
        pts = kpts[n]
        conf = pts[:, 2]
        good = conf >= kpt_thresh
        if not np.any(good):
            continue

        # lines
        for i, j in pairs:
            if i < J and j < J and good[i] and good[j]:
                xi, yi = int(round(pts[i,0])), int(round(pts[i,1]))
                xj, yj = int(round(pts[j,0])), int(round(pts[j,1]))
                if (-MARGIN <= xi <= W + MARGIN and -MARGIN <= yi <= H + MARGIN and
                        -MARGIN <= xj <= W + MARGIN and -MARGIN <= yj <= H + MARGIN):
                    cv2.line(frame_bgr, (xi, yi), (xj, yj), color_lines, thickness)

        # points
        for idx in range(J):
            if good[idx]:
                x, y = int(round(pts[idx,0])), int(round(pts[idx,1]))
                if -MARGIN <= x <= W + MARGIN and -MARGIN <= y <= H + MARGIN:
                    cv2.circle(frame_bgr, (x, y), radius, color_pts, -1)

        diag["n_drawn"] += 1

    return diag
