# AlphaPose_OSNet_Pipeline_Optimized/dara_utils/visualizer.py
import cv2
import numpy as np
from typing import Iterable, Sequence, Tuple, Optional

# COCO 17-keypoints order used by most AlphaPose configs:
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

COCO_SKELETON: Sequence[Tuple[int, int]] = (
    # legs
    (15, 13), (13, 11),          # left ankle-knee-hip
    (16, 14), (14, 12),          # right ankle-knee-hip
    (11, 12),                    # pelvis

    # body & shoulders
    (5, 11), (6, 12), (5, 6),    # hips to shoulders & shoulders together

    # arms
    (5, 7), (7, 9),              # left shoulder-elbow-wrist
    (6, 8), (8, 10),             # right shoulder-elbow-wrist

    # head / face
    (5, 1), (6, 2), (1, 2),      # shoulders to eyes & between eyes
    (1, 3), (2, 4),              # eyes to ears
)

# pleasant distinct colors for left/right/center groups
COLOR_LEFT   = (0, 215, 255)   # BGR (amber)
COLOR_RIGHT  = (255, 140, 0)   # BGR (blue-ish)
COLOR_MID    = (0, 255, 0)     # green
COLOR_FACE   = (255, 0, 255)   # magenta
COLOR_BOX    = (0, 255, 0)     # bbox default

# map each limb to a color
def _edge_color(i1: int, i2: int) -> Tuple[int, int, int]:
    left  = {1,3,5,7,9,11,13,15}
    right = {2,4,6,8,10,12,14,16}
    face  = {0,1,2,3,4}
    if i1 in face and i2 in face: return COLOR_FACE
    if i1 in left or i2 in left:  # any left-side limb
        if not (i1 in right or i2 in right):
            return COLOR_LEFT
    if i1 in right or i2 in right:
        return COLOR_RIGHT
    return COLOR_MID

def _auto_thickness(img_shape: Tuple[int,int,int]) -> Tuple[int, int]:
    h, w = img_shape[:2]
    base = max(1, int(round((h + w) / 600)))  # scales nicely from 480p to 4k
    return base, max(1, base - 1)

def _to_xyc_array(kpts: Iterable) -> Optional[np.ndarray]:
    """
    Normalize input to Nx3 [x, y, conf]. Accepts Nx2, Nx3, or list of tuples.
    Returns None if shape invalid.
    """
    if kpts is None:
        return None
    arr = np.asarray(kpts)
    if arr.ndim != 2 or arr.shape[0] < 5:  # need at least head/shoulders to make sense
        return None
    if arr.shape[1] == 2:
        # no confidence -> set to 1.0
        ones = np.ones((arr.shape[0], 1), dtype=arr.dtype)
        arr = np.concatenate([arr.astype(np.float32), ones], axis=1)
    elif arr.shape[1] == 3:
        arr = arr.astype(np.float32)
    else:
        return None
    return arr

def draw_bbox_and_id(frame: np.ndarray,
                     bbox_xyxy: Sequence[float],
                     label: str,
                     color: Tuple[int,int,int] = COLOR_BOX) -> None:
    """
    Draw a bounding box [x1, y1, x2, y2] and a small id label.
    """
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    thick, _ = _auto_thickness(frame.shape)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, thick))
    # label background
    txt = label if isinstance(label, str) else str(label)
    (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad = 3
    bx2, by2 = x1 + tw + 2 * pad, y1 - th - 2 * pad
    by2 = max(by2, 0)
    cv2.rectangle(frame, (x1, by2), (bx2, y1), color, -1)
    cv2.putText(frame, txt, (x1 + pad, y1 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_skeleton(frame: np.ndarray,
                  keypoints: Iterable,
                  conf_thresh: float = 0.2,
                  draw_points: bool = True,
                  draw_lines: bool = True) -> None:
    """
    Draw a COCO skeleton on 'frame' given 'keypoints'.

    keypoints: Nx2 or Nx3 (x, y[, conf]) in COCO 17-kp order.
    conf_thresh: minimum confidence to draw a joint/limb.
    """
    pts = _to_xyc_array(keypoints)
    if pts is None:
        return

    n_kp = pts.shape[0]
    if n_kp < 17:
        # If fewer than 17 are provided, draw what we can safely index.
        edges = [e for e in COCO_SKELETON if e[0] < n_kp and e[1] < n_kp]
    else:
        edges = COCO_SKELETON

    thick, radius = _auto_thickness(frame.shape)

    # Draw limbs first so joints are on top
    if draw_lines:
        for i1, i2 in edges:
            x1, y1, c1 = pts[i1]
            x2, y2, c2 = pts[i2]
            if c1 >= conf_thresh and c2 >= conf_thresh:
                col = _edge_color(i1, i2)
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, max(1, thick))

    # Draw joints
    if draw_points:
        for i, (x, y, c) in enumerate(pts):
            if c < conf_thresh:
                continue
            # choose color group per side
            if i in {1,3,5,7,9,11,13,15}:      # left
                col = COLOR_LEFT
            elif i in {2,4,6,8,10,12,14,16}:   # right
                col = COLOR_RIGHT
            else:                               # mid/face (0)
                col = COLOR_FACE
            cv2.circle(frame, (int(x), int(y)), radius + 1, (0, 0, 0), -1)  # outline
            cv2.circle(frame, (int(x), int(y)), radius, col, -1)
