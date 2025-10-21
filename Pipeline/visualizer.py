# # Pipeline/visualizer.py
# from __future__ import annotations
# import cv2
# import numpy as np
# from collections import defaultdict

# __all__ = ["draw_bbox_and_id", "draw_skeleton", "visualize_heatmap"]

# # =====================================================
# # Skeleton definitions
# # =====================================================

# # COCO (17 kpts)
# COCO_LIMBS = [
#     (0, 1), (0, 2), (1, 3), (2, 4),
#     (0, 5), (0, 6), (5, 7), (7, 9),
#     (6, 8), (8, 10), (5, 6),
#     (5, 11), (6, 12), (11, 12),
#     (11, 13), (13, 15), (12, 14), (14, 16),
# ]

# # COCO WholeBody slices
# COCO_WB_BODY_SLICE   = slice(0, 17)
# COCO_WB_FEET_SLICE   = slice(17, 23)
# COCO_WB_FACE68_SLICE = slice(23, 91)
# COCO_WB_LHAND_SLICE  = slice(91, 112)
# COCO_WB_RHAND_SLICE  = slice(112, 133)

# COCO_WB_BODY_EDGES = COCO_LIMBS
# COCO_WB_FEET_EDGES = [
#     (15, 17), (17, 19), (15, 18), (18, 19),
#     (16, 20), (20, 22), (16, 21), (21, 22),
# ]

# # HALPE-136 slices
# HALPE_BODY26_SLICE  = slice(0, 26)
# HALPE_FACE68_SLICE  = slice(26, 94)
# HALPE_LHAND21_SLICE = slice(94, 115)
# HALPE_RHAND21_SLICE = slice(115, 136)

# # Full HALPE-26 body edges (based on official spec)
# HALPE_BODY_EDGES = [
#     # head/torso
#     (0,1),(0,14),(0,15),(14,16),(15,17),
#     (1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
#     (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
#     (10,20),(20,22),(22,24),(13,21),(21,23),(23,25),
# ]

# # Hands (21 kpts)
# HAND21_CHAINS = [
#     (0,1),(1,2),(2,3),(3,4),
#     (0,5),(5,6),(6,7),(7,8),
#     (0,9),(9,10),(10,11),(11,12),
#     (0,13),(13,14),(14,15),(15,16),
#     (0,17),(17,18),(18,19),(19,20),
# ]

# # =====================================================
# # Temporal smoothing state
# # =====================================================
# # track_id → smoothed keypoints
# _smooth_state: dict[int, np.ndarray] = defaultdict(lambda: None)
# EMA_ALPHA = 0.4  # smoothing strength (lower = smoother, higher = more responsive)

# # =====================================================
# # Helpers
# # =====================================================
# def _inside_expanded_bbox(pt, bbox, margin: float = 0.1) -> bool:
#     x, y = pt
#     x1, y1, x2, y2 = map(int, bbox)
#     w = max(1, x2 - x1); h = max(1, y2 - y1)
#     x1 -= int(margin * w); y1 -= int(margin * h)
#     x2 += int(margin * w); y2 += int(margin * h)
#     return (x1 <= x <= x2) and (y1 <= y <= y2)

# def _draw_lines(img, kpts, edges, kpt_thresh, color, thick, bbox=None, max_rel_len=1.3):
#     K = kpts.shape[0]
#     diag = None
#     if bbox is not None:
#         x1, y1, x2, y2 = map(int, bbox)
#         diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
#         if diag < 1: diag = 1.0

#     H, W = img.shape[:2]
#     for a, b in edges:
#         if a >= K or b >= K: continue
#         xa, ya, ca = kpts[a]; xb, yb, cb = kpts[b]
#         if ca < kpt_thresh or cb < kpt_thresh: continue

#         pa, pb = (int(xa), int(ya)), (int(xb), int(yb))

#         # clip to frame
#         if not (0 <= pa[0] < W and 0 <= pa[1] < H): continue
#         if not (0 <= pb[0] < W and 0 <= pb[1] < H): continue

#         if bbox is not None:
#             if not (_inside_expanded_bbox(pa, bbox) and _inside_expanded_bbox(pb, bbox)):
#                 continue
#             dist = ((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) ** 0.5
#             if dist > max_rel_len * diag: continue

#         cv2.line(img, pa, pb, color, thick, cv2.LINE_AA)

# def _apply_smoothing(kpts: np.ndarray, track_id: int | None) -> np.ndarray:
#     if track_id is None: return kpts
#     prev = _smooth_state[track_id]
#     if prev is None:
#         _smooth_state[track_id] = kpts.copy()
#         return kpts
#     smoothed = EMA_ALPHA * kpts + (1 - EMA_ALPHA) * prev
#     _smooth_state[track_id] = smoothed
#     return smoothed

# # =====================================================
# # Public API
# # =====================================================
# def draw_bbox_and_id(img, box_xyxy, track_id=None, score=None, color=(0,255,0)):
#     x1, y1, x2, y2 = map(int, box_xyxy)
#     cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
#     label = []
#     if track_id is not None: label.append(f"ID {int(track_id)}")
#     if score is not None:    label.append(f"{float(score):.2f}")
#     if label:
#         cv2.putText(img, " ".join(label), (x1, max(0, y1-6)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# def draw_skeleton(
#     img: np.ndarray,
#     kpts: np.ndarray,
#     kpt_thresh: float = 0.5,
#     dataset: str = "coco",
#     draw_face: bool = False,
#     draw_hands: bool = True,
#     bbox=None,
#     hand_kpt_thresh: float = 0.35,
#     wrist_kpt_thresh: float = 0.4,
#     track_id: int | None = None,
# ) -> None:
#     if kpts.ndim != 2 or kpts.shape[1] < 2:
#         return

#     ds = dataset.lower()
#     kpts = _apply_smoothing(kpts, track_id)

#     # draw keypoint dots
#     H, W = img.shape[:2]
#     for i, (x, y, c) in enumerate(kpts):
#         thr = hand_kpt_thresh if (ds in ("coco_wholebody","halpe") and i >= 17) else kpt_thresh
#         if c >= thr and 0 <= int(x) < W and 0 <= int(y) < H:
#             cv2.circle(img, (int(x), int(y)), 2, (255,255,255), -1, cv2.LINE_AA)

#     if ds == "coco":
#         _draw_lines(img, kpts, COCO_LIMBS, kpt_thresh, (0,255,0), 2, bbox=bbox)

#     elif ds == "coco_wholebody":
#         _draw_lines(img, kpts, COCO_WB_BODY_EDGES, kpt_thresh, (0,255,0), 2, bbox=bbox)
#         _draw_lines(img, kpts, COCO_WB_FEET_EDGES, max(0.35,0.9*kpt_thresh), (0,255,255), 2, bbox=bbox)
#         if draw_hands:
#             if kpts.shape[0] >= 112 and float(kpts[91,2]) >= wrist_kpt_thresh:
#                 _draw_lines(img, kpts[COCO_WB_LHAND_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
#             if kpts.shape[0] >= 133 and float(kpts[112,2]) >= wrist_kpt_thresh:
#                 _draw_lines(img, kpts[COCO_WB_RHAND_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
#         if draw_face and kpts.shape[0] >= 91:
#             face = kpts[COCO_WB_FACE68_SLICE]
#             for i in range(1, face.shape[0]):
#                 x1,y1,c1 = face[i-1]; x2,y2,c2 = face[i]
#                 if c1 >= hand_kpt_thresh and c2 >= hand_kpt_thresh:
#                     cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),(200,200,255),1,cv2.LINE_AA)

#     elif ds == "halpe":
#         _draw_lines(img, kpts[HALPE_BODY26_SLICE], HALPE_BODY_EDGES, kpt_thresh, (0,255,255), 2, bbox=bbox)
#         if draw_hands:
#             if kpts.shape[0] >= 115 and float(kpts[94,2]) >= wrist_kpt_thresh:
#                 _draw_lines(img, kpts[HALPE_LHAND21_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
#             if kpts.shape[0] >= 136 and float(kpts[115,2]) >= wrist_kpt_thresh:
#                 _draw_lines(img, kpts[HALPE_RHAND21_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
#         if draw_face and kpts.shape[0] >= 94:
#             face = kpts[HALPE_FACE68_SLICE]
#             for i in range(1, face.shape[0]):
#                 x1,y1,c1 = face[i-1]; x2,y2,c2 = face[i]
#                 if c1 >= hand_kpt_thresh and c2 >= hand_kpt_thresh:
#                     cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),(200,200,255),1,cv2.LINE_AA)

# # =====================================================
# # Heatmap overlay
# # =====================================================
# def visualize_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray, bbox=None, alpha: float = 0.5) -> np.ndarray:
#     hm = heatmap.max(axis=0) if heatmap.ndim == 3 else heatmap
#     hm = (hm - hm.min()) / (np.ptp(hm) + 1e-6)
#     hm_color = cv2.applyColorMap((hm*255).astype(np.uint8), cv2.COLORMAP_JET)

#     if bbox is None:
#         hm_resized = cv2.resize(hm_color, (frame_bgr.shape[1], frame_bgr.shape[0]))
#         return cv2.addWeighted(frame_bgr, 1.0, hm_resized, alpha, 0)

#     x1,y1,x2,y2 = map(int, bbox)
#     x1 = max(0, min(x1, frame_bgr.shape[1]-1))
#     y1 = max(0, min(y1, frame_bgr.shape[0]-1))
#     x2 = max(0, min(x2, frame_bgr.shape[1]))
#     y2 = max(0, min(y2, frame_bgr.shape[0]))
#     if x2 <= x1 or y2 <= y1:
#         return frame_bgr

#     hm_resized = cv2.resize(hm_color, (x2-x1, y2-y1))
#     overlay = frame_bgr.copy()
#     roi = frame_bgr[y1:y2, x1:x2]
#     overlay[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0, hm_resized, alpha, 0)
#     cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,255), 1)
#     return overlay

# Pipeline/visualizer.py
from __future__ import annotations
import cv2
import numpy as np
from collections import defaultdict

__all__ = ["draw_bbox_and_id", "draw_skeleton", "visualize_heatmap"]

# =====================================================
# Skeleton definitions
# =====================================================

# COCO (17 kpts)
COCO_LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# COCO WholeBody slices
COCO_WB_BODY_SLICE   = slice(0, 17)
COCO_WB_FACE68_SLICE = slice(23, 91)
COCO_WB_LHAND_SLICE  = slice(91, 112)
COCO_WB_RHAND_SLICE  = slice(112, 133)

# HALPE-136 slices
HALPE_BODY26_SLICE  = slice(0, 26)
HALPE_FACE68_SLICE  = slice(26, 94)
HALPE_LHAND21_SLICE = slice(94, 115)
HALPE_RHAND21_SLICE = slice(115, 136)

# HALPE-26 edges
HALPE_BODY_EDGES = [
    (0,1),(0,14),(0,15),(14,16),(15,17),
    (1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
    (10,20),(20,22),(22,24),(13,21),(21,23),(23,25),
]

# Hands (21 kpts)
HAND21_CHAINS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

# =====================================================
# Temporal smoothing state
# =====================================================
_smooth_state: dict[int, np.ndarray] = defaultdict(lambda: None)
EMA_ALPHA = 0.4  # smoothing strength

# =====================================================
# Helpers
# =====================================================
def _inside_expanded_bbox(pt, bbox, margin: float = 0.1) -> bool:
    x, y = pt
    x1, y1, x2, y2 = map(int, bbox)
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    x1 -= int(margin * w); y1 -= int(margin * h)
    x2 += int(margin * w); y2 += int(margin * h)
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def _draw_lines(img, kpts, edges, kpt_thresh, color, thick, bbox=None, max_rel_len=0.6):
    K = kpts.shape[0]
    diag = None
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if diag < 1: diag = 1.0

    H, W = img.shape[:2]
    for a, b in edges:
        if a >= K or b >= K: continue
        xa, ya, ca = kpts[a]; xb, yb, cb = kpts[b]
        if ca < kpt_thresh or cb < kpt_thresh: continue

        pa, pb = (int(xa), int(ya)), (int(xb), int(yb))
        if not (0 <= pa[0] < W and 0 <= pa[1] < H): continue
        if not (0 <= pb[0] < W and 0 <= pb[1] < H): continue

        if bbox is not None:
            if not (_inside_expanded_bbox(pa, bbox) and _inside_expanded_bbox(pb, bbox)):
                continue
            dist = ((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) ** 0.5
            if dist > max_rel_len * diag: continue

        cv2.line(img, pa, pb, color, thick, cv2.LINE_AA)

def _apply_smoothing(
    kpts: np.ndarray,
    track_id: int | None,
    jump_thresh: float = 50.0,
    alpha: float = EMA_ALPHA,
) -> np.ndarray:
    """
    Hybrid smoothing: EMA + jump rejection
    - Normal frames: exponential moving average (EMA) for stability.
    - Sudden outliers: if a keypoint jumps more than jump_thresh pixels, keep the old smoothed value.
    """
    if track_id is None:
        return kpts

    prev = _smooth_state[track_id]
    if prev is None:
        _smooth_state[track_id] = kpts.copy()
        return kpts

    smoothed = prev.copy()
    for i in range(kpts.shape[0]):
        x_new, y_new, c_new = kpts[i]
        x_prev, y_prev, c_prev = prev[i]

        # Confidence check: keep old if new is very weak
        if c_new < 0.5:
            continue

        dist = np.linalg.norm([x_new - x_prev, y_new - y_prev])

        if dist < jump_thresh:
            # Normal update: EMA
            smoothed[i] = alpha * kpts[i] + (1 - alpha) * prev[i]
        else:
            # Reject outlier: keep old smoothed point
            smoothed[i] = prev[i]

    _smooth_state[track_id] = smoothed
    return smoothed


# =====================================================
# Public API
# =====================================================
def draw_bbox_and_id(img, box_xyxy, track_id=None, score=None, color=(0,255,0)):
    x1, y1, x2, y2 = map(int, box_xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    label = []
    if track_id is not None: label.append(f"ID {int(track_id)}")
    if score is not None:    label.append(f"{float(score):.2f}")
    if label:
        cv2.putText(img, " ".join(label), (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_skeleton(
    img: np.ndarray,
    kpts: np.ndarray,
    kpt_thresh: float = 0.5,
    dataset: str = "coco",
    draw_face: bool = False,
    draw_hands: bool = True,
    bbox=None,
    hand_kpt_thresh: float = 0.35,
    wrist_kpt_thresh: float = 0.4,
    track_id: int | None = None,
) -> None:
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        return

    ds = dataset.lower()
    kpts = _apply_smoothing(kpts, track_id)

    # draw keypoint dots
    H, W = img.shape[:2]
    for i, (x, y, c) in enumerate(kpts):
        thr = hand_kpt_thresh if (ds in ("coco_wholebody","halpe") and i >= 17) else kpt_thresh
        if c >= thr and 0 <= int(x) < W and 0 <= int(y) < H:
            cv2.circle(img, (int(x), int(y)), 2, (255,255,255), -1, cv2.LINE_AA)

    if ds == "coco":
        _draw_lines(img, kpts, COCO_LIMBS, kpt_thresh, (0,255,0), 2, bbox=bbox)

    elif ds == "coco_wholebody":
        # Only draw 17 COCO body limbs
        _draw_lines(img, kpts[COCO_WB_BODY_SLICE], COCO_LIMBS, kpt_thresh, (0,255,0), 2, bbox=bbox)
        # Draw hands with detail
        if draw_hands:
            if kpts.shape[0] >= 112 and float(kpts[91,2]) >= wrist_kpt_thresh:
                _draw_lines(img, kpts[COCO_WB_LHAND_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
            if kpts.shape[0] >= 133 and float(kpts[112,2]) >= wrist_kpt_thresh:
                _draw_lines(img, kpts[COCO_WB_RHAND_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
        # Face = abstract → just dots (no lines)
        if draw_face and kpts.shape[0] >= 91:
            face = kpts[COCO_WB_FACE68_SLICE]
            for (x, y, c) in face:
                if c >= 0.6:
                    cv2.circle(img, (int(x), int(y)), 1, (200,200,255), -1, cv2.LINE_AA)

    elif ds == "halpe":
        _draw_lines(img, kpts[HALPE_BODY26_SLICE], HALPE_BODY_EDGES, kpt_thresh, (0,255,255), 2, bbox=bbox)
        if draw_hands:
            if kpts.shape[0] >= 115 and float(kpts[94,2]) >= wrist_kpt_thresh:
                _draw_lines(img, kpts[HALPE_LHAND21_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
            if kpts.shape[0] >= 136 and float(kpts[115,2]) >= wrist_kpt_thresh:
                _draw_lines(img, kpts[HALPE_RHAND21_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
        if draw_face and kpts.shape[0] >= 94:
            face = kpts[HALPE_FACE68_SLICE]
            for (x, y, c) in face:
                if c >= 0.6:
                    cv2.circle(img, (int(x), int(y)), 1, (200,200,255), -1, cv2.LINE_AA)

# =====================================================
# Heatmap overlay
# =====================================================
def visualize_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray, bbox=None, alpha: float = 0.5) -> np.ndarray:
    hm = heatmap.max(axis=0) if heatmap.ndim == 3 else heatmap
    hm = (hm - hm.min()) / (np.ptp(hm) + 1e-6)
    hm_color = cv2.applyColorMap((hm*255).astype(np.uint8), cv2.COLORMAP_JET)

    if bbox is None:
        hm_resized = cv2.resize(hm_color, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return cv2.addWeighted(frame_bgr, 1.0, hm_resized, alpha, 0)

    x1,y1,x2,y2 = map(int, bbox)
    x1 = max(0, min(x1, frame_bgr.shape[1]-1))
    y1 = max(0, min(y1, frame_bgr.shape[0]-1))
    x2 = max(0, min(x2, frame_bgr.shape[1]))
    y2 = max(0, min(y2, frame_bgr.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return frame_bgr

    hm_resized = cv2.resize(hm_color, (x2-x1, y2-y1))
    overlay = frame_bgr.copy()
    roi = frame_bgr[y1:y2, x1:x2]
    overlay[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0, hm_resized, alpha, 0)
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,255), 1)
    return overlay
