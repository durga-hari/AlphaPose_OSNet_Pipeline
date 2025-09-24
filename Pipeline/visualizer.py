# Pipeline/visualizer.py
from __future__ import annotations
import cv2
import numpy as np

__all__ = ["draw_bbox_and_id", "draw_skeleton", "visualize_heatmap"]

# =========================
# COCO (17 kpts)
# =========================
COCO_LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# =========================
# COCO-WholeBody (133 kpts)
# =========================
COCO_WB_BODY_SLICE   = slice(0, 17)
COCO_WB_FEET_SLICE   = slice(17, 23)
COCO_WB_FACE68_SLICE = slice(23, 91)
COCO_WB_LHAND_SLICE  = slice(91, 112)
COCO_WB_RHAND_SLICE  = slice(112, 133)

COCO_WB_BODY_EDGES = COCO_LIMBS
COCO_WB_FEET_EDGES = [
    (15, 17), (17, 19),   # L ankle → L bigtoe → L heel
    (15, 18), (18, 19),
    (16, 20), (20, 22),   # R ankle → R bigtoe → R heel
    (16, 21), (21, 22),
]

# =========================
# HALPE-136 (136 kpts)
# =========================
HALPE_BODY26_SLICE  = slice(0, 26)
HALPE_FACE68_SLICE  = slice(26, 94)
HALPE_LHAND21_SLICE = slice(94, 115)
HALPE_RHAND21_SLICE = slice(115, 136)

HALPE_BODY_EDGES = [
    # arms
    (1,2),(2,3),(3,4),
    (1,5),(5,6),(6,7),
    # legs
    (1,8),(8,9),(9,10),
    (1,11),(11,12),(12,13),
    # feet
    (10,20),(20,22),(22,24),
    (13,21),(21,23),(23,25),
    # trunk/head minimal
    (0,1),(0,14),(14,16),(0,15),(15,17),
]

# =========================
# Hands (21)
# =========================
HAND21_CHAINS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
]

# =========================
# helpers
# =========================
def _inside_expanded_bbox(pt: tuple[int, int], bbox: tuple[int, int, int, int], margin: float = 0.1) -> bool:
    x, y = pt
    x1, y1, x2, y2 = map(int, bbox)
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    x1 -= int(margin * w); y1 -= int(margin * h)
    x2 += int(margin * w); y2 += int(margin * h)
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def _draw_lines(
    img: np.ndarray,
    kpts: np.ndarray,
    edges: list[tuple[int, int]],
    kpt_thresh: float,
    color: tuple[int, int, int],
    thick: int,
    bbox: tuple[float, float, float, float] | None = None,
    max_rel_len: float = 1.3,
) -> None:
    K = kpts.shape[0]
    diag = None
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if diag < 1:
            diag = 1.0

    H, W = img.shape[:2]

    for a, b in edges:
        if a >= K or b >= K:
            continue
        xa, ya, ca = kpts[a]
        xb, yb, cb = kpts[b]
        if ca < kpt_thresh or cb < kpt_thresh:
            continue
        pa = (int(xa), int(ya)); pb = (int(xb), int(yb))

        if not (0 <= pa[0] < W and 0 <= pa[1] < H): 
            continue
        if not (0 <= pb[0] < W and 0 <= pb[1] < H): 
            continue

        if bbox is not None:
            if not (_inside_expanded_bbox(pa, bbox) and _inside_expanded_bbox(pb, bbox)):
                continue
            dist = ((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) ** 0.5
            if dist > max_rel_len * diag:
                continue

        cv2.line(img, pa, pb, color, thick, cv2.LINE_AA)

# =========================
# public api
# =========================
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
    dataset: str = "coco",   # "coco" | "coco_wholebody" | "halpe"
    draw_face: bool = False,
    draw_hands: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
    hand_kpt_thresh: float = 0.25,
    wrist_kpt_thresh: float = 0.25,
) -> None:
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        return

    ds = dataset.lower()

    # draw dots
    for i, (x, y, c) in enumerate(kpts):
        thr = hand_kpt_thresh if (ds in ("coco_wholebody","halpe") and i >= 17) else kpt_thresh
        if c >= thr:
            cv2.circle(img, (int(x), int(y)), 2, (255,255,255), -1, cv2.LINE_AA)

    if ds == "coco":
        _draw_lines(img, kpts, COCO_LIMBS, kpt_thresh, (0,255,0), 2, bbox=bbox)

    elif ds == "coco_wholebody":
        _draw_lines(img, kpts, COCO_WB_BODY_EDGES, kpt_thresh, (0,255,0), 2, bbox=bbox)
        _draw_lines(img, kpts, COCO_WB_FEET_EDGES, max(0.35,0.9*kpt_thresh), (0,255,255), 2, bbox=bbox)
        if draw_hands:
            if kpts.shape[0] >= 112 and float(kpts[91,2]) >= wrist_kpt_thresh:
                _draw_lines(img, kpts[COCO_WB_LHAND_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
            if kpts.shape[0] >= 133 and float(kpts[112,2]) >= wrist_kpt_thresh:
                _draw_lines(img, kpts[COCO_WB_RHAND_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
        if draw_face and kpts.shape[0] >= 91:
            face = kpts[COCO_WB_FACE68_SLICE]
            for i in range(1, face.shape[0]):
                x1,y1,c1 = face[i-1]; x2,y2,c2 = face[i]
                if c1 >= hand_kpt_thresh and c2 >= hand_kpt_thresh:
                    cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),(200,200,255),1,cv2.LINE_AA)

    elif ds == "halpe":
        _draw_lines(img, kpts[HALPE_BODY26_SLICE], HALPE_BODY_EDGES, kpt_thresh, (0,255,255), 2, bbox=bbox)
        if draw_hands:
            if kpts.shape[0] >= 115 and float(kpts[94,2]) >= wrist_kpt_thresh:
                _draw_lines(img, kpts[HALPE_LHAND21_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
            if kpts.shape[0] >= 136 and float(kpts[115,2]) >= wrist_kpt_thresh:
                _draw_lines(img, kpts[HALPE_RHAND21_SLICE], HAND21_CHAINS, hand_kpt_thresh, (0,200,255), 2, bbox=bbox)
        if draw_face and kpts.shape[0] >= 94:
            face = kpts[HALPE_FACE68_SLICE]
            for i in range(1, face.shape[0]):
                x1,y1,c1 = face[i-1]; x2,y2,c2 = face[i]
                if c1 >= hand_kpt_thresh and c2 >= hand_kpt_thresh:
                    cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),(200,200,255),1,cv2.LINE_AA)

# =========================
# Heatmap overlay
# =========================
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
