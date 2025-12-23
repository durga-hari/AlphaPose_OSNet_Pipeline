#!/usr/bin/env python3
import os
import sys
from typing import List, Dict, Any

import numpy as np

# -------------------------------------------------------------------
# Add your RTMO DeepSORT path to sys.path so Python can find it
# -------------------------------------------------------------------
DEEPSORT_PATH = "/home/arun_remote/DaRA_Thesis/RTMO_DeepSORT_Pipeline/externals/deepsort"
if DEEPSORT_PATH not in sys.path:
    sys.path.append(DEEPSORT_PATH)

# -------------------------------------------------------------------
# Import DeepSORT core classes
# -------------------------------------------------------------------
from deep_sort.sort.detection import Detection
from deep_sort.sort.tracker import Tracker
from deep_sort.sort.nn_matching import NearestNeighborDistanceMetric


def xyxy_to_tlwh(bbox_xyxy):
    """Convert [x1,y1,x2,y2] → [x,y,w,h]."""
    return [bbox_xyxy[0], bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1]]


def _iou_xyxy(a, b) -> float:
    """IoU for [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    ub = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = ua + ub - inter + 1e-12
    return float(inter / union)


class DeepSortTracker:
    """
    Wrapper for DeepSORT tracker.

    Two usage patterns:

    1. RTMO-style (existing):
       tracks = tracker.update(boxes, scores, feats, kps_list)
         - boxes: list of [x1,y1,x2,y2]
         - scores: list of detection confidences
         - feats: (N,D) embeddings
         - kps_list: list of keypoints
         returns: list[ { "id", "bbox", "keypoints" } ]

    2. AlphaPose pipeline:
       ids = tracker.update_from_pipeline(boxes_xyxy, feats)
         - boxes_xyxy: np.ndarray (N,4)
         - feats: np.ndarray or tensor (N,D)
         returns: list[int] track IDs aligned with each box
    """

    def __init__(self, max_cosine_distance=0.2,
                 max_iou_distance=0.7, max_age=30, n_init=3):
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, None
        )
        self.tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
        )

    # ------------------------ RTMO-style API ------------------------
    def update(self, boxes, scores, feats, kps_list):
        """
        Args:
            boxes: list of [x1,y1,x2,y2] detections
            scores: list of detection confidences
            feats: torch.Tensor or list/np.ndarray (N,D)
            kps_list: list of keypoints arrays
        Returns:
            list of dicts {id, bbox, keypoints}
        """
        if boxes is None or len(boxes) == 0:
            self.tracker.predict()
            return []

        # Ensure embeddings are numpy float32 on CPU
        if isinstance(feats, np.ndarray):
            features_np = feats.astype(np.float32)
        elif hasattr(feats, "cpu"):
            features_np = feats.detach().cpu().numpy().astype(np.float32)
        else:
            features_np = np.array(feats, dtype=np.float32)

        detections = []
        for b, s, f in zip(boxes, scores, features_np):
            tlwh = xyxy_to_tlwh(b)
            det = Detection(tlwh, float(s), 1, f)  # class_id=1 (person)
            detections.append(det)

        # Predict + Update
        self.tracker.predict()
        self.tracker.update(detections)

        tracks_out = []
        for t in self.tracker.tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            tid = int(t.track_id)
            tb = t.to_tlwh()
            box = [tb[0], tb[1], tb[0] + tb[2], tb[1] + tb[3]]

            # keypoints are optional; keep zeros to satisfy callers
            kps = np.zeros((17, 3), dtype=np.float32)
            tracks_out.append(
                {
                    "id": tid,
                    "bbox": box,
                    "keypoints": kps,
                }
            )

        return tracks_out

    # ----------------- AlphaPose-pipeline helper -------------------
    def update_from_pipeline(
        self, boxes_xyxy: np.ndarray, feats: np.ndarray | None
    ) -> List[int]:
        """
        Pipeline-facing helper.

        boxes_xyxy: np.ndarray (N,4)
        feats: np.ndarray / tensor (N,D) or None
        returns: list[int] of track IDs aligned with the input boxes.
        """
        if boxes_xyxy is None or getattr(boxes_xyxy, "size", 0) == 0:
            self.tracker.predict()
            return []

        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        N = boxes.shape[0]

        # Scores: dummy 1.0 if not available
        scores = [1.0] * N

        # Keypoints: not used; fill with zeros
        kps_list = [np.zeros((17, 3), dtype=np.float32) for _ in range(N)]

        # If no features provided, fall back to zeros (motion-only tracking)
        if feats is None:
            feats_np = np.zeros((N, 512), dtype=np.float32)
        elif isinstance(feats, np.ndarray):
            feats_np = feats.astype(np.float32)
        elif hasattr(feats, "cpu"):
            feats_np = feats.detach().cpu().numpy().astype(np.float32)
        else:
            feats_np = np.array(feats, dtype=np.float32)

        tracks = self.update(
            boxes=boxes,
            scores=scores,
            feats=feats_np,
            kps_list=kps_list,
        )

        # Map each detection to the best-matching track via IoU
        ids = [-1] * N
        for i, b in enumerate(boxes):
            best_iou = 0.0
            best_id = -1
            for t in tracks:
                tiou = _iou_xyxy(b, t["bbox"])
                if tiou > best_iou:
                    best_iou = tiou
                    best_id = int(t["id"])
            if best_iou > 0.1:
                ids[i] = best_id

        return ids
