# -*- coding: utf-8 -*-

"""
Cosine-similarity + IoU-gated tracker with TTL.
Converts per-frame OSNet features into stable IDs.
"""

from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    ub = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = ua + ub - inter + 1e-12
    return float(inter / union)

class CosineReIDTracker:
    def __init__(self, sim_thr: float = 0.45, iou_thr: float = 0.20, ttl: int = 60):
        self.sim_thr = float(sim_thr)
        self.iou_thr = float(iou_thr)
        self.ttl = int(ttl)
        self._next_id = 1
        self._tracks: Dict[int, dict] = {}  # id -> {"feat": vec/None, "box": xyxy, "age": int}

    def _match(self, box, feat) -> Optional[int]:
        best_id, best = None, -1.0
        for tid, t in self._tracks.items():
            if _iou(box, t["box"]) < self.iou_thr:
                continue
            if feat is None or t["feat"] is None:
                sim = -1.0
            else:
                sim = float(np.dot(feat, t["feat"]))  # L2-normalized
            if sim > best:
                best, best_id = sim, tid
        return best_id if best >= self.sim_thr else None

    def update(self, boxes: List[List[float]], feats: List[Optional[np.ndarray]]) -> List[int]:
        # age & prune
        for t in self._tracks.values(): t["age"] += 1
        for tid in [tid for tid, t in self._tracks.items() if t["age"] > self.ttl]:
            self._tracks.pop(tid, None)

        ids: List[int] = []
        for i, box in enumerate(boxes):
            feat = feats[i] if i < len(feats) else None
            mid = self._match(box, feat)
            if mid is None:
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = {"feat": feat, "box": box, "age": 0}
                ids.append(tid)
            else:
                self._tracks[mid].update({"feat": feat or self._tracks[mid]["feat"], "box": box, "age": 0})
                ids.append(mid)
        return ids
