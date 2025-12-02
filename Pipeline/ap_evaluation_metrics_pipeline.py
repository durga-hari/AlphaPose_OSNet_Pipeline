#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified unsupervised metrics for DaRA pipelines (no ground truth).

Modes:
  - rtmo : RTMO + DeepSORT outputs in:
             /home/arun_remote/DaRA_Thesis/Output_RTMO_DeepSORT
           (any **/*_poses_union.jsonl + optional merged_global_tracks_union.jsonl)

  - ap   : AlphaPose + OSNet outputs in:
             /home/arun_remote/DaRA_Thesis/Output_AP_OSNET
           (any **/*_poses_union.jsonl + optional merged_global_tracks_union.jsonl)

For BOTH modes it writes ONLY CSVs (no JSON):

  per_track.csv     : per-track metrics (union of RTMO + AP track metrics)
  per_video.csv     : per-file/camera metrics (union of RTMO + AP dataset metrics)
  overall.csv       : single row aggregated across all files in that root
  crosscam.csv      : global cross-camera metrics (if global file present)
  limb_lengths.csv  : per-frame limb lengths for each track & bone
"""

from __future__ import annotations

import argparse
import json
import math
import glob
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Global defaults for your setup
# ---------------------------------------------------------------------------

DEFAULT_RTMO_ROOT = "/home/arun_remote/DaRA_Thesis/Output_RTMO_DeepSORT"
DEFAULT_AP_ROOT   = "/home/arun_remote/DaRA_Thesis/Output_AP_OSNET"
DEFAULT_GLOBAL_FILE = "merged_global_tracks_union.jsonl"

IOU_CONT_THR   = 0.3     # for proxy id switches / continuity
EMBED_JUMP_THR = 0.7


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max((box_a[2] - box_a[0]), 0.0) * max((box_a[3] - box_a[1]), 0.0)
    area_b = max((box_b[2] - box_b[0]), 0.0) * max((box_b[3] - box_b[1]), 0.0)
    union = max(area_a + area_b - inter, 1e-9)
    return float(inter / union)


def flatten_keypoints_xy(keypoints: np.ndarray) -> np.ndarray:
    """
    keypoints: [J, 2 or 3] -> [2J] (x1,y1,x2,y2,...)
    """
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise ValueError("keypoints must be [J,>=2]")
    return keypoints[:, :2].reshape(-1)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# RTMO: JSON parsing → frame_records[]
# ---------------------------------------------------------------------------

def rtmo_maybe_flatten_kps(kps):
    # Accept [x,y,score]*K or [[x,y,score], ...]
    if kps is None:
        return None
    if len(kps) == 0:
        return []
    if isinstance(kps[0], (list, tuple, np.ndarray)):
        flat = []
        for tri in kps:
            if len(tri) == 3:
                flat.extend(tri)
            elif len(tri) == 2:
                flat.extend([tri[0], tri[1], 1.0])
        return flat
    return kps


def rtmo_keypoints_to_arr(kps_flat, k: int = 17) -> Optional[np.ndarray]:
    """
    Convert flat [x,y,score,...] to [K,3] (x,y,score), padding/truncating to K joints.
    """
    if kps_flat is None:
        return None
    arr = np.asarray(kps_flat, dtype=np.float32)
    if arr.size == 0 or arr.size % 3 != 0:
        return None
    arr = arr.reshape(-1, 3)
    if arr.shape[0] < k:
        pad = np.full((k - arr.shape[0], 3), np.nan, dtype=np.float32)
        arr = np.vstack([arr, pad])
    elif arr.shape[0] > k:
        arr = arr[:k]
    return arr


def parse_rtmo_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize RTMO detection record to:
      { frame_idx, track_id, bbox, score, keypoints, embedding }
    """
    try:
        frame = int(
            raw.get("frame_id", raw.get("frame", raw.get("frameIndex", -1)))
        )
        tid = raw.get("id", raw.get("track_id", raw.get("tid", None)))
        bbox = raw.get("bbox", raw.get("box", None))
        score = raw.get("conf", raw.get("score", raw.get("confidence", None)))
        feat = raw.get("feature", raw.get("reid", None))
        kps  = raw.get("kps", raw.get("keypoints", raw.get("pose", None)))

        if tid is None or bbox is None or frame < 0:
            return None

        bbox = [float(b) for b in bbox]
        if len(bbox) != 4:
            return None

        feat = None if feat is None else [float(x) for x in feat]
        kps = rtmo_maybe_flatten_kps(kps)
        kps_arr = rtmo_keypoints_to_arr(kps) if kps is not None else None

        return dict(
            frame_idx=frame,
            track_id=str(tid),
            bbox=np.asarray(bbox, dtype=float),
            score=float(score) if score is not None else float("nan"),
            keypoints=kps_arr,
            embedding=np.asarray(feat, dtype=float) if feat is not None else None,
        )
    except Exception:
        return None


def load_rtmo_records(path: Path) -> List[Dict[str, Any]]:
    """
    Load RTMO *_poses_union.jsonl or JSON into normalized detection records.
    """
    txt = path.read_text(encoding="utf-8", errors="ignore")
    items: List[Dict[str, Any]] = []

    # Try JSONL
    if "\n" in txt:
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception:
                continue
            rec = parse_rtmo_record(raw)
            if rec:
                items.append(rec)
        if items:
            return items

    # Fallback plain JSON
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            for raw in data:
                rec = parse_rtmo_record(raw)
                if rec:
                    items.append(rec)
        elif isinstance(data, dict):
            seq = data.get("frames", data.get("data", []))
            for raw in seq:
                rec = parse_rtmo_record(raw)
                if rec:
                    items.append(rec)
    except Exception:
        pass

    return items


def build_frame_records_rtmo(path: Path) -> List[Dict[str, Any]]:
    """
    RTMO records → frame_records for metrics shared with AP code.

    frame_records:
      [
        {
          "frame_idx": int,
          "cam_id": str,
          "detections": [
            {
              "track_id": ...,
              "bbox": np.array([x1,y1,x2,y2]),
              "score": float,
              "keypoints": np.ndarray[J,3] or None,
              "embedding": np.ndarray[D] or None,
            }, ...
          ]
        }, ...
      ]
    """
    # infer cam from file/parent: e.g. AC11_output/AC11_output_poses_union.jsonl -> "AC11"
    fname = path.name
    cam_id = fname.split("_")[0] if "_" in fname else path.parent.name.split("_")[0]

    dets = load_rtmo_records(path)
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for d in dets:
        by_frame[d["frame_idx"]].append({
            "track_id": d["track_id"],
            "bbox": d["bbox"],
            "score": d["score"],
            "keypoints": d["keypoints"],
            "embedding": d["embedding"],
        })

    frames: List[Dict[str, Any]] = []
    for f in sorted(by_frame.keys()):
        frames.append({
            "frame_idx": f,
            "cam_id": cam_id,
            "detections": by_frame[f],
        })
    return frames


# ---------------------------------------------------------------------------
# AP: JSONL → frame_records[]
# ---------------------------------------------------------------------------

def build_frame_records_ap(path: Path, cam_id_hint: str) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for rec in load_jsonl(path):
        frame_idx = rec.get("frame")
        cam_id = rec.get("cam_id", cam_id_hint)
        poses = rec.get("poses", []) or []
        embeds = rec.get("embeds", []) or []
        detections = []
        for i, person in enumerate(poses):
            emb = embeds[i] if i < len(embeds) else None
            kps = person.get("keypoints") or person.get("pose") or person.get("joints")
            if kps is not None:
                kps_arr = np.asarray(kps, dtype=float)
            else:
                kps_arr = None
            bbox = person.get("bbox") or person.get("box")
            bbox_arr = np.asarray(bbox, dtype=float) if bbox is not None else None
            det = {
                "track_id": person.get("id") or person.get("track_id"),
                "bbox": bbox_arr,
                "score": float(person.get("score", person.get("conf", 1.0))),
                "keypoints": kps_arr,
                "embedding": np.asarray(emb, dtype=float) if emb is not None else None,
            }
            detections.append(det)

        frames.append({
            "frame_idx": frame_idx,
            "cam_id": cam_id,
            "detections": detections,
        })
    frames.sort(key=lambda x: x["frame_idx"])
    return frames


# ---------------------------------------------------------------------------
# Tracks from frame_records
# ---------------------------------------------------------------------------

def build_tracks_from_frame_records(
    frame_records: List[Dict[str, Any]]
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    track_id -> list of entries sorted by frame_idx.
    each entry: {frame_idx, cam_id, bbox, keypoints, embedding, score}
    """
    tracks: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for fr in frame_records:
        fidx = fr["frame_idx"]
        cam_id = fr.get("cam_id", None)
        for det in fr.get("detections", []):
            tid = det.get("track_id")
            if tid is None:
                continue
            tracks[tid].append({
                "frame_idx": fidx,
                "cam_id": cam_id,
                "bbox": det.get("bbox", None),
                "keypoints": det.get("keypoints", None),
                "embedding": det.get("embedding", None),
                "score": det.get("score", float("nan")),
            })
    for tid in tracks:
        tracks[tid].sort(key=lambda e: e["frame_idx"])
    return tracks


# ---------------------------------------------------------------------------
# Track-level metrics (based on RTMO version)
# ---------------------------------------------------------------------------

# COCO-style edges and angle triplets
COCO_EDGES = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (5, 6), (5, 11), (6, 12), (11, 12), (5, 12), (6, 11)
]

ANGLE_TRIPLETS = {
    "l_elbow": (5, 7, 9),
    "r_elbow": (6, 8, 10),
    "l_knee":  (11, 13, 15),
    "r_knee":  (12, 14, 16),
}


def safe_angle(p1, p2, p3, eps=1e-6):
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return np.nan
    cosang = np.dot(v1, v2) / (n1 * n2 + eps)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def compute_track_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Per-track metrics used by PQI (IoU continuity, jitter, embedding stability, etc.).
    entries: [{frame_idx, bbox, keypoints, embedding, score}, ...] sorted by frame_idx.
    """
    frames = np.array([e["frame_idx"] for e in entries], dtype=np.int32)
    boxes  = [e["bbox"] for e in entries if e.get("bbox") is not None]
    scores = np.array([e["score"] for e in entries], dtype=np.float32)
    has_conf = np.isfinite(scores).any()

    if boxes:
        boxes_arr = np.stack(boxes, axis=0)
    else:
        boxes_arr = np.empty((0, 4), dtype=float)

    # temporal IoU + motion
    ious = []
    vels = []
    if boxes_arr.shape[0] >= 2:
        centers = 0.5 * (boxes_arr[:, :2] + boxes_arr[:, 2:])
        for i in range(len(boxes_arr) - 1):
            ious.append(bbox_iou(boxes_arr[i], boxes_arr[i + 1]))
            vels.append(np.linalg.norm(centers[i + 1] - centers[i]))
    iou_mean = float(np.mean(ious)) if ious else np.nan
    vel_var  = float(np.var(vels)) if vels else np.nan
    vel_mean = float(np.mean(vels)) if vels else np.nan

    # embedding stability
    feats_raw = [
        np.asarray(e["embedding"], dtype=np.float32)
        for e in entries
        if e.get("embedding") is not None
    ]
    embed_var = np.nan
    embed_jump_mean = np.nan
    if len(feats_raw) >= 2:
        dim0 = feats_raw[0].shape[0]
        feats_aligned = [f for f in feats_raw if f.shape[0] == dim0]
        if len(feats_aligned) >= 2:
            feats_arr = np.stack(feats_aligned, axis=0)
            embed_var = float(np.mean(np.var(feats_arr, axis=0)))
            djumps = []
            for i in range(len(feats_arr) - 1):
                d = np.linalg.norm(feats_arr[i + 1] - feats_arr[i])
                djumps.append(d)
            if djumps:
                embed_jump_mean = float(np.mean(djumps))

    # pose-based metrics
    kps_all = [
        e["keypoints"] for e in entries
        if e.get("keypoints") is not None and e["keypoints"].size > 0
    ]
    kps_conf_mean = np.nan
    kps_conf_std  = np.nan
    bone_var      = np.nan
    jitter        = np.nan
    angle_bad_rate = np.nan

    if len(kps_all) >= 2:
        # confidence per joint
        kp_confs = []
        for kp in kps_all:
            if kp.shape[1] >= 3:
                kp_confs.append(kp[:, 2])
        if kp_confs:
            confs_kp = np.concatenate(kp_confs)
            if confs_kp.size > 0:
                kps_conf_mean = float(np.nanmean(confs_kp))
                kps_conf_std  = float(np.nanstd(confs_kp))

        # bone-length variance
        lengths_over_time = []
        for kp in kps_all:
            xy = kp[:, :2]
            lens = []
            for (i, j) in COCO_EDGES:
                if i < xy.shape[0] and j < xy.shape[0]:
                    p1, p2 = xy[i], xy[j]
                    if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                        continue
                    lens.append(np.linalg.norm(p1 - p2))
            if lens:
                lengths_over_time.append(np.mean(lens))
        if lengths_over_time:
            bone_var = float(np.var(lengths_over_time))

        # jitter = mean per-joint velocity magnitude
        vels_kp = []
        for a, b in zip(kps_all[:-1], kps_all[1:]):
            if a.shape != b.shape:
                continue
            dv = b[:, :2] - a[:, :2]
            if np.isnan(dv).all():
                continue
            mag = np.linalg.norm(dv, axis=1)
            if np.isnan(mag).all():
                continue
            vels_kp.append(np.nanmean(mag))
        if vels_kp:
            jitter = float(np.nanmean(vels_kp))

        # angle plausibility
        bad_flags = []
        for kp in kps_all:
            xy = kp[:, :2]
            for name, (i, j, k) in ANGLE_TRIPLETS.items():
                if max(i, j, k) >= xy.shape[0]:
                    continue
                p1, p2, p3 = xy[i], xy[j], xy[k]
                if (
                    np.any(np.isnan(p1))
                    or np.any(np.isnan(p2))
                    or np.any(np.isnan(p3))
                ):
                    continue
                ang = safe_angle(p1, p2, p3)
                if not math.isnan(ang):
                    bad_flags.append(ang < 5 or ang > 175)
        if bad_flags:
            angle_bad_rate = float(np.mean(bad_flags))

    conf_mean = float(np.nanmean(scores)) if has_conf else np.nan
    conf_std  = float(np.nanstd(scores)) if has_conf else np.nan

    # fragmentation: gaps > 1 frame
    gaps = np.diff(frames)
    frags = int(np.sum(gaps > 1))

    return dict(
        track_len=int(len(entries)),
        frame_first=int(frames[0]),
        frame_last=int(frames[-1]),
        iou_temporal_mean=iou_mean,
        vel_mean=vel_mean,
        vel_var=vel_var,
        conf_mean=conf_mean,
        conf_std=conf_std,
        embed_var=embed_var,
        embed_jump_mean=embed_jump_mean,
        kps_conf_mean=kps_conf_mean,
        kps_conf_std=kps_conf_std,
        bone_var=bone_var,
        kp_jitter=jitter,
        angle_bad_rate=angle_bad_rate,
        fragments=frags,
    )


def compute_proxy_id_switches(
    entries: List[Dict[str, Any]],
    iou_thr: float = IOU_CONT_THR,
    embed_jump_thr: float = EMBED_JUMP_THR,
) -> Tuple[int, int]:
    """
    Proxy for ID switch inside a single track:
    count events where temporal IoU is low AND embedding jump is large.
    """
    switches = 0
    checks = 0
    if len(entries) < 2:
        return 0, 0
    for i in range(len(entries) - 1):
        b1 = entries[i].get("bbox")
        b2 = entries[i + 1].get("bbox")
        if b1 is None or b2 is None:
            continue
        iou = bbox_iou(b1, b2)
        f1 = entries[i].get("embedding")
        f2 = entries[i + 1].get("embedding")
        if f1 is None or f2 is None:
            continue
        f1 = np.asarray(f1, dtype=np.float32)
        f2 = np.asarray(f2, dtype=np.float32)
        if f1.shape != f2.shape:
            continue
        dj = float(np.linalg.norm(f1 - f2))
        checks += 1
        if iou < iou_thr and dj > embed_jump_thr:
            switches += 1
    return switches, checks


# ---------------------------------------------------------------------------
# NEW: per-frame limb-length time series
# ---------------------------------------------------------------------------

def compute_limb_length_timeseries(
    tracks: Dict[Any, List[Dict[str, Any]]],
    bones: Optional[List[Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """
    Produce a DataFrame with one row per (track, frame, bone):

      track_id, frame_idx, cam_id, bone, joint_i, joint_j, limb_length
    """
    if bones is None:
        bones = [
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (5, 7), (7, 9),
            (2, 4), (4, 6),
            (5, 6), (11, 12),
        ]

    rows = []
    for tid, entries in tracks.items():
        for e in entries:
            kps = e.get("keypoints")
            if kps is None or kps.size == 0:
                continue
            xy = kps[:, :2]
            J = xy.shape[0]
            frame_idx = e["frame_idx"]
            cam_id = e.get("cam_id")
            for (i, j) in bones:
                if i >= J or j >= J:
                    continue
                p1, p2 = xy[i], xy[j]
                if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                    continue
                length = float(np.linalg.norm(p1 - p2))
                rows.append({
                    "track_id": tid,
                    "frame_idx": frame_idx,
                    "cam_id": cam_id,
                    "bone": f"{i}-{j}",
                    "joint_i": i,
                    "joint_j": j,
                    "limb_length": length,
                })
    if not rows:
        return pd.DataFrame(
            columns=[
                "track_id",
                "frame_idx",
                "cam_id",
                "bone",
                "joint_i",
                "joint_j",
                "limb_length",
            ]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RTMO-style frame-level stats (duplicate IoU etc.)
# ---------------------------------------------------------------------------

def frame_level_stats(frame_records: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    frames = sorted({fr["frame_idx"] for fr in frame_records})
    det_counts = []
    conf_means = []
    conf_stds  = []
    dup_iou_means = []
    for f in frames:
        dets = []
        for fr in frame_records:
            if fr["frame_idx"] == f:
                dets.extend(fr.get("detections", []))
        det_counts.append(len(dets))
        scores = [d.get("score", float("nan")) for d in dets]
        scores = [s for s in scores if np.isfinite(s)]
        conf_means.append(np.mean(scores) if scores else np.nan)
        conf_stds.append(np.std(scores) if scores else np.nan)

        # intra-frame duplicate IoU
        ious = []
        boxes = [d.get("bbox") for d in dets if d.get("bbox") is not None]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ious.append(bbox_iou(boxes[i], boxes[j]))
        dup_iou_means.append(np.mean(ious) if ious else 0.0)

    return dict(
        frames=np.array(frames, dtype=np.int32),
        det_counts=np.array(det_counts, dtype=np.float32),
        conf_means=np.array(conf_means, dtype=np.float32),
        conf_stds=np.array(conf_stds, dtype=np.float32),
        dup_iou_means=np.array(dup_iou_means, dtype=np.float32),
    )


def compute_pqi(per_track_df: pd.DataFrame) -> float:
    """
    Pipeline Quality Index (0..1) – same heuristic as original RTMO script:
      0.3 * norm(iou_temporal_mean)
    + 0.2 * (1 - norm(kp_jitter))
    + 0.2 * (1 - norm(embed_var))
    + 0.3 * (1 - norm(angle_bad_rate))
    """
    def norm01(series: pd.Series) -> pd.Series:
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return series * 0 + 0.5
        lo, hi = float(s.quantile(0.05)), float(s.quantile(0.95))
        if hi <= lo:
            return series * 0 + 0.5
        return (series.clip(lo, hi) - lo) / (hi - lo)

    if per_track_df.empty:
        return float("nan")

    iou_n  = norm01(per_track_df["iou_temporal_mean"])
    jit_n  = norm01(per_track_df["kp_jitter"])
    embv_n = norm01(per_track_df["embed_var"])
    ang_n  = norm01(per_track_df["angle_bad_rate"])

    pqi_series = (
        0.3 * iou_n.fillna(0.5)
        + 0.2 * (1 - jit_n.fillna(0.5))
        + 0.2 * (1 - embv_n.fillna(0.5))
        + 0.3 * (1 - ang_n.fillna(0.5))
    )
    return float(pqi_series.mean())


# ---------------------------------------------------------------------------
# 1. Detection metrics (AP-style + extra RTMO bits)
# ---------------------------------------------------------------------------

def detection_frame_count_stats(frame_records: List[Dict[str, Any]]) -> Dict[str, float]:
    counts = [len(fr.get("detections", [])) for fr in frame_records]
    mean_c = safe_mean(counts)
    std_c  = safe_std(counts)
    fds = float(std_c / (mean_c + 1e-9)) if not math.isnan(mean_c) else float("nan")
    return {
        "mean_dets_per_frame": mean_c,
        "std_dets_per_frame": std_c,
        "frame_detection_stability": fds,
    }


def detection_temporal_persistence(frame_records: List[Dict[str, Any]],
                                   iou_thresh: float = 0.5) -> Dict[str, float]:
    frames_sorted = sorted(frame_records, key=lambda fr: fr["frame_idx"])
    if len(frames_sorted) < 2:
        return {"temporal_persistence_rate": float("nan")}
    num_dets = 0
    num_persist = 0
    for i in range(len(frames_sorted) - 1):
        fr = frames_sorted[i]
        fr_next = frames_sorted[i + 1]
        dets = fr.get("detections", [])
        dets_next = fr_next.get("detections", [])
        boxes_next = [d["bbox"] for d in dets_next if d.get("bbox") is not None]
        for det in dets:
            box = det.get("bbox", None)
            if box is None or not boxes_next:
                continue
            num_dets += 1
            best_iou = max(bbox_iou(box, b) for b in boxes_next)
            if best_iou >= iou_thresh:
                num_persist += 1
    tpr = float(num_persist / num_dets) if num_dets > 0 else float("nan")
    return {"temporal_persistence_rate": tpr}


def detection_confidence_stats(frame_records: List[Dict[str, Any]]) -> Dict[str, float]:
    scores = []
    for fr in frame_records:
        for det in fr.get("detections", []):
            s = det.get("score", None)
            if s is None:
                continue
            scores.append(float(s))
    return {
        "mean_detection_score": safe_mean(scores),
        "std_detection_score": safe_std(scores),
    }


def intraframe_dup_iou_stats(frame_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extra RTMO metric: mean intra-frame duplicate IoU across frames.
    """
    frame_ids = sorted({fr["frame_idx"] for fr in frame_records})
    per_frame_means = []
    for f in frame_ids:
        dets = []
        for fr in frame_records:
            if fr["frame_idx"] == f:
                dets.extend(fr.get("detections", []))
        boxes = [d.get("bbox") for d in dets if d.get("bbox") is not None]
        if len(boxes) < 2:
            per_frame_means.append(0.0)
            continue
        ious = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ious.append(bbox_iou(boxes[i], boxes[j]))
        per_frame_means.append(np.mean(ious) if ious else 0.0)
    return {"dup_iou_mean": safe_mean(per_frame_means)}


# ---------------------------------------------------------------------------
# 2. Pose metrics (AP-style)
# ---------------------------------------------------------------------------

def pose_confidence_stats(tracks: Dict[Any, List[Dict[str, Any]]]) -> Dict[str, float]:
    confs = []
    for _, entries in tracks.items():
        for e in entries:
            kps = e.get("keypoints", None)
            if kps is None or kps.size == 0:
                continue
            if kps.shape[1] >= 3:
                c = kps[:, 2]
            else:
                c = np.ones(kps.shape[0], dtype=float)
            confs.extend(c.tolist())
    return {
        "mean_pose_confidence": safe_mean(confs),
        "std_pose_confidence": safe_std(confs),
    }


def pose_temporal_smoothness(tracks: Dict[Any, List[Dict[str, Any]]]) -> Dict[str, float]:
    velocities = []
    for _, entries in tracks.items():
        if len(entries) < 2:
            continue
        for i in range(1, len(entries)):
            k_prev = entries[i - 1].get("keypoints")
            k_cur  = entries[i].get("keypoints")
            if k_prev is None or k_cur is None:
                continue
            if k_prev.shape != k_cur.shape:
                continue
            v = np.linalg.norm(k_cur[:, :2] - k_prev[:, :2], axis=1)
            velocities.extend(v.tolist())
    return {
        "mean_pose_velocity": safe_mean(velocities),
        "std_pose_velocity": safe_std(velocities),
    }


def pose_limb_length_consistency(
    tracks: Dict[Any, List[Dict[str, Any]]],
    bones: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    if bones is None:
        bones = [
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (5, 7), (7, 9),
            (2, 4), (4, 6),
            (5, 6), (11, 12),
        ]
    cvs = []
    for _, entries in tracks.items():
        limb_len_per_bone = {b: [] for b in bones}
        for e in entries:
            kps = e.get("keypoints")
            if kps is None:
                continue
            J = kps.shape[0]
            for b in bones:
                i, j = b
                if i >= J or j >= J:
                    continue
                l = float(np.linalg.norm(kps[i, :2] - kps[j, :2]))
                limb_len_per_bone[b].append(l)
        per_track_cvs = []
        for b, vals in limb_len_per_bone.items():
            if len(vals) < 2:
                continue
            arr = np.asarray(vals, dtype=float)
            m = np.mean(arr)
            s = np.std(arr)
            if m > 1e-6:
                per_track_cvs.append(float(s / m))
        if per_track_cvs:
            cvs.append(float(np.mean(per_track_cvs)))
    return {
        "mean_limb_length_cv_over_tracks": safe_mean(cvs),
        "std_limb_length_cv_over_tracks": safe_std(cvs),
    }


def pose_self_similarity_over_time(
    tracks: Dict[Any, List[Dict[str, Any]]]
) -> Dict[str, float]:
    sims = []
    for _, entries in tracks.items():
        if len(entries) < 2:
            continue
        prev_vec = None
        for e in entries:
            kps = e.get("keypoints")
            if kps is None:
                prev_vec = None
                continue
            vec = flatten_keypoints_xy(kps)
            if prev_vec is not None and vec.shape == prev_vec.shape:
                num = float(np.dot(vec, prev_vec))
                den = float(np.linalg.norm(vec) * np.linalg.norm(prev_vec) + 1e-9)
                sims.append(num / den)
            prev_vec = vec
    return {
        "mean_pose_temporal_cosine": safe_mean(sims),
        "std_pose_temporal_cosine": safe_std(sims),
    }


# ---------------------------------------------------------------------------
# 3. Tracking metrics (AP-style)
# ---------------------------------------------------------------------------

def tracking_basic_stats(
    tracks: Dict[Any, List[Dict[str, Any]]],
    total_frames: Optional[int],
) -> Dict[str, float]:
    lengths = [len(v) for v in tracks.values()]
    if not lengths:
        return {
            "num_tracks_basic": 0.0,
            "mean_track_length_basic": float("nan"),
            "track_fragmentation_index": float("nan"),
            "track_continuity_ratio": float("nan"),
        }
    mean_len = safe_mean(lengths)
    min_len = 10
    frac_short = float(sum(1 for L in lengths if L < min_len) / len(lengths))
    if total_frames is None or total_frames <= 0:
        tcr = float("nan")
    else:
        tcr = float(mean_len / total_frames)
    return {
        "num_tracks_basic": float(len(lengths)),
        "mean_track_length_basic": mean_len,
        "track_fragmentation_index": frac_short,
        "track_continuity_ratio": tcr,
    }


def tracking_bbox_drift(tracks: Dict[Any, List[Dict[str, Any]]]) -> Dict[str, float]:
    norm_disp = []
    for _, entries in tracks.items():
        if len(entries) < 2:
            continue
        for i in range(1, len(entries)):
            b_prev = entries[i - 1].get("bbox", None)
            b_cur  = entries[i].get("bbox", None)
            if b_prev is None or b_cur is None:
                continue
            b_prev = np.asarray(b_prev, dtype=float)
            b_cur  = np.asarray(b_cur, dtype=float)
            cx_prev = 0.5 * (b_prev[0] + b_prev[2])
            cy_prev = 0.5 * (b_prev[1] + b_prev[3])
            cx_cur  = 0.5 * (b_cur[0] + b_cur[2])
            cy_cur  = 0.5 * (b_cur[1] + b_cur[3])
            disp = math.hypot(cx_cur - cx_prev, cy_cur - cy_prev)
            bw = max(b_prev[2] - b_prev[0], 1e-6)
            bh = max(b_prev[3] - b_prev[1], 1e-6)
            scale = max(bw, bh)
            norm_disp.append(float(disp / scale))
    return {
        "mean_bbox_normalized_displacement": safe_mean(norm_disp),
        "std_bbox_normalized_displacement": safe_std(norm_disp),
    }


# ---------------------------------------------------------------------------
# 4. ReID / embedding metrics (AP-style)
# ---------------------------------------------------------------------------

def reid_embedding_dispersion(
    tracks: Dict[Any, List[Dict[str, Any]]],
    max_samples: int = 3000,
) -> Dict[str, float]:
    embs = []
    for _, entries in tracks.items():
        for e in entries:
            emb = e.get("embedding", None)
            if emb is None:
                continue
            embs.append(np.asarray(emb, dtype=float))
            if len(embs) >= max_samples:
                break
        if len(embs) >= max_samples:
            break
    if len(embs) < 2:
        return {
            "embedding_dispersion_mean": float("nan"),
            "embedding_dispersion_std": float("nan"),
        }
    X = np.stack(embs, axis=0)
    n = X.shape[0]
    N_pairs = min(20000, n * (n - 1) // 2)
    idx_i = np.random.randint(0, n, size=N_pairs)
    idx_j = np.random.randint(0, n, size=N_pairs)
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    a = X[idx_i]
    b = X[idx_j]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    cos_sim = num / den
    cos_dist = 1.0 - cos_sim
    return {
        "embedding_dispersion_mean": float(np.mean(cos_dist)),
        "embedding_dispersion_std": float(np.std(cos_dist)),
    }


def reid_temporal_embedding_stability(
    tracks: Dict[Any, List[Dict[str, Any]]]
) -> Dict[str, float]:
    track_stds = []
    for _, entries in tracks.items():
        embs = [e.get("embedding", None) for e in entries]
        embs = [np.asarray(x, dtype=float) for x in embs if x is not None]
        if len(embs) < 2:
            continue
        X = np.stack(embs, axis=0)
        per_dim_std = np.std(X, axis=0)
        track_stds.append(float(np.mean(per_dim_std)))
    return {
        "mean_embedding_temporal_std": safe_mean(track_stds),
        "std_embedding_temporal_std": safe_std(track_stds),
    }


# ---------------------------------------------------------------------------
# 5. Cross-camera metrics (AP + RTMO global)
# ---------------------------------------------------------------------------

def build_global_tracks(global_path: Path) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Supports two shapes:

      A) flattened per-detection with global_id (RTMO-ish):
         {
           "global_id": int,
           "cam_id": "AC10",
           "frame" or "frame_idx": int,
           "bbox": [...],
           "kps"/"keypoints"/"pose"/"joints": [...],
           "feature"/"reid"/"embedding"/"embed": [...]
         }

      B) same as per-camera AP file, with:
         {
           "cam_id": "...",
           "frame": ...,
           "poses": [ { "global_id": ..., "bbox": ..., "keypoints"/"pose"/"joints"/"kps": ... }, ... ],
           "embeds": [ [...], ... ]
         }
    """
    global_tracks: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in load_jsonl(global_path):
        if "poses" in rec:
            # shape B (AP-style)
            cam_id = rec.get("cam_id")
            frame_idx = rec.get("frame") or rec.get("frame_idx")
            poses = rec.get("poses") or []
            embeds = rec.get("embeds") or []
            for i, person in enumerate(poses):
                gid = person.get("global_id") or person.get("gid")
                if gid is None:
                    continue
                emb = embeds[i] if i < len(embeds) else None
                kps = (
                    person.get("keypoints")
                    or person.get("pose")
                    or person.get("joints")
                    or person.get("kps")
                )
                bbox = person.get("bbox") or person.get("box")
                entry = {
                    "cam_id": cam_id,
                    "frame_idx": frame_idx,
                    "bbox": bbox,
                    "keypoints": kps,
                    "embedding": emb,
                }
                global_tracks[gid].append(entry)
        else:
            # shape A (flattened, likely RTMO-style)
            gid = rec.get("global_id") or rec.get("gid")
            if gid is None:
                continue
            cam_id = rec.get("cam_id")
            frame_idx = rec.get("frame") or rec.get("frame_idx")
            bbox = rec.get("bbox") or rec.get("box")
            kps = (
                rec.get("keypoints")
                or rec.get("pose")
                or rec.get("joints")
                or rec.get("kps")
            )
            emb = (
                rec.get("embedding")
                or rec.get("embed")
                or rec.get("feature")
                or rec.get("reid")
            )
            entry = {
                "cam_id": cam_id,
                "frame_idx": frame_idx,
                "bbox": bbox,
                "keypoints": kps,
                "embedding": emb,
            }
            global_tracks[gid].append(entry)
    return global_tracks


def crosscam_graph_connectivity(global_tracks: Dict[Any, List[Dict[str, Any]]]
                                ) -> Dict[str, float]:
    if not global_tracks:
        return {
            "global_id_count": 0.0,
            "graph_connectivity_ratio": float("nan"),
        }
    multi_cam = 0
    for _, entries in global_tracks.items():
        cams = {e.get("cam_id", None) for e in entries if e.get("cam_id", None) is not None}
        if len(cams) > 1:
            multi_cam += 1
    total = len(global_tracks)
    gcr = float(multi_cam / total) if total > 0 else float("nan")
    return {
        "global_id_count": float(total),
        "graph_connectivity_ratio": gcr,
    }


def crosscam_pairwise_embedding_consistency(
    global_tracks: Dict[Any, List[Dict[str, Any]]]
) -> Dict[str, float]:
    sims_per_gid = []
    for _, entries in global_tracks.items():
        per_cam = defaultdict(list)
        for e in entries:
            cam = e.get("cam_id", None)
            emb = e.get("embedding", None)
            if cam is None or emb is None:
                continue
            per_cam[cam].append(np.asarray(emb, dtype=float))
        cams = list(per_cam.keys())
        if len(cams) < 2:
            continue
        cam_mean = {}
        for c in cams:
            X = np.stack(per_cam[c], axis=0)
            cam_mean[c] = np.mean(X, axis=0)
        cs = []
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                a = cam_mean[cams[i]]
                b = cam_mean[cams[j]]
                num = float(np.dot(a, b))
                den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                cs.append(num / den)
        if cs:
            sims_per_gid.append(float(np.mean(cs)))
    return {
        "mean_crosscam_embedding_similarity": safe_mean(sims_per_gid),
        "std_crosscam_embedding_similarity": safe_std(sims_per_gid),
    }


def compute_crosscam_metrics(global_path: Path) -> Dict[str, Any]:
    if not global_path.exists():
        return {}
    global_tracks = build_global_tracks(global_path)
    if not global_tracks:
        return {}
    metrics: Dict[str, Any] = {}
    metrics.update(crosscam_graph_connectivity(global_tracks))
    metrics.update(crosscam_pairwise_embedding_consistency(global_tracks))
    return metrics


# ---------------------------------------------------------------------------
# 6. Pipeline health metrics
# ---------------------------------------------------------------------------

def pipeline_stage_yield(frame_records: List[Dict[str, Any]]) -> Dict[str, float]:
    total = 0
    with_pose = 0
    with_emb  = 0
    for fr in frame_records:
        for det in fr.get("detections", []):
            total += 1
            if det.get("keypoints", None) is not None:
                with_pose += 1
            if det.get("embedding", None) is not None:
                with_emb += 1
    if total == 0:
        return {
            "total_detections": 0.0,
            "yield_pose": float("nan"),
            "yield_embedding": float("nan"),
        }
    return {
        "total_detections": float(total),
        "yield_pose": float(with_pose / total),
        "yield_embedding": float(with_emb / total),
    }


# ---------------------------------------------------------------------------
# Orchestrator for one file (shared between RTMO & AP)
# ---------------------------------------------------------------------------

def compute_all_metrics_for_file(
    frame_records: List[Dict[str, Any]],
    mode: str,
    cam_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      per_track_df  (one row per track)
      limb_df       (one row per track-frame-bone limb length)
      per_video_row (dict of all metrics for this file/cam)
    """
    tracks = build_tracks_from_frame_records(frame_records)
    total_frames = len(frame_records)

    # ---- per-track RTMO-style metrics + id-switch proxy ----
    per_track_rows = []
    total_proxy_sw = 0
    total_proxy_chk = 0
    for tid, entries in tracks.items():
        row = compute_track_metrics(entries)
        sw, chk = compute_proxy_id_switches(entries)
        total_proxy_sw += sw
        total_proxy_chk += chk
        row.update({
            "track_id": tid,
            "cam_id": cam_id,
            "mode": mode,
        })
        per_track_rows.append(row)
    per_track_df = pd.DataFrame(per_track_rows)

    # ---- per-frame limb-length time series ----
    limb_df = compute_limb_length_timeseries(tracks)
    # file/mode/cam_id columns set by caller

    # ---- RTMO frame-level stats ----
    fl_stats = frame_level_stats(frame_records)
    det_count_std = float(np.nanstd(fl_stats["det_counts"])) if fl_stats["det_counts"].size else float("nan")
    conf_mean_overall = (
        float(np.nanmean(fl_stats["conf_means"]))
        if np.isfinite(fl_stats["conf_means"]).any()
        else float("nan")
    )
    dup_iou_mean_overall = (
        float(np.nanmean(fl_stats["dup_iou_means"]))
        if fl_stats["dup_iou_means"].size
        else float("nan")
    )

    # ---- AP detection / pose / tracking / ReID / health metrics ----
    metrics: Dict[str, Any] = {}
    metrics.update(detection_frame_count_stats(frame_records))
    metrics.update(detection_temporal_persistence(frame_records))
    metrics.update(detection_confidence_stats(frame_records))
    metrics.update(intraframe_dup_iou_stats(frame_records))

    if tracks:
        metrics.update(pose_confidence_stats(tracks))
        metrics.update(pose_temporal_smoothness(tracks))
        metrics.update(pose_limb_length_consistency(tracks))
        metrics.update(pose_self_similarity_over_time(tracks))
        metrics.update(tracking_basic_stats(tracks, total_frames=total_frames))
        metrics.update(tracking_bbox_drift(tracks))
        metrics.update(reid_embedding_dispersion(tracks))
        metrics.update(reid_temporal_embedding_stability(tracks))

    metrics.update(pipeline_stage_yield(frame_records))

    # ---- aggregate RTMO track metrics into per-video ----
    if not per_track_df.empty:
        metrics.update({
            "n_tracks": int(len(per_track_df)),
            "mean_track_len": float(per_track_df["track_len"].mean()),
            "id_fragments": float(per_track_df["fragments"].sum()),
            "iou_temporal_mean": float(per_track_df["iou_temporal_mean"].mean()),
            "kp_jitter_mean": float(per_track_df["kp_jitter"].mean()),
            "embed_var_mean": float(per_track_df["embed_var"].mean()),
            "angle_bad_rate_mean": float(per_track_df["angle_bad_rate"].mean()),
            "pqi": compute_pqi(per_track_df),
        })
    else:
        metrics.update({
            "n_tracks": 0,
            "mean_track_len": float("nan"),
            "id_fragments": 0.0,
            "iou_temporal_mean": float("nan"),
            "kp_jitter_mean": float("nan"),
            "embed_var_mean": float("nan"),
            "angle_bad_rate_mean": float("nan"),
            "pqi": float("nan"),
        })

    # RTMO-style frame stats
    metrics.update({
        "det_count_std": det_count_std,
        "conf_mean_rtmo": conf_mean_overall,
        "dup_iou_mean_rtmo": dup_iou_mean_overall,
        "proxy_id_switches": int(total_proxy_sw),
        "proxy_id_checks": int(total_proxy_chk),
    })

    # meta
    metrics.update({
        "mode": mode,
        "cam_id": cam_id,
        "n_frames": int(total_frames),
    })

    return per_track_df, limb_df, metrics


# ---------------------------------------------------------------------------
# Mode-specific drivers
# ---------------------------------------------------------------------------

def run_rtmo(root: str, global_file: str):
    base_dir = Path(root)
    files = []
    for pat in ["**/*_poses_union.jsonl"]:
        files.extend(glob.glob(str(base_dir / pat), recursive=True))
    files = [Path(f) for f in files if Path(f).is_file()]
    if not files:
        print(f"[WARN] [RTMO] No *_poses_union.jsonl under {base_dir}")
        return

    all_track_dfs = []
    all_video_rows = []
    all_limb_dfs = []

    for p in sorted(files):
        fname = p.name
        cam_id = fname.split("_")[0] if "_" in fname else p.parent.name.split("_")[0]
        print(f"[INFO] [RTMO] Evaluating {p}")
        frame_records = build_frame_records_rtmo(p)
        per_track_df, limb_df, metrics = compute_all_metrics_for_file(frame_records, mode="rtmo", cam_id=cam_id)
        rel_file = str(p.relative_to(base_dir))
        metrics.update({"file": rel_file})

        if not per_track_df.empty:
            per_track_df["file"] = rel_file
            per_track_df["mode"] = "rtmo"
            per_track_df["cam_id"] = cam_id
            all_track_dfs.append(per_track_df)

        if not limb_df.empty:
            limb_df["file"] = rel_file
            limb_df["mode"] = "rtmo"
            limb_df["cam_id"] = cam_id
            all_limb_dfs.append(limb_df)

        all_video_rows.append(metrics)

    outdir = base_dir
    outdir.mkdir(parents=True, exist_ok=True)

    per_track = pd.concat(all_track_dfs, ignore_index=True) if all_track_dfs else pd.DataFrame()
    per_video = pd.DataFrame(all_video_rows)
    per_limb = pd.concat(all_limb_dfs, ignore_index=True) if all_limb_dfs else pd.DataFrame()

    # overall = column-wise mean for numeric cols, sums for obvious counts
    if not per_video.empty:
        overall = {}
        num_cols = per_video.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col in {"n_frames", "total_detections"}:
                overall[col] = float(per_video[col].sum(skipna=True))
            elif col in {"n_tracks"}:
                overall[col] = float(per_video[col].sum(skipna=True))
            else:
                overall[col] = float(per_video[col].mean(skipna=True))
        overall["mode"] = "rtmo"
    else:
        overall = {"mode": "rtmo"}

    per_track.to_csv(outdir / "per_track.csv", index=False)
    per_video.to_csv(outdir / "per_video.csv", index=False)
    pd.DataFrame([overall]).to_csv(outdir / "overall.csv", index=False)
    per_limb.to_csv(outdir / "limb_lengths.csv", index=False)

    print(f"[OK] [RTMO] Wrote {outdir/'per_track.csv'}")
    print(f"[OK] [RTMO] Wrote {outdir/'per_video.csv'}")
    print(f"[OK] [RTMO] Wrote {outdir/'overall.csv'}")
    print(f"[OK] [RTMO] Wrote {outdir/'limb_lengths.csv'}")

    # cross-camera CSV for RTMO if global file present
    global_path = base_dir / global_file
    crosscam_metrics = compute_crosscam_metrics(global_path)
    if crosscam_metrics:
        pd.DataFrame([crosscam_metrics]).to_csv(outdir / "crosscam.csv", index=False)
        print(f"[OK] [RTMO] Wrote {outdir/'crosscam.csv'}")
    else:
        print(f"[WARN] [RTMO] No cross-camera metrics (global file missing or empty)")


def run_ap(root: str, global_file: str):
    out_root = Path(root)
    cam_files = sorted(out_root.rglob("*_poses_union.jsonl"))
    if not cam_files:
        print(f"[ERROR] [AP] No *_poses_union.jsonl under {out_root}")
        return

    print(f"[INFO] [AP] Found {len(cam_files)} camera files:")
    for cf in cam_files:
        print("   -", cf.relative_to(out_root))

    all_track_dfs = []
    all_video_rows = []
    all_limb_dfs = []

    for cf in cam_files:
        cam_name = cf.name.split("_")[0]
        print(f"[INFO] [AP] Evaluating {cf}")
        frame_records = build_frame_records_ap(cf, cam_name)
        per_track_df, limb_df, metrics = compute_all_metrics_for_file(frame_records, mode="ap", cam_id=cam_name)
        rel_file = str(cf.relative_to(out_root))
        metrics.update({"file": rel_file})

        if not per_track_df.empty:
            per_track_df["file"] = rel_file
            per_track_df["mode"] = "ap"
            per_track_df["cam_id"] = cam_name
            all_track_dfs.append(per_track_df)

        if not limb_df.empty:
            limb_df["file"] = rel_file
            limb_df["mode"] = "ap"
            limb_df["cam_id"] = cam_name
            all_limb_dfs.append(limb_df)

        all_video_rows.append(metrics)

    outdir = out_root
    outdir.mkdir(parents=True, exist_ok=True)

    per_track = pd.concat(all_track_dfs, ignore_index=True) if all_track_dfs else pd.DataFrame()
    per_video = pd.DataFrame(all_video_rows)
    per_limb = pd.concat(all_limb_dfs, ignore_index=True) if all_limb_dfs else pd.DataFrame()

    # overall
    if not per_video.empty:
        overall = {}
        num_cols = per_video.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col in {"n_frames", "total_detections"}:
                overall[col] = float(per_video[col].sum(skipna=True))
            elif col in {"n_tracks"}:
                overall[col] = float(per_video[col].sum(skipna=True))
            else:
                overall[col] = float(per_video[col].mean(skipna=True))
        overall["mode"] = "ap"
    else:
        overall = {"mode": "ap"}

    per_track.to_csv(outdir / "per_track.csv", index=False)
    per_video.to_csv(outdir / "per_video.csv", index=False)
    pd.DataFrame([overall]).to_csv(outdir / "overall.csv", index=False)
    per_limb.to_csv(outdir / "limb_lengths.csv", index=False)

    print(f"[OK] [AP] Wrote {outdir/'per_track.csv'}")
    print(f"[OK] [AP] Wrote {outdir/'per_video.csv'}")
    print(f"[OK] [AP] Wrote {outdir/'overall.csv'}")
    print(f"[OK] [AP] Wrote {outdir/'limb_lengths.csv'}")

    # cross-camera CSV (if global file present)
    global_path = out_root / global_file
    crosscam_metrics = compute_crosscam_metrics(global_path)
    if crosscam_metrics:
        pd.DataFrame([crosscam_metrics]).to_csv(outdir / "crosscam.csv", index=False)
        print(f"[OK] [AP] Wrote {outdir/'crosscam.csv'}")
    else:
        print(f"[WARN] [AP] No cross-camera metrics (global file missing or empty)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified unsupervised metrics (RTMO + DeepSORT / AlphaPose + OSNet)."
    )
    parser.add_argument(
        "--mode",
        choices=["rtmo", "ap"],
        default="rtmo",
        help="Which pipeline outputs to evaluate (default: rtmo).",
    )
    parser.add_argument(
        "--rtmo-root",
        type=str,
        default=DEFAULT_RTMO_ROOT,
        help="Root dir for RTMO+DeepSORT outputs.",
    )
    parser.add_argument(
        "--ap-root",
        type=str,
        default=DEFAULT_AP_ROOT,
        help="Root dir for AlphaPose+OSNet outputs.",
    )
    parser.add_argument(
        "--global-file",
        type=str,
        default=DEFAULT_GLOBAL_FILE,
        help="Global tracks JSONL filename for both modes.",
    )
    args = parser.parse_args()

    if args.mode == "rtmo":
        run_rtmo(args.rtmo_root, args.global_file)
    else:
        run_ap(args.ap_root, args.global_file)


if __name__ == "__main__":
    main()
