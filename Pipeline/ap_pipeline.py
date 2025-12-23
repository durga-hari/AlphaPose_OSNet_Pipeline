#!/usr/bin/env python3
from __future__ import annotations
import logging, json, gc, time, colorsys, random, os, csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

from io_utils import VideoReader, VideoWriter, JsonlWriter
from common import ensure_dir, safe_stem
from detectors import build_detector
from pose import build_pose_estimator, AlphaPoseCfg
from reid import build_reid
from tracking import build_tracker
from visualizer import draw_bbox_and_id, draw_skeleton
from crosscam_stitch import stitch


# ============================================================
#   Utility: color mapping per unique ID
# ============================================================
def color_for_id(id_num: int) -> tuple[int, int, int]:
    random.seed(int(id_num) * 99991)
    hue = (int(id_num) * 37) % 360 / 360.0
    sat, val = 0.8, 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


# ============================================================
#   Visualization: annotate videos with local vs global IDs
# ============================================================
def annotate_videos_with_ids(out_root: Path, max_frames: int = None):
    """
    Create two annotated videos per camera folder: local IDs and global IDs.
    Automatically aligns frame counts using 'max_frames' from pipeline config.
    Handles dropped or mismatched frames safely.
    """
    import colorsys, cv2, json, random
    from tqdm import tqdm

    # --------------------------
    # Color generator
    # --------------------------
    def color_for_id(id_num: int, mode="local"):
        seed_offset = 1 if mode == "local" else 99991
        random.seed(int(id_num) * seed_offset)
        hue = ((int(id_num) * 37) % 360) / 360.0
        sat, val = (0.9, 0.9) if mode == "local" else (0.8, 0.8)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return int(b * 255), int(g * 255), int(r * 255)

    # --------------------------
    # Load JSONL to dict
    # --------------------------
    def load_jsonl(path):
        frame_dict = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                frame = int(rec.get("frame", 0))
                poses = rec.get("poses", [])
                frame_dict.setdefault(frame, []).extend(poses)
        return frame_dict

    def annotate(video_path: Path, anno_dict: dict, out_path: Path, mode="local"):
        """Draw either local or global IDs and ensure equal frame counts for both videos."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Respect the pipeline's max_frames setting (None = no limit)
        if max_frames is not None and max_frames > 0:
            total = min(total, max_frames)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        if len(anno_dict) == 0:
            print(f"⚠️ No annotations found in {video_path.stem} ({mode})")
            # Still create an empty video with the same frame count for consistency
            for _ in range(total):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            cap.release()
            out.release()
            print(f"✅ (empty) Saved → {out_path}")
            return

        fmin, fmax = min(anno_dict), max(anno_dict)
        print(f"Annotating {video_path.name} ({mode}): {len(anno_dict)} annotated frames "
              f"[{fmin}-{fmax}] → writing {total} total frames")

        for frame_idx in tqdm(range(total), desc=f"{video_path.stem} ({mode})"):
            ret, frame = cap.read()
            if not ret:
                break

            # Handle ±1 mismatch in annotation frame indices
            poses = (anno_dict.get(frame_idx)
                     or anno_dict.get(frame_idx + 1)
                     or anno_dict.get(frame_idx - 1)
                     or [])

            # Always write frame, even if there are no poses
            for p in poses:
                id_val = p.get("id") if mode == "local" else p.get("global_id")
                bbox = p.get("bbox")
                if id_val is None or bbox is None:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                color = color_for_id(int(id_val), mode)
                label = f"{'LID' if mode == 'local' else 'GID'} {id_val}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out.write(frame)

        cap.release()
        out.release()
        print(f" Saved → {out_path}")

    # --------------------------
    # Iterate through all cameras
    # --------------------------
    for cam_dir in sorted(out_root.glob("*")):
        if not cam_dir.is_dir():
            continue

        # Always use the raw session video as input, not any annotated outputs
        src_dir = Path("/home/arun_remote/DaRA_Thesis/Session_4")
        video = src_dir / f"{cam_dir.name}.mp4"
        if not video.exists():
            print(f" Skipping {cam_dir.name}: cannot find raw video {video}")
            continue

        json_local = next(cam_dir.glob("*_poses.jsonl"), None)
        json_global = next(cam_dir.glob("*_poses_union.jsonl"), None)

        if not (video and json_local and json_global):
            print(f" Skipping {cam_dir.name}: missing video or JSONL.")
            continue

        local_dict = load_jsonl(json_local)
        global_dict = load_jsonl(json_global)

        out_local = cam_dir / f"{cam_dir.name}_localids.mp4"
        out_global = cam_dir / f"{cam_dir.name}_globalids.mp4"

        annotate(video, local_dict, out_local, mode="local")
        annotate(video, global_dict, out_global, mode="global")


def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [ndarray_to_list(o) for o in obj]
    return obj


# ============================================================
#   SequentialPipeline (AlphaPose + ReID + Tracker)
# ============================================================
class SequentialPipeline:
    def __init__(
        self,
        detector_name: str,
        detector_args: dict,
        pose_name: str,
        pose_args: dict,
        reid_name: str,
        reid_args: dict,
        tracker_name: str = "none",
        tracker_args: dict | None = None,
        draw: bool = True,
        kpt_thresh: float = 0.2,
        dataset_hint: str = "auto",
        debug_heatmaps: bool = False,
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.detector = build_detector(detector_name, **detector_args)
        self.pose = build_pose_estimator(pose_name, **pose_args)
        self.reid = build_reid(reid_name, **(reid_args or {}))
        self.tracker = build_tracker(tracker_name, **(tracker_args or {}))
        self.draw = draw
        self.kpt_thresh = kpt_thresh
        self.dataset_hint = dataset_hint.lower()
        self.debug_heatmaps = debug_heatmaps

    # --------------------------------------------------------
    def run(
        self,
        video_path: str | Path,
        out_dir: str | Path,
        save_video: bool,
        save_json: bool,
        save_embeds: bool,
        max_frames: Optional[int] = None,
    ):
        video_path = Path(video_path)
        out_dir = Path(out_dir)
        ensure_dir(out_dir)

        rdr = VideoReader(video_path)
        base = safe_stem(video_path)

        vw = VideoWriter(out_dir, rdr, enabled=save_video)
        jw = JsonlWriter(out_dir / f"{base}_poses.jsonl") if save_json else None

        frame_count = 0
        try:
            for idx, frame_bgr in rdr:
                frame_count += 1
                if max_frames and frame_count > max_frames:
                    break

                boxes_xyxy = self.detector(frame_bgr)
                if boxes_xyxy is None or getattr(boxes_xyxy, "size", 0) == 0:
                    vw.write(frame_bgr)
                    continue

                kpts_list, pose_scores, _ = self.pose(frame_bgr, boxes_xyxy)

                if not kpts_list:
                    vw.write(frame_bgr)
                    continue

                embeds = None
                if hasattr(self.reid, "is_ready") and self.reid.is_ready():
                    emb_list = self.reid(frame_bgr, boxes_xyxy)
                    try:
                        embeds = np.vstack([
                            (e.detach().cpu().numpy().reshape(-1)
                             if hasattr(e, "detach")
                             else np.asarray(e, dtype=np.float32).reshape(-1))
                            for e in emb_list
                        ])
                        embeds /= (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12)
                    except Exception:
                        embeds = None

                # ---- Tracking: StrongSORT vs DeepSORT ----
                if self.tracker:
                    # DeepSORT wrapper exposes update_from_pipeline(boxes, feats)
                    if hasattr(self.tracker, "update_from_pipeline"):
                        ids = self.tracker.update_from_pipeline(boxes_xyxy, embeds)
                    else:
                        # StrongSORT-style: update(boxes, feats) -> ids
                        ids = self.tracker.update(boxes_xyxy, embeds)
                else:
                    ids = list(range(1, boxes_xyxy.shape[0] + 1))

                annotated = frame_bgr.copy()
                if self.draw:
                    for i, box in enumerate(boxes_xyxy):
                        pid = int(ids[i])
                        color = color_for_id(pid)
                        score_i = float(pose_scores[i]) if pose_scores is not None else None
                        draw_bbox_and_id(annotated, box, track_id=pid, score=score_i, color=color)
                        draw_skeleton(
                            annotated,
                            np.asarray(kpts_list[i], dtype=np.float32),
                            kpt_thresh=self.kpt_thresh,
                            dataset=self.dataset_hint,
                            draw_face=False,
                            draw_hands=False,
                            bbox=box,
                            track_id=pid,
                            color=color,
                        )

                vw.write(annotated)

                if jw:
                    rec = {
                        "frame": idx,
                        "cam_id": base,
                        "poses": [
                            {
                                "id": int(ids[i]),
                                "bbox": boxes_xyxy[i].astype(float).tolist(),
                                "keypoints": ndarray_to_list(kpts_list[i]),
                                "score": float(pose_scores[i]) if pose_scores is not None else None,
                            }
                            for i in range(len(kpts_list))
                        ],
                    }
                    if save_embeds and embeds is not None:
                        rec["embeds"] = embeds.tolist()
                    jw.write(rec)
        finally:
            rdr.close()
            vw.close()
            if jw:
                jw.close()
            self.log.info(f"Finished {video_path} | frames={frame_count}")


# ============================================================
#   Union-based Cross-camera Global ID Matching
# ============================================================
def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / (n + eps) if np.isfinite(n) and n > eps else np.zeros_like(v)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.dot(l2_normalize(a, eps), l2_normalize(b, eps)))


class UnionFind:
    def __init__(self): self.p = {}
    def find(self, x):
        if x not in self.p: self.p[x] = x
        if self.p[x] != x: self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


def build_track_centroids(cam_dir: Path, min_samples: int = 3) -> Dict[int, np.ndarray]:
    poses_path = next(cam_dir.glob("*_poses.jsonl"), None)
    if not poses_path: return {}
    sums, counts = {}, {}
    for line in open(poses_path):
        rec = json.loads(line)
        poses, emb_list = rec.get("poses", []), rec.get("embeds")
        if isinstance(emb_list, list) and len(emb_list) == len(poses):
            for p, e in zip(poses, emb_list):
                tid = int(p.get("id", -1))
                if tid < 0: continue
                v = l2_normalize(np.asarray(e, np.float32))
                sums[tid] = sums.get(tid, 0) + v
                counts[tid] = counts.get(tid, 0) + 1
    centroids = {}
    for tid, s in sums.items():
        c = counts.get(tid, 0)
        if c >= min_samples:
            m = l2_normalize(s / c)
            centroids[tid] = m
    return centroids


# ============================================================
#   Enhanced Union-based Global ID Matching (with Pose Similarity)
# ============================================================
def normalize_keypoints(kps: np.ndarray) -> np.ndarray:
    """Normalize keypoints to mean 0 / unit variance (ignoring zero points)."""
    kps = np.asarray(kps, np.float32)
    if kps.ndim == 3:
        kps = kps[..., :2]
    valid = np.all(kps > 0, axis=-1)
    if not np.any(valid):
        return np.zeros_like(kps)
    mean = np.mean(kps[valid], axis=0)
    std = np.std(kps[valid], axis=0).mean() + 1e-6
    return (kps - mean) / std


def pose_similarity(kpsA: np.ndarray, kpsB: np.ndarray) -> float:
    """Compute structural similarity between two normalized poses."""
    if kpsA is None or kpsB is None:
        return 0.0
    kpsA, kpsB = normalize_keypoints(kpsA), normalize_keypoints(kpsB)
    if kpsA.shape != kpsB.shape:
        J = min(kpsA.shape[0], kpsB.shape[0])
        kpsA, kpsB = kpsA[:J], kpsB[:J]
    dist = np.linalg.norm(kpsA - kpsB, axis=-1).mean()
    return float(np.exp(-dist))


def hybrid_similarity(embA: np.ndarray, embB: np.ndarray,
                      kpsA: np.ndarray, kpsB: np.ndarray,
                      alpha: float = 0.6) -> float:
    """Weighted blend of embedding cosine and pose similarity."""
    emb_sim = cosine(embA, embB)
    pose_sim = pose_similarity(kpsA, kpsB)
    return alpha * emb_sim + (1 - alpha) * pose_sim


def build_track_centroids_with_pose(cam_dir: Path, min_samples: int = 35):
    poses_path = next(cam_dir.glob("*_poses.jsonl"), None)
    if not poses_path:
        return {}

    tracks = defaultdict(lambda: {"embs": [], "kps": [], "frames": []})

    with poses_path.open("r") as f:
        for line in f:
            rec = json.loads(line)
            frame_idx = rec.get("frame", None)
            poses, emb_list = rec.get("poses", []), rec.get("embeds")

            if not isinstance(emb_list, list) or len(emb_list) != len(poses):
                continue

            for p, e in zip(poses, emb_list):
                tid = int(p.get("id", -1))
                if tid < 0:
                    continue
                v = l2_normalize(np.asarray(e, np.float32))
                tracks[tid]["embs"].append(v)
                tracks[tid]["kps"].append(np.asarray(p.get("keypoints"), np.float32))
                if frame_idx is not None:
                    tracks[tid]["frames"].append(int(frame_idx))

    centroids = {}
    for tid, data in tracks.items():
        if len(data["embs"]) < min_samples:
            continue

        embs = np.stack(data["embs"])
        kps = np.stack(data["kps"])
        mean_emb = l2_normalize(embs.mean(axis=0))
        mean_kps = kps.mean(axis=0)

        if data["frames"]:
            start = min(data["frames"])
            end = max(data["frames"])
        else:
            start = 0
            end = 0

        centroids[tid] = {
            "emb": mean_emb,
            "pose": mean_kps,
            "start": start,
            "end": end,
        }

    return centroids


def merge_intra_camera_tracks(out_root: Path, sim_thr: float = 0.80):
    """
    Merge fragmented local tracks within each camera using embedding similarity.
    Updates each *_poses.jsonl in-place (with new merged IDs).
    """
    MIN_TRACK_LEN = 35
    logging.info(f"[IntraCam] Merging local tracks within each camera (thr={sim_thr})")
    for cam_dir in sorted(out_root.glob("*")):
        if not cam_dir.is_dir():
            continue
        poses_path = next(cam_dir.glob("*_poses.jsonl"), None)
        if not poses_path:
            continue

        # Build centroids and track lengths
        raw_vecs = {}
        for line in open(poses_path):
            rec = json.loads(line)
            poses, emb_list = rec.get("poses", []), rec.get("embeds")
            if isinstance(emb_list, list) and len(emb_list) == len(poses):
                for p, e in zip(poses, emb_list):
                    tid = int(p.get("id", -1))
                    if tid < 0:
                        continue
                    v = l2_normalize(np.asarray(e, np.float32))
                    raw_vecs.setdefault(tid, []).append(v)

        # Drop very short tracks first
        centroids = {}
        for tid, vecs in raw_vecs.items():
            if len(vecs) < MIN_TRACK_LEN:
                continue
            arr = np.stack(vecs)
            centroids[tid] = l2_normalize(arr.mean(axis=0))

        tids = list(centroids.keys())
        if len(tids) <= 1:
            logging.info(f"[IntraCam] {cam_dir.name}: not enough valid tracks after filtering.")
            continue

        # Compute pairwise cosine matrix
        mat = np.zeros((len(tids), len(tids)), np.float32)
        for i, t1 in enumerate(tids):
            for j, t2 in enumerate(tids):
                if i < j:
                    mat[i, j] = cosine(centroids[t1], centroids[t2])
                    mat[j, i] = mat[i, j]

        # Union similar tracks
        uf = UnionFind()
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                if mat[i, j] >= sim_thr:
                    uf.union(tids[i], tids[j])

        # Build new ID map
        id_map = {}
        next_id = 1
        for t in tids:
            root = uf.find(t)
            if root not in id_map:
                id_map[root] = next_id
                next_id += 1
        remap = {t: id_map[uf.find(t)] for t in tids}

        # Rewrite poses.jsonl
        tmp_path = poses_path.with_name(poses_path.stem + "_intra.jsonl")
        with open(poses_path) as fr, open(tmp_path, "w") as fw:
            for line in fr:
                rec = json.loads(line)
                for p in rec.get("poses", []):
                    tid = int(p.get("id", -1))
                    if tid in remap:
                        p["id"] = remap[tid]
                fw.write(json.dumps(rec) + "\n")
        os.replace(tmp_path, poses_path)
        logging.info(f"[IntraCam] {cam_dir.name}: merged → {len(set(remap.values()))} local IDs")


def run_crosscam_union_stitch(out_root: Path,
                              cos_thr: float = 0.45,
                              alpha: float = 0.50,
                              time_slack: int = 30):
    """
    Union-based global ID stitching with hybrid (embedding + pose) similarity.
    cos_thr – final blended similarity threshold (lower for more merges)
    alpha – weight for ReID embedding vs pose similarity
    """
    cams = sorted([d.name for d in out_root.iterdir() if d.is_dir()])
    per_cam = {cam: build_track_centroids_with_pose(out_root / cam)
               for cam in cams}

    uf = UnionFind()
    key = lambda cam, tid: f"{cam}#{tid}"

    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            ci, cj = cams[i], cams[j]
            ti, tj = per_cam.get(ci, {}), per_cam.get(cj, {})
            if not ti or not tj:
                continue
            tids_i, tids_j = list(ti.keys()), list(tj.keys())
            mat = np.zeros((len(tids_i), len(tids_j)), np.float32)

            # Compute hybrid similarity matrix
            for r, tid_i in enumerate(tids_i):
                ti_rec = ti[tid_i]
                vi, pi = ti_rec["emb"], ti_rec["pose"]
                si_start, si_end = ti_rec.get("start", 0), ti_rec.get("end", 0)

                for c, tid_j in enumerate(tids_j):
                    tj_rec = tj[tid_j]
                    vj, pj = tj_rec["emb"], tj_rec["pose"]
                    sj_start, sj_end = tj_rec.get("start", 0), tj_rec.get("end", 0)

                    # --- temporal gating: skip pairs that never overlap in time ---
                    latest_start = max(si_start, sj_start)
                    earliest_end = min(si_end, sj_end)
                    if latest_start > earliest_end + time_slack:
                        # no temporal overlap even with slack → cannot be same person
                        mat[r, c] = -1.0
                        continue

                    # --- hybrid similarity (appearance + pose) ---
                    mat[r, c] = hybrid_similarity(vi, vj, pi, pj, alpha=alpha)

            best_j = np.argmax(mat, 1)
            best_i = np.argmax(mat, 0)
            for r, c in enumerate(best_j):
                if best_i[c] == r and mat[r, c] >= cos_thr:
                    uf.union(key(ci, tids_i[r]), key(cj, tids_j[c]))

    comps, gid_map, next_gid = defaultdict(list), {cam: {} for cam in cams}, 1
    for cam in cams:
        for tid in per_cam.get(cam, {}):
            root = uf.find(key(cam, tid))
            comps[root].append((cam, tid))
    for members in comps.values():
        gid = next_gid
        next_gid += 1
        for cam, tid in members:
            gid_map[cam][tid] = gid

    with open(out_root / "crosscam_map_union.json", "w") as f:
        json.dump(gid_map, f, indent=2)
    with open(out_root / "crosscam_summary_union.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_id", "members"])
        for gid, members in enumerate(comps.values(), 1):
            w.writerow([gid, "; ".join(f"{c}:{t}" for c, t in members)])

    logging.info(f"[Union] Hybrid similarity (α={alpha}, thr={cos_thr}) → "
                 f"{len(comps)} global IDs written.")
    return gid_map


# ============================================================
#   Helper: Inject global IDs and report counts
# ============================================================
def inject_global_ids(out_root: Path, crosscam_map: dict, suffix="_poses_union.jsonl"):
    for cam_dir in sorted(out_root.glob("*")):
        if not cam_dir.is_dir():
            continue
        jsonl_file = next(cam_dir.glob("*_poses.jsonl"), None)
        if not jsonl_file:
            continue
        cam_name = cam_dir.name
        local2global = crosscam_map.get(cam_name, {})
        out_path = cam_dir / f"{cam_name}{suffix}"
        with open(jsonl_file, "r") as fr, open(out_path, "w") as fw:
            for line in fr:
                item = json.loads(line)
                for p in item.get("poses", []):
                    lid = int(p.get("id", -1))
                    gid = local2global.get(lid)
                    if gid:
                        p["global_id"] = gid
                fw.write(json.dumps(item) + "\n")
        logging.info(f"[Union] Injected globals → {out_path.name}")


def merge_tracks_ap(out_root: Path, crosscam_map: Dict[str, Dict[int, int]], out_path: Path):
    """
    Merge AlphaPose per-camera JSONLs into a unified file.
    Prefers *_poses_union.jsonl (after global injection), else falls back to *_poses.jsonl.
    crosscam_map must be {camera -> {local_id (int) -> global_id (int)}}.
    """
    out_root = Path(out_root)
    merged_records: List[dict] = []

    # Prefer union files produced by inject_global_ids(..., suffix="_poses_union.jsonl")
    jsonl_files = sorted(out_root.glob("*/*_poses_union.jsonl"))
    if not jsonl_files:
        jsonl_files = sorted(out_root.glob("*/*_poses.jsonl"))

    if not jsonl_files:
        logging.error(f"[Merge] No *_poses*.jsonl found under {out_root}")
        return

    for jf in jsonl_files:
        cam = jf.parent.name
        # Normalize map keys to int (in case a JSON import made them strings)
        raw_map = crosscam_map.get(cam, {}) or {}
        local_to_global = {int(k): int(v) for k, v in raw_map.items()}

        with open(jf, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                frame = item.get("frame")
                poses = item.get("poses", [])

                for p in poses:
                    lid = p.get("id", None)
                    if lid is None:
                        continue
                    try:
                        lid_int = int(lid)
                    except Exception:
                        continue

                    gid = local_to_global.get(lid_int)
                    # If union-injected file already has global_id, prefer that; else map it
                    gid = p.get("global_id", gid)
                    if gid is None:
                        continue

                    merged_records.append({
                        "cam_id": cam,          # was "camera"
                        "frame_idx": frame,     # was "frame"
                        "local_id": lid_int,
                        "global_id": int(gid),
                        "bbox": p.get("bbox"),
                        "keypoints": p.get("keypoints"),
                        "score": p.get("score"),
                    })

    if not merged_records:
        logging.warning("[Merge] No valid records to merge after reading pose JSONLs.")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fw:
        for r in merged_records:
            fw.write(json.dumps(r) + "\n")
    logging.info(f"[Merge] merged_global_tracks written → {out_path} (records={len(merged_records)})")


def count_global_individuals(merged_path: Path):
    seen = set()
    with open(merged_path) as f:
        for line in f:
            gid = json.loads(line).get("global_id")
            if gid:
                seen.add(gid)
    logging.info(f"✅ Total unique individuals across all cameras: {len(seen)}")
    return len(seen)


# ============================================================
#   Config + Main
# ============================================================
def load_config(config_path: str | Path) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("ap_pipeline_config.yaml")
    video_path = Path(cfg["video_path"])
    out_root = Path(cfg["out_dir"])

    detector_cfg = cfg["detector"]
    pose_cfg = cfg["pose"]
    reid_cfg = cfg["reid"]
    tracker_cfg = cfg.get("tracker", {})
    pipe_cfg = cfg.get("pipeline", {})

    # --- Build modules ---
    detector_args = dict(
        model=detector_cfg["model"],
        weights=detector_cfg["weights"],
        device=detector_cfg.get("device", "cuda"),
        conf=float(detector_cfg.get("conf", 0.5)),
        iou=float(detector_cfg.get("iou", 0.6)),
        imgsz=int(detector_cfg.get("imgsz", 640)),
        half=bool(detector_cfg.get("half", False)),
    )
    pose_args = dict(
        config=AlphaPoseCfg(
            input_size=tuple(pose_cfg.get("input_size", [256, 192])),
            device=pose_cfg.get("device", "cuda"),
            cfg_yaml=pose_cfg["cfg"],
            checkpoint=pose_cfg["ckpt"],
            fp16=pose_cfg.get("fp16", False),
        )
    )
    reid_args = (
        dict(weights=reid_cfg["weights"], device=reid_cfg.get("device", "cuda"))
        if reid_cfg["type"] != "none"
        else {}
    )

    # tracker args: StrongSORT vs DeepSORT
    tracker_type = tracker_cfg.get("type", "none").lower()
    if tracker_type == "strongsort":
        tracker_args = dict(
            sim_thr=float(tracker_cfg.get("sim_thr", 0.55)),
            iou_thr=float(tracker_cfg.get("iou_thr", 0.6)),
            ttl=int(tracker_cfg.get("ttl", 80)),
        )
    elif tracker_type == "deepsort":
        tracker_args = dict(
            max_cosine_distance=float(tracker_cfg.get("max_cosine_distance", 0.2)),
            max_iou_distance=float(tracker_cfg.get("max_iou_distance", 0.7)),
            max_age=int(tracker_cfg.get("max_age", 30)),
            n_init=int(tracker_cfg.get("n_init", 3)),
        )
    else:
        tracker_args = {}

    pipe = SequentialPipeline(
        detector_name=detector_cfg["type"],
        detector_args=detector_args,
        pose_name=pose_cfg["type"],
        pose_args=pose_args,
        reid_name=reid_cfg["type"],
        reid_args=reid_args,
        tracker_name=tracker_cfg.get("type", "none"),
        tracker_args=tracker_args,
        draw=bool(pipe_cfg.get("draw", True)),
        kpt_thresh=float(pipe_cfg.get("kpt_thresh", 0.2)),
        dataset_hint=str(pipe_cfg.get("kp_set", "coco_wholebody")).lower(),
        debug_heatmaps=bool(pipe_cfg.get("debug_heatmaps", False)),
    )

    # --- Run per-camera inference ---
    video_list = sorted(video_path.glob("*.mp4")) if video_path.is_dir() else [video_path]
    for v in video_list:
        logging.info(f"Processing video: {v.name}")
        out_dir = out_root / v.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            pipe.run(
                video_path=v,
                out_dir=out_dir,
                save_video=bool(pipe_cfg.get("save_video", True)),
                save_json=bool(pipe_cfg.get("save_json", True)),
                save_embeds=bool(pipe_cfg.get("save_embeds", True)),
                max_frames=(None if int(pipe_cfg.get("max_frames", 0)) == 0 else int(pipe_cfg.get("max_frames"))),
            )
        except Exception as e:
            logging.exception(f"Error processing {v.name}: {e}")

    cv2.destroyAllWindows()
    gc.collect()
    time.sleep(1.0)

    merge_intra_camera_tracks(out_root, sim_thr=0.75)

    # --- Stage 1: greedy deterministic crosscam stitch ---
    logging.info("Running initial deterministic stitching...")
    crosscam_map = stitch(out_root, sim_thr=0.6, time_win=300)

    # --- Stage 2: mutual-best Union-Find refinement ---
    logging.info("Running union-based global ID refinement...")
    crosscam_union_map = run_crosscam_union_stitch(
        out_root,
        cos_thr=0.55,   # was 0.5 → slightly easier to merge true same-person tracks
        alpha=0.60,
        time_slack=30,  # give pose a bit more weight vs pure ReID emb
    )

    # Inject and merge results
    inject_global_ids(out_root, crosscam_union_map, suffix="_poses_union.jsonl")
    merge_tracks_ap(out_root, crosscam_union_map, out_root / "merged_global_tracks_union.jsonl")
    merged = out_root / "merged_global_tracks_union.jsonl"
    if merged.exists():
        count_global_individuals(merged)

    max_frames_cfg = int(pipe_cfg.get("max_frames", 0))
    max_frames_final = None if max_frames_cfg <= 0 else max_frames_cfg
    annotate_videos_with_ids(out_root, max_frames=max_frames_final)

    logging.info(" Full AlphaPose + Union Global-ID pipeline complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    main()
