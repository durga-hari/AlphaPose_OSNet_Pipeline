from __future__ import annotations
import logging, json, gc, time, colorsys, random, os
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from io_utils import VideoReader, VideoWriter, JsonlWriter
from common import ensure_dir, safe_stem
from detectors import build_detector
from pose import build_pose_estimator, AlphaPoseCfg
from reid import build_reid
from tracking import build_tracker
from visualizer import draw_bbox_and_id, draw_skeleton, visualize_heatmap

# --- cross-camera utilities (we import, but will override merge function locally) ---
from crosscam_stitch import stitch, build_track_embeds
from crosscam_merge import load_crosscam_map as _ignored_load_crosscam_map   # not used
from crosscam_merge import merge_tracks as _ignored_merge_tracks              # not used


# ============================================================
#   Utility: color mapping per unique ID
# ============================================================
def color_for_id(id_num: int) -> tuple[int, int, int]:
    random.seed(int(id_num) * 99991)
    hue = (int(id_num) * 37) % 360 / 360.0
    sat, val = 0.8, 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [ndarray_to_list(o) for o in obj]
    return obj


# ============================================================
#   SequentialPipeline (AlphaPose + OSNet + Tracker)
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

        if self.debug_heatmaps:
            (out_dir / "debug_heatmaps").mkdir(exist_ok=True)

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

                kpts_list, pose_scores, heatmaps = self.pose(frame_bgr, boxes_xyxy)
                if not kpts_list:
                    vw.write(frame_bgr)
                    continue

                embeds = None
                if hasattr(self.reid, "is_ready") and self.reid.is_ready():
                    emb_list = self.reid(frame_bgr, boxes_xyxy)
                    try:
                        # -> convert tensors safely from GPU to CPU numpy
                        embeds = np.vstack(
                            [
                                (e.detach().cpu().numpy().reshape(-1)
                                 if hasattr(e, "detach") else
                                 (np.asarray(e, dtype=np.float32).reshape(-1) if e is not None else np.zeros((1,), np.float32)))
                                for e in emb_list
                            ]
                        )
                    except Exception:
                        embeds = None

                ids = (
                    self.tracker.update(boxes_xyxy, embeds)
                    if self.tracker
                    else list(range(1, boxes_xyxy.shape[0] + 1))
                )

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
                        rec["embeds"] = embeds.tolist()  # keep "embeds" (your files use this)
                    jw.write(rec)

        finally:
            rdr.close()
            vw.close()
            if jw:
                jw.close()
            self.log.info(f"Finished {video_path} | frames={frame_count}")


# ============================================================
#   Helper: Inject global IDs into JSONLs
# ============================================================
def inject_global_ids(out_root: Path, crosscam_map: dict):
    logging.info("Injecting global IDs into per-camera jsonl files...")
    for cam_dir in sorted(out_root.glob("*")):
        if not cam_dir.is_dir():
            continue
        jsonl_file = next(cam_dir.glob("*_poses.jsonl"), None)
        if not jsonl_file:
            continue
        cam_name = cam_dir.name
        local2global = crosscam_map.get(cam_name, {})
        out_path = jsonl_file.with_name(jsonl_file.stem + "_global.jsonl")
        with open(jsonl_file, "r") as fr, open(out_path, "w") as fw:
            for line in tqdm(fr, desc=f"{cam_name}"):
                item = json.loads(line)
                for p in item.get("poses", []):
                    lid = str(p.get("id"))
                    gid = local2global.get(lid)
                    if gid:
                        p["global_id"] = gid
                fw.write(json.dumps(item) + "\n")
        logging.info(f"  → {out_path.name} written.")


# ============================================================
#   Cross-camera stitching wrapper
# ============================================================
def run_crosscam_stitch(out_root: Path, sim_thr=0.55, time_win=900):
    logging.info(f"[CrossCam] Running stitching on {out_root}")
    try:
        mapping = stitch(out_root, sim_thr=sim_thr, time_win=time_win)
    except Exception as e:
        logging.exception(f"CrossCam Stitch failed: {e}")
        return None

    if not mapping:
        logging.warning("[CrossCam] No mapping returned — check embeddings.")
        return None

    json_out = out_root / "crosscam_map.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    rows = [{"camera": cam, "local_id": lid, "global_id": gid}
            for cam, ids in mapping.items() for lid, gid in ids.items()]
    csv_out = out_root / "crosscam_summary.csv"
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    logging.info(f"[CrossCam] crosscam_map.json and crosscam_summary.csv written.")
    return mapping


# ============================================================
#   Local merge compatible with *_poses.jsonl  (overrides import)
# ============================================================
def merge_tracks(out_root: Path, crosscam_map: Dict[str, Dict[str, int]], out_path: Path):
    """
    Merge per-camera *_poses.jsonl using crosscam_map into a global timeline.
    Writes merged_global_tracks.jsonl with fields:
      camera, frame, local_id, global_id, bbox, keypoints, score
    """
    out_root = Path(out_root)
    merged_records: List[dict] = []

    jsonl_files = sorted(out_root.glob("*/*_poses.jsonl"))
    if not jsonl_files:
        logging.error(f"[Merge] No *_poses.jsonl found under {out_root}")
        return

    for jf in jsonl_files:
        cam = jf.parent.name
        l2g = crosscam_map.get(cam, {})
        with open(jf, "r") as f:
            for line in f:
                item = json.loads(line)
                frame = item.get("frame")
                poses = item.get("poses", [])
                for p in poses:
                    lid = str(p.get("id"))
                    gid = l2g.get(lid)
                    if gid is None:
                        continue
                    merged_records.append({
                        "camera": cam,
                        "frame": frame,
                        "local_id": int(lid),
                        "global_id": int(gid),
                        "bbox": p.get("bbox"),
                        "keypoints": p.get("keypoints"),
                        "score": p.get("score"),
                        # No per-record embedding here; consolidation will fallback
                    })

    if not merged_records:
        logging.warning("[Merge] No valid records to merge.")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fw:
        for r in merged_records:
            fw.write(json.dumps(r) + "\n")
    logging.info(f"[Merge] merged_global_tracks.jsonl written → {out_path}")


# ============================================================
#   Adaptive Global-ID Consolidation
# ============================================================
def _fallback_gid_means_from_percamera(out_root: Path, crosscam_map: dict) -> Dict[int, np.ndarray]:
    """
    Build mean embedding per global_id by aggregating per-camera local track means.
    Uses build_track_embeds(camera_dir) from crosscam_stitch.
    """
    gid_to_vecs: Dict[int, List[np.ndarray]] = defaultdict(list)

    for cam_dir in sorted(Path(out_root).glob("*")):
        if not cam_dir.is_dir():
            continue
        cam = cam_dir.name
        l2g = crosscam_map.get(cam, {})
        if not l2g:
            continue
        # build_track_embeds expects a camera directory; it should return local_id-> {"emb":vec, "start":..,"end":..}
        try:
            track_dict = build_track_embeds(cam_dir)
        except Exception as e:
            logging.warning(f"[Consolidate] build_track_embeds failed for {cam}: {e}")
            continue
        for lid, stats in track_dict.items():
            try:
                gid = l2g.get(str(lid))
                if gid is None:
                    continue
                emb = stats.get("emb")
                if emb is None:
                    continue
                v = np.asarray(emb, dtype=np.float32).reshape(-1)
                if np.isfinite(v).all() and np.linalg.norm(v) > 1e-6:
                    gid_to_vecs[int(gid)].append(v)
            except Exception:
                continue

    # Mean per gid
    gid_means: Dict[int, np.ndarray] = {}
    for gid, vecs in gid_to_vecs.items():
        if not vecs:
            continue
        gid_means[gid] = np.mean(np.stack(vecs), axis=0)
    return gid_means


def consolidate_global_ids(out_root: Path, sim_threshold: float = 0.85,
                           min_actors: int = 6, max_actors: int = 10):
    """
    Merge redundant global IDs adaptively based on cosine similarity.
    If merged records lack embeddings, fallback to per-camera mean embeddings.
    Also rewrites per-camera *_poses_global.jsonl to use final consolidated IDs.
    """
    out_root = Path(out_root)
    merged_path = out_root / "merged_global_tracks.jsonl"
    if not merged_path.exists():
        logging.warning("No merged_global_tracks.jsonl found, skipping consolidation.")
        return

    # Load crosscam_map to allow fallback means
    map_path = out_root / "crosscam_map.json"
    if not map_path.exists():
        logging.warning("crosscam_map.json missing; cannot consolidate.")
        return
    with open(map_path, "r") as f:
        crosscam_map = json.load(f)

    # Try to get embeddings directly from merged file (if present)
    emb_dict = defaultdict(list)
    all_records = []
    with open(merged_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            all_records.append(rec)
            gid = rec.get("global_id")
            emb = rec.get("embedding") or rec.get("emb") or rec.get("embeds")
            if emb is not None:
                v = np.asarray(emb, dtype=np.float32).reshape(-1)
                if np.isfinite(v).all() and np.linalg.norm(v) > 1e-6:
                    emb_dict[gid].append(v)

    # Fallback: build per-gid means from per-camera local track means
    if not emb_dict:
        gid_means = _fallback_gid_means_from_percamera(out_root, crosscam_map)
        if not gid_means:
            logging.warning("No embeddings found for consolidation; aborting.")
            return
        gids = sorted(gid_means.keys())
        mean_embs = np.stack([gid_means[g] for g in gids])
    else:
        gids, mean_embs = [], []
        for gid, vecs in emb_dict.items():
            gids.append(gid)
            mean_embs.append(np.mean(np.stack(vecs), axis=0))
        mean_embs = np.stack(mean_embs)

    # If using fallback, gids may be unset
    if not emb_dict:
        # gids assigned above from gid_means path
        pass

    sims = cosine_similarity(mean_embs)
    # Build groups by similarity, auto-tune threshold to land 6–10 groups
    def make_groups(thr: float):
        visited, groups = set(), []
        for i, g in enumerate(gids):
            if g in visited:
                continue
            grp = {g}
            visited.add(g)
            for j, g2 in enumerate(gids):
                if g2 in visited:
                    continue
                if sims[i, j] >= thr:
                    grp.add(g2)
                    visited.add(g2)
            groups.append(grp)
        return groups

    thr = sim_threshold
    for _ in range(30):
        groups = make_groups(thr)
        if len(groups) < min_actors and thr > 0.60:
            thr -= 0.05
        elif len(groups) > max_actors and thr < 0.95:
            thr += 0.02
        else:
            break

    gid_remap = {}
    for new_gid, grp in enumerate(sorted(groups, key=lambda s: min(s)), start=1):
        for old in grp:
            gid_remap[int(old)] = int(new_gid)

    # Rewrite merged file (consolidated)
    consolidated_path = out_root / "merged_global_tracks_consolidated.jsonl"
    with open(consolidated_path, "w") as fw:
        for rec in all_records:
            g = rec.get("global_id")
            if g in gid_remap:
                rec["global_id"] = gid_remap[g]
            fw.write(json.dumps(rec) + "\n")
    logging.info(f"[Consolidate] Wrote → {consolidated_path}")

    # Rewrite crosscam_map (consolidated)
    with open(map_path, "r") as f:
        cmap = json.load(f)
    for cam, idmap in cmap.items():
        for lid, gid in list(idmap.items()):
            if int(gid) in gid_remap:
                cmap[cam][lid] = gid_remap[int(gid)]
    new_map_path = out_root / "crosscam_map_consolidated.json"
    with open(new_map_path, "w") as f:
        json.dump(cmap, f, indent=2)
    logging.info(f"[Consolidate] Wrote → {new_map_path}")

    # Rewrite per-camera *_poses_global.jsonl with consolidated IDs (so colors are stable)
    logging.info("[Consolidate] Rewriting per-camera *_poses_global.jsonl ...")
    for cam_dir in sorted(out_root.glob("*")):
        if not cam_dir.is_dir():
            continue
        jg = next(cam_dir.glob("*_poses_global.jsonl"), None)
        if not jg:
            continue
        tmp_path = cam_dir / (jg.stem + "_final.jsonl")
        with open(jg, "r") as fr, open(tmp_path, "w") as fw:
            for line in fr:
                item = json.loads(line)
                for p in item.get("poses", []):
                    g = p.get("global_id")
                    if g in gid_remap:
                        p["global_id"] = gid_remap[g]
                fw.write(json.dumps(item) + "\n")
        os.replace(tmp_path, jg)
    logging.info("[Consolidate] Per-camera globals updated.")


# ============================================================
#   Main
# ============================================================
def load_config(config_path: str | Path) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("pipeline_config.yaml")

    video_path = Path(cfg["video_path"])
    out_root = Path(cfg["out_dir"])

    detector_cfg = cfg["detector"]
    pose_cfg = cfg["pose"]
    reid_cfg = cfg["reid"]
    tracker_cfg = cfg.get("tracker", {})
    pipe_cfg = cfg.get("pipeline", {})

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

    tracker_args = (
        dict(
            sim_thr=float(tracker_cfg.get("sim_thr", 0.55)),
            iou_thr=float(tracker_cfg.get("iou_thr", 0.6)),
            ttl=int(tracker_cfg.get("ttl", 80)),
        )
        if tracker_cfg.get("type", "none") == "strongsort"
        else {}
    )

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

    # --- Run per-camera pipelines ---
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

    if pipe_cfg.get("save_json", True):
        logging.info("Running cross-camera stitching and merging...")
        crosscam_map = run_crosscam_stitch(out_root, sim_thr=0.55, time_win=900)
        if crosscam_map:
            # Use our local, compatible merge (overrides imported one)
            merge_tracks(out_root, crosscam_map, out_root / "merged_global_tracks.jsonl")
            inject_global_ids(out_root, crosscam_map)
            # Adaptive consolidation (6–10) + rewrite per-camera globals
            consolidate_global_ids(out_root, sim_threshold=0.85, min_actors=6, max_actors=10)
            logging.info("✅ Global merge + consolidation complete.")
        else:
            logging.warning("⚠️ Cross-camera map not generated, skipping merge.")

    logging.info("✅ Full pipeline complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    main()
