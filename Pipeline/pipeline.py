# Pipeline/pipeline.py
from __future__ import annotations
import logging, json, gc, time
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

from io_utils import VideoReader, VideoWriter, JsonlWriter
from common import ensure_dir, safe_stem
from detectors import build_detector
from pose import build_pose_estimator, AlphaPoseCfg
from reid import build_reid
from tracking import build_tracker
from visualizer import draw_bbox_and_id, draw_skeleton, visualize_heatmap
from crosscam_stitch import stitch


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
    #   Main per-video run
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

                # 1. Detection
                boxes_xyxy = self.detector(frame_bgr)
                if boxes_xyxy is None or getattr(boxes_xyxy, "size", 0) == 0:
                    vw.write(frame_bgr)
                    continue

                # 2. Pose estimation
                kpts_list, pose_scores, heatmaps = self.pose(frame_bgr, boxes_xyxy)
                if not kpts_list:
                    vw.write(frame_bgr)
                    continue

                # 3. Optional heatmaps
                if self.debug_heatmaps and heatmaps is not None:
                    for i, hm in enumerate(heatmaps):
                        try:
                            overlay = visualize_heatmap(frame_bgr, hm, bbox=boxes_xyxy[i])
                            cv2.imwrite(
                                str(out_dir / "debug_heatmaps" / f"frame{idx}_person{i}.png"), overlay
                            )
                        except Exception as e:
                            self.log.debug(f"Heatmap save failed: frame {idx} person {i} {e}")

                # 4. Re-ID embeddings
                embeds = None
                if hasattr(self.reid, "is_ready") and self.reid.is_ready():
                    emb_list = self.reid(frame_bgr, boxes_xyxy)
                    try:
                        embeds = np.vstack(
                            [e if e is not None else np.zeros((1,), np.float32) for e in emb_list]
                        )
                    except Exception:
                        embeds = None

                # 5. Tracking
                ids = (
                    self.tracker.update(boxes_xyxy, embeds)
                    if self.tracker
                    else list(range(1, boxes_xyxy.shape[0] + 1))
                )

                # 6. Draw skeletons
                annotated = frame_bgr.copy()
                if self.draw:
                    for i, box in enumerate(boxes_xyxy):
                        score_i = float(pose_scores[i]) if pose_scores is not None else None
                        draw_bbox_and_id(annotated, box, track_id=ids[i], score=score_i)
                        draw_skeleton(
                            annotated,
                            np.asarray(kpts_list[i], dtype=np.float32),
                            kpt_thresh=self.kpt_thresh,
                            dataset=self.dataset_hint,
                            draw_face=False,
                            draw_hands=False,  # no hand inference
                            bbox=box,
                        )

                vw.write(annotated)

                # 7. Write JSONL
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

                if frame_count % 100 == 0:
                    self.log.info(f"{video_path.name}: processed {frame_count} frames")

        finally:
            rdr.close()
            vw.close()
            if jw:
                jw.close()
            self.log.info(f"Finished {video_path} | frames={frame_count}")


# ============================================================
#   YAML loader + main batch launcher
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

    # --- Batch mode: folder or single file ---
    if video_path.is_dir():
        video_list = sorted(video_path.glob("*.mp4"))
    else:
        video_list = [video_path]

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
                max_frames=(
                    None
                    if int(pipe_cfg.get("max_frames", 0)) == 0
                    else int(pipe_cfg.get("max_frames"))
                ),
            )
        except Exception as e:
            logging.exception(f"Error processing {v.name}: {e}")

    # --- Cleanup & cross-camera stitching ---
    cv2.destroyAllWindows()
    gc.collect()
    time.sleep(1.0)

    if pipe_cfg.get("save_json", True):
        logging.info("Running cross-camera stitching...")
        crosscam_map = stitch(root=Path(out_root), sim_thr=0.55, time_win=900)
        with open(Path(out_root) / "crosscam_map.json", "w", encoding="utf-8") as f:
            json.dump(crosscam_map, f, indent=2)
        logging.info(f"Cross-camera map written to {out_root}/crosscam_map.json")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    main()
