from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Optional
import numpy as np
import av
import cv2

from common import ensure_dir, safe_stem
from detectors import build_detector
from pose import build_pose_estimator, AlphaPoseCfg
from reid import build_reid
from tracking import build_tracker
from visualizer import draw_bbox_and_id, draw_skeleton, visualize_heatmap
from crosscam_stitch import stitch


# --------------------------
# VideoReader using PyAV
# --------------------------
class VideoReader:
    """
    VideoReader using PyAV to support HEVC/10-bit videos.
    Yields frames as 8-bit BGR numpy arrays.
    """
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self.container = av.open(str(self.video_path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

    def __iter__(self):
        frame_idx = 0
        for frame in self.container.decode(video=0):
            img = frame.to_ndarray(format='bgr24')  # converts to 8-bit BGR
            yield frame_idx, img
            frame_idx += 1

    def close(self):
        self.container.close()


# --------------------------
# Helper to convert ndarray -> list recursively
# --------------------------
def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [ndarray_to_list(o) for o in obj]
    return obj


# --------------------------
# SequentialPipeline
# --------------------------
class SequentialPipeline:
    def __init__(self,
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
                 debug_heatmaps: bool = True,
                 hand_kpt_thresh: float = 0.25,       # <-- NEW
                 wrist_kpt_thresh: float = 0.25):     # <-- NEW

        self.log = logging.getLogger(self.__class__.__name__)
        self.detector = build_detector(detector_name, **detector_args)
        self.pose = build_pose_estimator(pose_name, **pose_args)
        self.reid = build_reid(reid_name, **(reid_args or {}))
        self.tracker = build_tracker(tracker_name, **(tracker_args or {}))
        self.draw = draw
        self.kpt_thresh = float(kpt_thresh)
        self.dataset_hint = dataset_hint
        self.debug_heatmaps = debug_heatmaps

        # store the new thresholds
        self.hand_kpt_thresh = float(hand_kpt_thresh)
        self.wrist_kpt_thresh = float(wrist_kpt_thresh)


    def run(self, video_path: str | Path, out_dir: str | Path,
            save_video: bool, save_json: bool, save_embeds: bool,
            max_frames: Optional[int] = None):

        video_path = Path(video_path)
        out_dir = Path(out_dir)
        ensure_dir(out_dir)

        rdr = VideoReader(video_path)
        base = safe_stem(video_path)

        # Video writer
        vw = None
        if save_video:
            first_frame = next(iter(rdr))[1]
            vw = cv2.VideoWriter(
                str(out_dir / f"{base}_annot.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,  # default FPS
                (first_frame.shape[1], first_frame.shape[0])
            )
            rdr = VideoReader(video_path)  # reset iterator

        # JSONL writer
        jw = open(out_dir / f"{base}_poses.jsonl", "w") if save_json else None

        # Heatmap debug folder
        if self.debug_heatmaps:
            heatmap_dir = out_dir / "debug_heatmaps"
            heatmap_dir.mkdir(exist_ok=True)

        frame_count = 0
        try:
            for idx, frame_bgr in rdr:
                if frame_bgr is None:
                    self.log.warning(f"Frame {idx} is None, skipping...")
                    continue

                frame_count += 1
                if max_frames and frame_count > max_frames:
                    break

                # 1) Detection
                boxes_xyxy = self.detector(frame_bgr)
                print("Detected boxes:", boxes_xyxy)
                if boxes_xyxy is None or boxes_xyxy.size == 0:
                    if vw: vw.write(frame_bgr)
                    continue

                # 2) Pose
                # If your pose estimator returns heatmaps as third output
                kpts_list, pose_scores, heatmaps = self.pose(frame_bgr, boxes_xyxy)
                print("Pose scores:", pose_scores)
                print("Keypoints length:", kpts_list[0].shape if kpts_list else 0)
                for idx, (x, y, c) in enumerate(kpts_list[0][:25]):  # first 25 points
                    print(f"  idx={idx}: ({x:.1f},{y:.1f}) conf={c:.2f}")
                if not kpts_list:
                    if vw: vw.write(frame_bgr)
                    continue

                # Optional: save heatmaps
                if self.debug_heatmaps and heatmaps is not None:
                    for i, hm in enumerate(heatmaps):
                        overlay = visualize_heatmap(frame_bgr, hm, bbox=boxes_xyxy[i])
                        cv2.imwrite(str(heatmap_dir / f"frame{idx}_person{i}.png"), overlay)

                # 3) ReID embeddings
                embeds = None
                if hasattr(self.reid, "is_ready") and self.reid.is_ready():
                    emb_list = self.reid(frame_bgr, boxes_xyxy)
                    try:
                        embeds = np.vstack([e if e is not None else np.zeros((1,), np.float32) for e in emb_list])
                    except Exception:
                        embeds = None

                # 4) Tracking
                ids = None
                if self.tracker:
                    ids = self.tracker.update(boxes_xyxy, embeds)
                else:
                    ids = list(range(1, boxes_xyxy.shape[0] + 1))

                # 5) Draw & Write
                annotated = frame_bgr.copy()
                if self.draw:
                    for i, box in enumerate(boxes_xyxy):
                        draw_bbox_and_id(annotated, box, track_id=ids[i], score=pose_scores[i])
                        kpts = np.asarray(kpts_list[i], dtype=np.float32)
                        draw_skeleton(annotated, kpts, 
                                      kpt_thresh=self.kpt_thresh, 
                                      dataset=self.dataset_hint,
                                      draw_face=False,
                                      draw_hands=True,
                                      bbox=box,
                                      hand_kpt_thresh=self.hand_kpt_thresh,     
                                      wrist_kpt_thresh=self.wrist_kpt_thresh, 
                                      )
                if vw: vw.write(annotated)

                # 6) JSONL
                if jw:
                    rec = {
                        "frame": idx,
                        "cam_id": base,
                        "poses": [{
                            "id": int(ids[i]),
                            "bbox": boxes_xyxy[i].astype(float).tolist(),
                            "keypoints": kpts_list[i].tolist() if isinstance(kpts_list[i], np.ndarray) else kpts_list[i],
                            "score": float(pose_scores[i])
                        } for i in range(len(kpts_list))]
                    }
                    if save_embeds and embeds is not None:
                        rec["embeds"] = embeds.tolist()
                    jw.write(json.dumps(ndarray_to_list(rec)) + "\n")

                if frame_count % 100 == 0:
                    self.log.info("Processed %d frames", frame_count)

        finally:
            rdr.close()
            if vw: vw.release()
            if jw: jw.close()
            self.log.info("Finished %s | frames=%d", str(video_path), frame_count)


# --------------------------
# YAML loader & main
# --------------------------
def load_config(config_path: str | Path) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    cfg = load_config("pipeline_config.yaml")

    video_path = cfg["video_path"]
    out_dir = cfg["out_dir"]

    detector_cfg = cfg["detector"]
    pose_cfg = cfg["pose"]
    reid_cfg = cfg["reid"]
    tracker_cfg = cfg.get("tracker", {})
    pipe_cfg = cfg.get("pipeline", {})

    detector_args = dict(
        model=detector_cfg["model"],
        weights=detector_cfg["weights"],
        device=detector_cfg.get("device", "cuda"),
        conf=detector_cfg.get("conf", 0.5),
        iou=detector_cfg.get("iou", 0.6),
        imgsz=detector_cfg.get("imgsz", 640),
        half=detector_cfg.get("half", False)
    )

    pose_args = dict(config=AlphaPoseCfg(
        input_size=tuple(pose_cfg.get("input_size", [256,192])),
        device=pose_cfg.get("device", "cuda"),
        cfg_yaml=pose_cfg["cfg"],
        checkpoint=pose_cfg["ckpt"],
        fp16=pose_cfg.get("fp16", False)
    ))

    reid_args = dict(
        weights=reid_cfg["weights"],
        device=reid_cfg.get("device", "cuda")
    ) if reid_cfg["type"] != "none" else {}

    tracker_args = dict(
        sim_thr=tracker_cfg.get("sim_thr", 0.55),
        iou_thr=tracker_cfg.get("iou_thr", 0.6),
        ttl=tracker_cfg.get("ttl", 80)
    ) if tracker_cfg.get("type", "none") == "strongsort" else {}

    pipe = SequentialPipeline (
        detector_name=detector_cfg["type"], detector_args=detector_args,
        pose_name=pose_cfg["type"], pose_args=pose_args,
        reid_name=reid_cfg["type"], reid_args=reid_args,
        tracker_name=tracker_cfg.get("type", "none"), tracker_args=tracker_args,
        draw=pipe_cfg.get("draw", True),
        kpt_thresh=pipe_cfg.get("kpt_thresh", 0.2),
        hand_kpt_thresh   = pipe_cfg.get( "hand_kpt_thresh", 0.25),
        wrist_kpt_thresh = pipe_cfg.get( "wrist_kpt_thresh", 0.25),
        dataset_hint= pipe_cfg.get("kp_set", "coco_wholebody"),
        debug_heatmaps= pipe_cfg.get("debug_heatmaps", False)
    )

    pipe.run(
        video_path=video_path,
        out_dir=out_dir,
        save_video=pipe_cfg.get("save_video", True),
        save_json=pipe_cfg.get("save_json", True),
        save_embeds=pipe_cfg.get("save_embeds", True),
        max_frames=(None if pipe_cfg.get("max_frames", 0) == 0 else pipe_cfg.get("max_frames"))
    )

    # Cross-camera stitching
    if pipe_cfg.get("save_json", True):
        logging.info("Running cross-camera stitching...")
        crosscam_map = stitch(root=Path(out_dir), sim_thr=0.55, time_win=900)
        with open(Path(out_dir) / "crosscam_map.json", "w", encoding="utf-8") as f:
            json.dump(crosscam_map, f, indent=2)
        logging.info(f"Cross-camera map written to {out_dir}/crosscam_map.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
