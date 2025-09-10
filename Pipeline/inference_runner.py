#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Optional
import json, logging, cv2

from AlphaPose_OSNet_Pipeline.utils.io_utils import ensure_dir, video_writer, jsonl_writer, safe_stem
from AlphaPose_OSNet_Pipeline.utils.visualizer import draw_bbox_and_id, draw_skeleton_coco
from .alphapose_wrapper import AlphaPoseRunner
from .osnet_wrapper import OSNetExtractor
from .strongsort_tracker import StrongSORTTracker
from .yolov8_wrapper import YOLOv8Detector

log = logging.getLogger(__name__)

class InferenceRunner:
    def __init__(self,
        alphapose: AlphaPoseRunner,
        osnet: Optional[OSNetExtractor] = None,
        tracker: Optional[StrongSORTTracker] = None,
        detector: str = "alphapose",     # "alphapose" | "yolov8"
        y8: Optional[YOLOv8Detector] = None,
        draw: bool = True,
        save_embeds: bool = False,
    ):
        self.ap = alphapose
        self.osn = osnet
        self.trk = tracker
        self.detector = detector.lower().strip()
        self.y8 = y8
        self.draw = bool(draw)
        self.save_embeds = bool(save_embeds)

    def process_frame(self, frame_bgr):
        if self.detector == "yolov8":
            boxes_full = self.y8.detect(frame_bgr) if self.y8 else []
            boxes = [[b[0], b[1], b[2], b[3]] for b in boxes_full]
            kpts, pose_scores = self.ap.pose_on_boxes(frame_bgr, boxes_full)
            feats = self.osn.extract(frame_bgr, boxes) if (self.osn and boxes) else [None]*len(boxes)
            ids = self.trk.update(boxes, feats) if (self.trk and boxes) else list(range(len(boxes)))
            return {"boxes": boxes, "keypoints": kpts, "scores": pose_scores, "ids": ids, "feats": feats}
        else:
            boxes, kpts, scores, _ = self.ap.infer(frame_bgr, use_internal_detector=True)
            feats = self.osn.extract(frame_bgr, boxes) if (self.osn and boxes) else [None]*len(boxes)
            ids = self.trk.update(boxes, feats) if (self.trk and boxes) else list(range(len(boxes)))
            return {"boxes": boxes, "keypoints": kpts, "scores": scores, "ids": ids, "feats": feats}

    def run_on_video(self, video_path: Path, out_dir: Path, cam_id: str,
                     save_video: bool = True, save_json: bool = True,
                     start_frame: int = 0, max_frames: Optional[int] = None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): log.error("Failed to open %s", video_path); return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        ensure_dir(out_dir)

        vw = video_writer(out_dir / f"{safe_stem(video_path)}_annotated.mp4", fps, (w,h)) if save_video else None
        jl = jsonl_writer(out_dir / f"{safe_stem(video_path)}_poses.jsonl") if save_json else None

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        done = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            out = self.process_frame(frame)
            boxes, kpts, scores, ids, feats = out["boxes"], out["keypoints"], out["scores"], out["ids"], out["feats"]

            if self.draw:
                for i, b in enumerate(boxes):
                    pid = ids[i] if i < len(ids) else i
                    draw_bbox_and_id(frame, b, pid=pid)
                    if i < len(kpts):
                        draw_skeleton_coco(frame, kpts[i], pid=pid)

            if jl is not None:
                rec = {"frame": frame_idx, "cam_id": cam_id, "poses": []}
                for i, b in enumerate(boxes):
                    pts = kpts[i] if i < len(kpts) else []
                    pid = ids[i] if i < len(ids) else i
                    pose = {
                        "id": int(pid),
                        "bbox": [float(v) for v in b],
                        "keypoints": [[float(x), float(y), float(s)] for (x, y, s) in pts],
                        "score": float(scores[i]) if i < len(scores) else 0.0,
                    }
                    if self.save_embeds and i < len(feats) and feats[i] is not None:
                        pose["emb"] = [float(v) for v in feats[i].tolist()]
                    rec["poses"].append(pose)
                jl.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if vw is not None: vw.write(frame)
            done += 1
            if max_frames is not None and done >= max_frames: break

        cap.release()
        if vw is not None: vw.release()
        if jl is not None: jl.close()
        log.info("Finished %s frames=%d", video_path, done)
